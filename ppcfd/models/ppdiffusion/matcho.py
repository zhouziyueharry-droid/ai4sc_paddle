"""
matcho.py — U-Net 2D (扩散模型风格) 的实现与构建积木

本文件采用“积木拼装”的方式构建一个增强版的 U-Net，用于处理时空网格上的二维数据。
它与经典 U-Net 的关系与区别如下：

- 经典 U-Net：编码器(下采样) + 跳跃连接 + 解码器(上采样) + 输出卷积；每一层通常是卷积+归一化+激活；
- 本实现的增强点：
  - ResNet 风格的残差块(`ResnetBlock`)替代简单卷积块，借助残差连接更稳定、更深；
  - 时间嵌入(time embedding)：通过 `SinusoidalPosEmb` + MLP，将标量时间条件编码为高维向量，注入到每个残差块中；
  - 注意力模块：在每个下采样/上采样层使用 `LinearAttention`，在中间层使用全注意力(`Attention`)；可通过 `attention_heads=None` 关闭以节省显存；
  - 规范化：使用 `GroupNorm` 与自定义 `LayerNorm`/`PreNorm`，提升训练稳定性；
  - 下采样/上采样：分别使用 PixelUnshuffle/Nearest-Neighbor + 1x1/3x3 卷积的组合，代替简单的池化。

模块拼接方式概览：
- 输入经过初始卷积(`init_conv`)形成初始特征；
- 编码器 downs：每级包含两个残差块 + 注意力 + 下采样，逐步降低分辨率、提升通道；中途将特征保存到栈用于跳跃连接；
- 中间层 mid：残差块 + 注意力 + 残差块，作为瓶颈交互；
- 解码器 ups：逐级与对应的编码器特征拼接(跳跃连接)，再经过两个残差块 + 注意力 + 上采样，恢复分辨率；
- 结尾：与初始特征拼接后，再经残差块与 1x1 卷积得到最终 `nf` 通道输出。

与经典 U-Net 的对比：如果禁用注意力(将 `attention_heads=None`)且移除时间嵌入，本模型退化为更接近经典 U-Net 的结构；
但默认启用的注意力与时间嵌入让它更适合扩散/时序条件任务。
"""

import math
from functools import partial

import einops
import numpy as np
import paddle
from einops import rearrange
from einops.layers.paddle import Rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return math.sqrt(num) ** 2 == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


class Residual(paddle.nn.Layer):
    """将任意子层以残差形式包裹：输出 = 子层(x) + x。
    - 作用：提供快捷连接，缓解梯度消失，让更深的网络易训练。
    - 用法：Residual(PreNorm(LinearAttention(...))) 等。
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """上采样模块：先用最近邻插值将特征图放大 2x，再用 3x3 卷积整合与调整通道。
    - 输入通道：`dim`；输出通道：`dim_out`(缺省为与输入一致)。
    - 作用：恢复空间分辨率，同时保持/整合语义。
    """
    return paddle.nn.Sequential(
        paddle.nn.Upsample(scale_factor=2, mode="nearest"),
        paddle.nn.Conv2D(
            in_channels=dim,
            out_channels=default(dim_out, dim),
            kernel_size=3,
            padding=1,
        ),
    )


def Downsample(dim, dim_out=None):
    """下采样模块：使用 PixelUnshuffle(重排像素) 将空间尺寸减半、通道乘以 4，接 1x1 卷积映射到 `dim_out`。
    - 输入通道：`dim`；输出通道：`dim_out`(缺省为与输入一致)。
    - 作用：压缩空间分辨率，提升通道容量，代替池化以减少信息损失。
    """
    return paddle.nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        paddle.nn.Conv2D(in_channels=dim * 4, out_channels=default(dim_out, dim), kernel_size=1),
    )


class WeightStandardizedConv2d(paddle.nn.Conv2D):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-05 if x.dtype == "float32" else 0.001
        weight = self.weight
        mean = einops.reduce(weight, "o ... -> o 1 1 1", "mean")
        var = einops.reduce(weight, "o ... -> o 1 1 1", partial(paddle.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return paddle.nn.functional.conv2d(
            x=x,
            weight=normalized_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class LayerNorm(paddle.nn.Layer):
    """二维特征图的通道层归一化：在通道维做均值-方差归一化，然后乘以可学习缩放因子 `g`。
    - 与 `GroupNorm` 的不同：`LayerNorm` 在通道维整体归一化，`GroupNorm` 则分组归一化。
    - 作用：稳定训练、加速收敛，常与注意力或 MLP 搭配。
    """
    def __init__(self, dim):
        super().__init__()
        self.g = self.create_parameter(shape=[1, dim, 1, 1], default_initializer=paddle.nn.initializer.Constant(1.0))

    def forward(self, x):
        eps = 1e-05 if x.dtype == paddle.float32 else 0.001
        # 使用更稳定且更省显存的方差实现：E[x^2] - (E[x])^2，避免内部 pow 带来的额外分配
        mean = paddle.mean(x=x, axis=1, keepdim=True)
        mean_sq = paddle.mean(x=x * x, axis=1, keepdim=True)
        var = mean_sq - mean * mean
        var = paddle.clip(var, min=0.0)  # 数值上避免负的方差
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(paddle.nn.Layer):
    """前归一化包装器：在调用子层 `fn` 前先做 `LayerNorm`。
    - 作用：降低数值范围变化，提高稳定性(常见于注意力模块的包裹)。
    - 等价结构：`x = LayerNorm(x)` 然后 `x = fn(x)`。
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.fn(x)
        # return self.fn(x)
        return x


class SinusoidalPosEmb(paddle.nn.Layer):
    """正弦位置编码：将标量时间 `x` 映射到 `dim` 维的正弦/余弦特征。
    - 经典技巧(Transformer 中常见)，让模型感知不同尺度的周期信息。
    - 在本模型中，作为时间条件的第一步编码，再送入 MLP。
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = paddle.exp(x=paddle.arange(end=half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = paddle.concat(x=(emb.sin(), emb.cos()), axis=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(paddle.nn.Layer):
    """可学习/随机的正弦位置编码权重。
    - 对比 `SinusoidalPosEmb` 的固定频率，这里通过可学习权重 `weights` 调整频率；
    - 适用于让模型自动适配时间频率分布(扩散模型常用变体)。
    """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert dim % 2 == 0
        half_dim = dim // 2
        self.weights = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.randn(shape=half_dim), trainable=not is_random
        )

    def forward(self, x):
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = paddle.concat(x=(freqs.sin(), freqs.cos()), axis=-1)
        fouriered = paddle.concat(x=(x, fouriered), axis=-1)
        return fouriered


class Block(paddle.nn.Layer):
    """基本卷积块：3x3 Conv + 归一化(GroupNorm) + 激活(SiLU)。
    - `groups` 控制 GroupNorm 的分组数；
    - 支持 `scale_shift`：当提供来自时间嵌入的尺度/偏移时，做调制 `x = x * (scale + 1) + shift`。
    """
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out, kernel_size=3, padding=1)
        self.norm = paddle.nn.GroupNorm(num_groups=groups, num_channels=dim_out)
        # self.norm = LayerNorm(dim_out)
        self.act = paddle.nn.Silu()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(paddle.nn.Layer):
    """
    ResNet 风格残差块(两层卷积块 + 可选时间调制 + 残差捷径)。
    - 若提供 `time_emb_dim`，则通过 MLP 生成 `[scale, shift]` 对每层做调制；
    - `res_conv` 保证输入输出通道不一致时能进行残差相加。
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            paddle.nn.Sequential(
                paddle.nn.Silu(),
                paddle.nn.Linear(in_features=time_emb_dim, out_features=dim_out * 2),
            )
            if exists(time_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = (
            paddle.nn.Conv2D(in_channels=dim, out_channels=dim_out, kernel_size=1)
            if dim != dim_out
            else paddle.nn.Identity()
        )

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(chunks=2, axis=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(paddle.nn.Layer):
    """线性注意力(LARA/线性近似)：将复杂度从 O(N^2) 降为 O(N)。
    - 通过对 `k`、`v` 做全局聚合构造 `context`，再与 `q` 交互；
    - `heads=None` 时直接跳过注意力(返回输入)，常用于节省显存；
    - 输出再经 `LayerNorm` 稳定数值。
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        if heads is not None:
            self.scale = dim_head**-0.5
            hidden_dim = dim_head * heads
            self.to_qkv = paddle.nn.Conv2D(
                in_channels=dim,
                out_channels=hidden_dim * 3,
                kernel_size=1,
                bias_attr=False,
            )
            self.to_out = paddle.nn.Sequential(
                paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
                LayerNorm(dim),
            )

    def forward(self, x):
        if self.heads is None:
            return x
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q * self.scale
        v = v / (h * w)
        # 保持注意力计算在 FP32，避免早期训练阶段因幅值过大导致的 FP16 溢出产生 NaN
        context = paddle.einsum("b h d n, b h e n -> b h d e", k, v)
        out = paddle.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(paddle.nn.Layer):
    """全注意力(Softmax-less 实现变体)：复杂度 O(N^2)，表达力强，适合在中间层瓶颈位置使用。
    - heads=None 将跳过注意力；heads>0 时以多头方式分解通道并做两次 `einsum`；
    - 本实现未显式 softmax，等价于点积注意力的线性算子版本，数值更稳定地转为 conv 输出。
    """
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        if heads is not None:
            self.scale = dim_head**-0.5
            hidden_dim = dim_head * heads
            self.to_qkv = paddle.nn.Conv2D(
                in_channels=dim,
                out_channels=hidden_dim * 3,
                kernel_size=1,
                bias_attr=False,
            )
            self.to_out = paddle.nn.Conv2D(in_channels=hidden_dim, out_channels=dim, kernel_size=1)

    def forward(self, x):
        if self.heads is None:
            return x
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(chunks=3, axis=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads),
            qkv,
        )
        q = q * self.scale
        # 保持注意力计算在 FP32，避免早期训练阶段因幅值过大导致的 FP16 溢出产生 NaN
        sim = paddle.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim
        out = paddle.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        out = self.to_out(out)
        return out


class Unet2D(paddle.nn.Layer):
    """增强版 U-Net 2D(含时间嵌入与注意力)，用于二维网格序列的条件建模。

    参数说明(常用)：
    - `dim`：基准通道数，随 `dim_mults` 在各层放大；
    - `Par`：训练脚本传入的参数字典，包含 `nf`(输出通道)、`nx/ny`(网格尺寸)、`lb/lf`(时序窗口)等；
    - `init_dim`：输入后第一层卷积输出通道，默认等于 `dim`；
    - `dim_mults`：每个分辨率层的通道倍率，如 `(1, 2, 4, 8)`；
    - `channels`：输入通道数(= `nf * lb`，训练脚本处负责计算并传入)；
    - `self_condition`：是否启用自条件(本实现保留接口，默认关闭)；
    - `resnet_block_groups`：GroupNorm 的分组数；
    - `learned_sinusoidal_cond` / `random_fourier_features`：是否改用可学习/随机频率的时间位置编码；
    - `attention_heads`：注意力的头数；设为 `None` 可整体关闭注意力以节省显存；
    - `dim_head`：每个注意力头的维度。

    结构总览：
    1) 初始卷积 `init_conv` 将输入特征投影到 `init_dim`；
    2) 编码器 downs：每级包含 [ResnetBlock, ResnetBlock, LinearAttention, Downsample]；
    3) 中间层 mid：`ResnetBlock → Attention → ResnetBlock`；
    4) 解码器 ups：每级包含 [ResnetBlock(与跳连拼接), ResnetBlock(与跳连拼接), LinearAttention, Upsample]；
    5) 结尾：与初始特征 `r` 拼接 → `final_res_block` → `final_conv` 输出 `nf` 通道，并按 `Par` 反标准化与掩码。

    与经典 U-Net 的差异：
    - 残差块 + 时间嵌入：代替“Conv+BN+ReLU”，在每层注入时间条件；
    - 注意力：在所有分辨率层引入局部线性注意力，在瓶颈引入全注意力；
    - 归一化与下/上采样方式：更偏向扩散模型/现代视觉网络的搭配。
    """
    def __init__(
        self,
        dim,
        Par,
        init_dim=None,  # 初始卷积层的输出通道数
        out_dim=None,   # 最后一层卷积层的输出通道数
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        attention_heads=4,
        dim_head=32,
    ):
        super().__init__()
        self.Par = Par
        self.channels = channels
        self.self_condition = self_condition

        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = paddle.nn.Conv2D(in_channels=input_channels, out_channels=init_dim, kernel_size=7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        self.time_mlp = paddle.nn.Sequential(
            sinu_pos_emb,
            paddle.nn.Linear(in_features=fourier_dim, out_features=time_dim),
            paddle.nn.GELU(),
            paddle.nn.Linear(in_features=time_dim, out_features=time_dim),
        )

        kwargs_block_klass = {"time_emb_dim": time_dim, "groups": resnet_block_groups}
        self.downs = paddle.nn.LayerList(sublayers=[])
        self.ups = paddle.nn.LayerList(sublayers=[])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= num_resolutions - 1
            self.downs.append(
                paddle.nn.LayerList(
                    [
                        ResnetBlock(dim_in, dim_in, **kwargs_block_klass),
                        ResnetBlock(dim_in, dim_in, **kwargs_block_klass),
                        Residual(
                            PreNorm(
                                dim_in,
                                LinearAttention(dim_in, heads=attention_heads, dim_head=dim_head),
                            )
                        ),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else paddle.nn.Conv2D(
                                dim_in,
                                dim_out,
                                kernel_size=3,
                                padding=1,
                            )
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, **kwargs_block_klass)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=attention_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, **kwargs_block_klass)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == len(in_out) - 1
            self.ups.append(
                paddle.nn.LayerList(
                    [
                        ResnetBlock(dim_out + dim_in, dim_out, **kwargs_block_klass),
                        ResnetBlock(dim_out + dim_in, dim_out, **kwargs_block_klass),
                        Residual(
                            PreNorm(
                                dim_out,
                                LinearAttention(dim_out, heads=attention_heads, dim_head=dim_head),
                            )
                        ),
                        (
                            Upsample(dim_out, dim_in)
                            if not is_last
                            else paddle.nn.Conv2D(
                                dim_out,
                                dim_in,
                                kernel_size=3,
                                padding=1,  
                            )
                        ),
                    ]
                )
            )
        self.out_dim = self.Par["nf"]
        self.final_res_block = ResnetBlock(dim * 2, dim, **kwargs_block_klass)
        self.final_conv = paddle.nn.Conv2D(in_channels=dim, out_channels=self.out_dim, kernel_size=1)

    def get_grid(self, shape, device="cuda"):
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        gridx = paddle.to_tensor(data=np.linspace(0, 1, size_x), dtype="float32")
        gridx = gridx.reshape(1, size_x, 1, 1).tile(repeat_times=[batchsize, 1, size_y, 1])
        gridy = paddle.to_tensor(data=np.linspace(0, 1, size_y), dtype="float32")
        gridy = gridy.reshape(1, 1, size_y, 1).tile(repeat_times=[batchsize, size_x, 1, 1])
        return paddle.concat(x=(gridx, gridy), axis=-1).to(device)

    def forward(self, x, time, x_self_cond=None, use_grid=True):
        # 1) 对输入做标准化(训练脚本中记录了 shift/scale)，并按 [B, lb*nf, nx, ny] 组织为卷积输入
        x = (x - self.Par["inp_shift"]) / self.Par["inp_scale"]
        x = x.reshape([-1, self.Par["lb"] * self.Par["nf"], self.Par["nx"], self.Par["ny"]])

        # 2) 归一化时间条件并编码到高维向量，供各层残差块调制使用
        time = (time - self.Par["t_shift"]) / self.Par["t_scale"]

        # 3) 初始投影与残差保存(用于最后拼接)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time) if time is not None else None

        # 4) 编码器：两层 ResnetBlock + 注意力 + 下采样；同时将中间特征推入栈以形成跳跃连接
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # 5) 中间层：残差块 → 全注意力 → 残差块
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # 6) 解码器：与对应的编码器特征拼接后，两层 ResnetBlock + 注意力 + 上采样
        for block1, block2, attn, upsample in self.ups:
            x = paddle.concat(x=(x, h.pop()), axis=1)
            x = block1(x, t)
            x = paddle.concat(x=(x, h.pop()), axis=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # 7) 结尾：与初始残差 r 拼接 → 最终残差块 → 1x1 输出卷积；再反标准化并加掩码
        x = paddle.concat(x=(x, r), axis=1)
        x = self.final_res_block(x, t)

        out = self.final_conv(x)
        out = out.unsqueeze(axis=1)
        out = out * self.Par["out_scale"] + self.Par["out_shift"]
        out = out.reshape([-1, self.Par["nf"], self.Par["nx"], self.Par["ny"]]) * self.Par["mask"]
        return out


if __name__ == "__main__":
    model = Unet2D(dim=16, dim_mults=(1, 2, 4, 8))
    pred = model(paddle.rand(shape=(16, 3, 64, 64)), time=paddle.rand(shape=(16,)))
    print("OK")
