# syntax=docker/dockerfile:1.7
FROM mambaorg/micromamba:1.5.8
WORKDIR /app
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ARG MAMBA_ROOT_PREFIX=/opt/conda

USER root
# 安装系统依赖（Paddle 需要 libgomp.so.1；Open3D 可能需要 libGL.so.1）
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libstdc++6 && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# 预置 condarc，避免运行时 libmamba 解析警告
COPY .condarc ${MAMBA_ROOT_PREFIX}/.condarc

# 创建 Conda 环境
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n ai4sc -f /tmp/environment.yml && micromamba clean --all --yes

# 使用 micromamba run 在指定环境中执行 pip 安装，避免 PATH/激活问题
SHELL ["bash", "-lc"]
# 预创建 pip/mamba 缓存目录，避免运行时路径缺失
RUN mkdir -p /home/mambauser/.cache/pip /home/mambauser/.cache/mamba
# 安装 GPU 版 Paddle（优先尝试 2.6.2，不可用则回退到 2.6.1/2.5.2）。
# 说明：Paddle 的 GPU 轮子仅在官方索引提供，部分版本对 Python 版本/平台有限制。
RUN micromamba run -n ai4sc python -m pip install --timeout 1200 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple -f https://www.paddlepaddle.org.cn/whl/linux/gpu paddlepaddle-gpu==2.6.2 \
 || micromamba run -n ai4sc python -m pip install --timeout 1200 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple -f https://www.paddlepaddle.org.cn/whl/linux/gpu paddlepaddle-gpu==2.6.1 \
 || micromamba run -n ai4sc python -m pip install --timeout 1200 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple -f https://www.paddlepaddle.org.cn/whl/linux/gpu paddlepaddle-gpu==2.5.2 \
 && micromamba run -n ai4sc python -m pip install --timeout 1200 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple pynvml \
 && micromamba run -n ai4sc python -m pip install --timeout 1200 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple nvidia-cudnn-cu11==8.9.6.50 \
 && micromamba run -n ai4sc python -m pip install --timeout 1200 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple nvidia-cublas-cu11 \
 && micromamba run -n ai4sc python -m pip install --timeout 1200 --retries 10 -i https://pypi.tuna.tsinghua.edu.cn/simple nvidia-cufft-cu11 \
 && bash -lc 'CUDNN_LIB_DIR=/opt/conda/envs/ai4sc/lib/python3.10/site-packages/nvidia/cudnn/lib; \
    for so in libcudnn.so.8 libcudnn_ops_infer.so.8 libcudnn_ops_train.so.8 libcudnn_cnn_infer.so.8 libcudnn_cnn_train.so.8; do \
      [ -f "$CUDNN_LIB_DIR/$so" ] && ln -sf "$so" "$CUDNN_LIB_DIR/${so%.so.8}.so"; \
    done' \
 && bash -lc 'CUBLAS_LIB_DIR=/opt/conda/envs/ai4sc/lib/python3.10/site-packages/nvidia/cublas/lib; \
    for so in libcublas.so.11 libcublasLt.so.11; do \
      [ -f "$CUBLAS_LIB_DIR/$so" ] && ln -sf "$so" "$CUBLAS_LIB_DIR/${so%.so.11}.so"; \
    done' \
 && bash -lc 'CUFFT_LIB_DIR=/opt/conda/envs/ai4sc/lib/python3.10/site-packages/nvidia/cufft/lib; \
    for so in libcufft.so.10; do \
      [ -f "$CUFFT_LIB_DIR/$so" ] && ln -sf "$so" "$CUFFT_LIB_DIR/${so%.so.10}.so"; \
    done' \
 && micromamba run -n ai4sc python -c "import paddle; print('Paddle version:', paddle.__version__); print('Compiled with CUDA:', paddle.is_compiled_with_cuda())"

SHELL ["bash", "-lc"]
ENV MAMBA_DEFAULT_ENV=ai4sc
ENV PATH=/opt/conda/envs/ai4sc/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/ai4sc/lib:/opt/conda/envs/ai4sc/lib/python3.10/site-packages/nvidia/cudnn/lib:/opt/conda/envs/ai4sc/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/envs/ai4sc/lib/python3.10/site-packages/nvidia/cufft/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# 复制项目代码
COPY . /app

USER $MAMBA_USER
CMD ["python", "scripts/test.py"]