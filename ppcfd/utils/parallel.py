import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.io import DistributedBatchSampler


def init_dist_env(config):
    if config.enable_mp:
        dist.init_parallel_env()
        mesh = dist.ProcessMesh(
            np.arange(0, paddle.distributed.get_world_size()), dim_names=["mp"]
        )
        dist.auto_parallel.set_mesh(mesh)
    elif config.enable_pp:
        mesh = dist.ProcessMesh(
            np.arange(0, paddle.distributed.get_world_size()), dim_names=["pp"]
        )
        dist.auto_parallel.set_mesh(mesh)
    elif config.enable_dp:
        fleet.init(is_collective=True)
    else:
        NotImplementedError("No distributed training enabled")


def get_mesh():
    return dist.auto_parallel.get_mesh()


def parallelize(model, optimizer, config):
    if config.enable_mp:
        parallel_config = {
            "mp_config": {"parallelize_plan": {}},
        }
        for name, layer in model.named_sublayers():
            layer_type = str(type(layer))
            if "Linear" in layer_type and "mlp2" not in name:
                parallel_config["mp_config"]["parallelize_plan"][
                    name
                ] = dist.ColWiseParallel()
    elif config.enable_pp:
        parallel_config = {
            # TODO: fix for other model
            "pp_config": {
                "split_spec": {
                    "preprocess.linear_post": dist.SplitPoint.END,
                    "blocks.0.pp_layer": dist.SplitPoint.END,
                    "blocks.1.pp_layer": dist.SplitPoint.END,
                },
            },
        }
        assert paddle.distributed.get_world_size() == 4, "currently only support 4 GPUs"
    elif config.enable_dp:
        parallel_config = None
        model = fleet.distributed_model(model)
        optimizer = fleet.distributed_optimizer(optimizer)
    else:
        parallel_config = None
    if parallel_config is not None:
        model, optimizer = dist.parallelize(
            model=model, optimizer=optimizer, config=parallel_config
        )
    
    if config.enable_cinn is True:
        paddle.framework.core._set_prim_all_enabled(True)
        model = paddle.jit.to_static(model, full_graph=True, backend='CINN', input_spec = paddle.static.InputSpec(shape=[1, 32186, 7], dtype='float32'))
    return model, optimizer


def setup_singleprocess_dp(model, optimizer=None):
    """
    单进程 DataParallel 封装（仅在可见 GPU 数量 > 1 时启用）。
    - 不需要分布式启动；当 `CUDA_VISIBLE_DEVICES` 暴露多卡时，使用单进程 DP 以并行加速。
    - 单 GPU 时直接返回原模型，避免不必要的包裹与额外开销。

    Args:
        model: Paddle 模型实例。
        optimizer: 可选优化器；原样返回。

    Returns:
        (model, optimizer): 可能被 DataParallel 包裹的模型，以及原优化器。
    """
    try:
        # 仅在可见设备数量大于 1 时启用 DataParallel
        gpu_count = 0
        try:
            gpu_count = paddle.device.cuda.device_count()
        except Exception:
            gpu_count = 0
        if gpu_count > 1:
            model = paddle.DataParallel(model)
        return model, optimizer
    except Exception:
        # 失败时回退：返回原对象
        return model, optimizer


def get_world_size():
    return dist.get_world_size()


def setup_module(config, model, optimizer=None):
    init_dist_env(config)
    model, optimizer = parallelize(model, optimizer, config)
    return model, optimizer


def setup_dataloaders(config, dataloader, datamodule=None):
    if config.enable_mp or config.enable_pp:
        meshes = get_mesh()
        if meshes is None:
            raise ValueError("Mesh not initialized for MP/PP training")
        shard_dims = "dp" if config.enable_dp else None
        dataloader = dist.shard_dataloader(
            dataloader=dataloader,
            meshes=meshes,
            shard_dims=shard_dims,
        )
    elif config.enable_dp:
        train_sampler = DistributedBatchSampler(
            datamodule.train_data,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        dataloader = datamodule.train_dataloader(
            num_workers=config.num_workers,
            batch_sampler=train_sampler,
        )

    return dataloader
