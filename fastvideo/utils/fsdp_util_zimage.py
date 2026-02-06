import functools

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_z_image import ZImageTransformerBlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


class FSDPConfig:
    def __init__(
        self,
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE",
        cpu_offload=False,
        num_replicate=1,
        num_shard=8,
        mixed_precision_dtype=torch.bfloat16,
        use_device_mesh=False,
    ):
        self.sharding_strategy = sharding_strategy
        self.backward_prefetch = backward_prefetch
        self.cpu_offload = cpu_offload
        self.num_replicate = num_replicate
        self.num_shard = num_shard
        self.mixed_precision_dtype = mixed_precision_dtype
        self.use_device_mesh = use_device_mesh


def fsdp_wrapper(model, fsdp_config, ignored_modules=None):
    if ignored_modules is None:
        ignored_modules = []

    device_mesh = None
    if fsdp_config.sharding_strategy == "HYBRID_SHARD" and fsdp_config.use_device_mesh:
        device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=(fsdp_config.num_replicate, fsdp_config.num_shard),
            mesh_dim_names=("replicate", "shard"),
        )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={ZImageTransformerBlock},
        ),
        ignored_modules=ignored_modules,
        mixed_precision=MixedPrecision(
            param_dtype=fsdp_config.mixed_precision_dtype,
            reduce_dtype=fsdp_config.mixed_precision_dtype,
            buffer_dtype=fsdp_config.mixed_precision_dtype,
        ),
        device_id=dist.get_rank() % torch.cuda.device_count(),
        sharding_strategy=ShardingStrategy[fsdp_config.sharding_strategy],
        backward_prefetch=BackwardPrefetch[fsdp_config.backward_prefetch],
        cpu_offload=CPUOffload(offload_params=fsdp_config.cpu_offload),
        device_mesh=device_mesh,
        use_orig_params=True,
    )
    return fsdp_model


def apply_fsdp_checkpointing(model, selective_checkpointing=1.0):
    selective_checkpointing = eval(selective_checkpointing) if isinstance(selective_checkpointing, str) else selective_checkpointing
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    block_idx = 0
    cut_off = 0.5

    def check_fn(submodule):
        nonlocal block_idx
        nonlocal cut_off
        if isinstance(submodule, ZImageTransformerBlock):
            block_idx += 1
            if block_idx * selective_checkpointing >= cut_off:
                cut_off += 1.0
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )
