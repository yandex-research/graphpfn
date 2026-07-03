from collections.abc import Callable

import torch
import torch.nn as nn
import torch.utils.checkpoint


def apply_dynamic_checkpointing(
    module: nn.Module,
    should_checkpoint_fn: Callable[..., bool],
    submodule_filter_fn: Callable[[str, nn.Module], bool],
    use_reentrant: bool = False,
) -> None:
    should_checkpoint_holder = [False]  # mutable holder for `should_checkpoint` flag

    def check_should_checkpoint(_, args, kwargs):
        should_checkpoint_holder[0] = should_checkpoint_fn(*args, **kwargs)
        return args, kwargs

    module.register_forward_pre_hook(check_should_checkpoint, with_kwargs=True)

    def patch_submodule(submodule: nn.Module) -> None:
        original_forward = submodule.forward
        submodule.forward = lambda *args, **kwargs: (
            torch.utils.checkpoint.checkpoint(
                original_forward,
                *args,
                use_reentrant=use_reentrant,
                **kwargs,
            )
            if should_checkpoint_holder[0] and torch.is_grad_enabled()
            else original_forward(*args, **kwargs)
        )

    for name, submodule in module.named_modules():
        if (submodule is module) or not submodule_filter_fn(name, submodule):
            continue
        patch_submodule(submodule)
