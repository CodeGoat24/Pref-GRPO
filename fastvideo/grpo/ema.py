from collections.abc import Iterable

import torch


class EMAModuleWrapper:
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        decay: float = 0.999,
        update_step_interval: int = 1,
        device: torch.device | None = None,
    ):
        params = list(parameters)
        self.shadow_parameters = [p.detach().clone().to(device) for p in params]
        self.backup_parameters = None
        self.decay = float(decay)
        self.update_step_interval = int(update_step_interval)
        self.device = device

    def _check_length(self, parameters):
        if len(parameters) != len(self.shadow_parameters):
            raise ValueError(
                "EMA parameter count mismatch: "
                f"{len(parameters)} vs {len(self.shadow_parameters)}."
            )

    @torch.no_grad()
    def step(self, parameters: Iterable[torch.nn.Parameter], optimization_step: int):
        if (optimization_step + 1) % self.update_step_interval != 0:
            return
        params = list(parameters)
        self._check_length(params)
        decay = self.decay
        one_minus_decay = 1.0 - decay
        for shadow, param in zip(self.shadow_parameters, params):
            if not param.requires_grad:
                continue
            if shadow.device == param.device:
                shadow.mul_(decay).add_(param.detach(), alpha=one_minus_decay)
            else:
                param_copy = param.detach().to(shadow.device)
                shadow.mul_(decay).add_(param_copy, alpha=one_minus_decay)

    def to(self, device: torch.device | None = None, dtype: torch.dtype | None = None):
        self.device = device
        updated = []
        for param in self.shadow_parameters:
            if dtype is not None and param.is_floating_point():
                updated.append(param.to(device=device, dtype=dtype))
            else:
                updated.append(param.to(device=device))
        self.shadow_parameters = updated

    @torch.no_grad()
    def copy_to(self, parameters: Iterable[torch.nn.Parameter], store_temp: bool = True):
        params = list(parameters)
        self._check_length(params)
        if store_temp:
            self.backup_parameters = [p.detach().cpu().clone() for p in params]
        for shadow, param in zip(self.shadow_parameters, params):
            param.data.copy_(shadow.to(param.device).data)

    @torch.no_grad()
    def restore(self, parameters: Iterable[torch.nn.Parameter]):
        if self.backup_parameters is None:
            return
        params = list(parameters)
        self._check_length(params)
        for backup, param in zip(self.backup_parameters, params):
            param.data.copy_(backup.to(param.device).data)
        self.backup_parameters = None
