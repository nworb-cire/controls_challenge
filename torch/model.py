import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from onnx2torch import convert
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from data import DataModule
from tinyphysics import DEL_T, LAT_ACCEL_COST_MULTIPLIER, LATACCEL_RANGE


class LightningModel(pl.LightningModule):
    CONTEXT_WINDOW = 20

    def __init__(
        self,
        onnx_model_path: str,
        controls_model: torch.nn.Module,
    ):
        super().__init__()
        self.state_model = convert(onnx_model_path)
        self.controls_model = controls_model
        self.bins = torch.tensor(np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], 1024), dtype=torch.float32)

    def setup(self, stage: str) -> None:
        self.bins = self.bins.to(self.device)

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        value = self.clip(value)
        return torch.bucketize(value, self.bins, right=True)

    def decode(self, token: torch.Tensor) -> torch.Tensor:
        return self.bins[token]

    def clip(self, value: torch.Tensor) -> torch.Tensor:
        return torch.clamp(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])

    def get_current_lataccel(
        self,
        states,
        tokens,
        top_k: int = 4,
    ):
        assert states.ndim == 3
        assert tokens.ndim == 2
        assert tokens.size(0) == states.size(0)
        assert tokens.size(1) == states.size(1)
        logits = self.state_model(states, tokens)  # B x T x V
        logits = logits[:, -1, :]  # B x V
        v, _ = torch.topk(logits, top_k, dim=-1)  # B x K
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)  # B x V
        token = torch.multinomial(probs, 1)  # B x 1
        return token

    def controls_step(self, target_lataccel, current_lataccel, state, future_plan):
        input_ = torch.cat([target_lataccel, state], dim=-1)
        return self.controls_model(input_)

    def loss_fn(self, preds, targets):
        lat_accel_cost = torch.mean((preds - targets) ** 2)
        jerk_cost = torch.mean(((preds[1:] - preds[:-1]) / DEL_T) ** 2)
        return lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost

    def rollout(self, states, tokens, exog, targets):
        states = torch.cat((
            states,
            torch.cat((
                torch.zeros(exog.size(0), exog.size(1), 1, device=exog.device),
                exog,
            ), dim=-1),
        ), dim=1)
        tokens = torch.cat((tokens, torch.zeros_like(tokens)), dim=-1)
        tokens = self.encode(tokens)
        for i in range(self.CONTEXT_WINDOW):
            predicted_tokens = self.get_current_lataccel(states[:, i:i+self.CONTEXT_WINDOW, :], tokens[:, i:i+self.CONTEXT_WINDOW])
            tokens[:, [i+self.CONTEXT_WINDOW]] = predicted_tokens
            lataccel = self.decode(predicted_tokens)
            control = self.controls_model(lataccel)
            states[:, [i+self.CONTEXT_WINDOW], 0] = control
        return states[:, self.CONTEXT_WINDOW:, 0]

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        preds = self.rollout(*batch)
        targets = batch[-1]
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        preds = self.rollout(*batch)
        targets = batch[-1]
        loss = self.loss_fn(preds, targets)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.controls_model.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data_module = DataModule()
    controls_model = nn.Sequential(
        nn.Linear(1, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    model = LightningModel("models/tinyphysics.onnx", controls_model)

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=data_module)

    torch.onnx.export(
        controls_model,
        torch.randn(2, 1),
        "models/tinyphysics_controls.onnx",
        verbose=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "b"},
            "output": {0: "b"},
        }
    )
