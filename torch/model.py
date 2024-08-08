import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from onnx2torch import convert
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

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
        # disable gradient computation for the state model
        for param in self.state_model.parameters():
            param.requires_grad = False
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
        tokens = self.encode(tokens)
        new_tokens = []
        steer_commands = []
        for i in range(self.CONTEXT_WINDOW):
            tokens_ = torch.concat((
                tokens,
                *new_tokens,
            ), dim=1)
            if i == 0:
                states_ = states
            else:
                states_ = torch.concat((
                    states,
                    torch.concat((
                        torch.stack(steer_commands, dim=1),  # B x i x 1
                        exog[:, :i, :],  # B x i x 3
                    ), dim=-1)  # B x i x 4
                ), dim=1)  # B x (T + i) x 4
            current_token = self.get_current_lataccel(
                states_[:, i:, :],
                tokens_[:, i:],
            )
            new_tokens.append(current_token)
            current_lataccel = self.decode(current_token)
            steer_command = self.controls_step(current_lataccel, targets[:, [i]], exog[:, i, :], None)
            steer_commands.append(steer_command)
        return torch.concat(new_tokens, dim=1)

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        predicted_tokens = self.rollout(*batch)
        loss = self.loss_fn(predicted_tokens, batch[-1])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        predicted_tokens = self.rollout(*batch)
        loss = self.loss_fn(predicted_tokens, batch[-1])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.controls_model.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    data_module = DataModule()
    model = LightningModel("models/tinyphysics.onnx", torch.nn.Linear(4, 1))

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
    )
    trainer.fit(model, datamodule=data_module)
