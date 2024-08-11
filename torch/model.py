import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from onnx2torch import convert
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from tqdm import trange

from data import DataModule
from tinyphysics import DEL_T, LAT_ACCEL_COST_MULTIPLIER, LATACCEL_RANGE, run_rollout, CONTEXT_LENGTH, COST_END_IDX, \
    CONTROL_START_IDX, VOCAB_SIZE


class ControlsModel(pl.LightningModule):
    def __init__(
        self,
        input_size: int = 5,
        state_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 1,
    ):
        super().__init__()
        self.input_size = input_size
        self.state_dim = state_dim
        self.fc1 = nn.Linear(input_size + state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.drop1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.drop2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(hidden_dim, out_dim + state_dim)

    def forward(self, x, st):
        x = torch.cat([x, st], dim=-1)
        x = F.tanh(self.fc1(x))
        x = self.bn1(x)
        x = self.drop1(x)

        x = x + F.tanh(self.fc2(x))
        x = self.bn2(x)
        x = self.drop2(x)

        x = F.tanh(self.fc3(x))
        x, st = x.split([1, self.state_dim], dim=-1)
        return x, st


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        onnx_model_path: str,
        controls_model: torch.nn.Module,
    ):
        super().__init__()
        self.state_model = convert(onnx_model_path)
        self.controls_model = controls_model
        bins = torch.tensor(np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], VOCAB_SIZE), dtype=torch.float32)
        self.bins = nn.Parameter(bins, requires_grad=False)

    def tokenize(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return torch.bucketize(value, self.bins, right=True)

    def detokenize(self, token: torch.Tensor) -> torch.Tensor:
        return self.bins[token.to(torch.long)]

    def detokenize_differentiable(self, token: torch.Tensor) -> torch.Tensor:
        source_min, source_max = 0, VOCAB_SIZE
        target_min, target_max = LATACCEL_RANGE

        # Perform the linear mapping
        return (token - source_min) * (target_max - target_min) / (source_max - source_min) + target_min

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

    def controls_step(self, target_lataccel, current_lataccel, state, future_plan, st):
        inp = torch.cat([target_lataccel, current_lataccel, state], dim=-1)
        return self.controls_model(inp, st)

    def loss_fn(self, preds, targets):
        lat_accel_cost = torch.mean((preds - targets) ** 2)
        jerk_cost = torch.mean(((preds[1:] - preds[:-1]) / DEL_T) ** 2)
        return lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost

    def rollout(self, inp):
        # mask controls
        inp[:, CONTEXT_LENGTH:, 0] = 0
        inp[:, :, -1] = self.tokenize(inp[:, :, -1])
        st = torch.zeros(inp.size(0), self.controls_model.state_dim, dtype=torch.float32, device=self.device)
        pbar = trange(COST_END_IDX - CONTROL_START_IDX, desc="Rollout")
        for i in pbar:
            predicted_tokens = self.get_current_lataccel(
                inp[:, i:i+CONTEXT_LENGTH, :-1],
                inp[:, i:i+CONTEXT_LENGTH, -1].to(torch.long),
            )
            control, st = self.controls_step(
                self.detokenize(predicted_tokens.to(torch.float32)),
                self.detokenize(inp[:, [i+CONTEXT_LENGTH-1], -1]),
                inp[:, i+CONTEXT_LENGTH, 1:-1],
                None,
                st,
            )
            inp[:, [i+CONTEXT_LENGTH], -1] = predicted_tokens.to(torch.float32)
            inp[:, [i+CONTEXT_LENGTH], 0] = control
        pbar.close()
        return inp[:, CONTEXT_LENGTH:, -1]

    def training_step(self, batch, *args, **kwargs) -> STEP_OUTPUT:
        targets = batch[:, CONTEXT_LENGTH:, -1].clone()
        preds = self.rollout(batch)
        preds = self.detokenize_differentiable(preds)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def save(self):
        torch.onnx.export(
            self.controls_model,
            (
                torch.randn(2, self.controls_model.input_size, device=self.device),
                torch.randn(2, self.controls_model.state_dim, device=self.device),
            ),
            "models/tinyphysics_controls.onnx",
            verbose=True,
            input_names=["input", "state"],
            output_names=["output", "state1"],
            dynamic_axes={
                "input": {0: "b"},
                "state": {0: "b"},
                "output": {0: "b"},
                "state1": {0: "b"},
            }
        )

    def on_validation_epoch_start(self) -> None:
        self.save()

    def validation_step(self, file) -> STEP_OUTPUT:
        cost, _, _ = run_rollout(
            data_path=file,
            controller_type="nn",
            model_path="models/tinyphysics.onnx",
        )
        loss = cost["total_cost"]
        self.log("val_loss", loss, batch_size=1, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.controls_model.parameters(), lr=1e-3)
        return optimizer


if __name__ == "__main__":
    pl.seed_everything(0)
    data_module = DataModule()
    model = LightningModel("models/tinyphysics.onnx", ControlsModel())

    trainer = pl.Trainer(
        max_epochs=10,
        val_check_interval=5,
    )
    trainer.fit(model, datamodule=data_module)
