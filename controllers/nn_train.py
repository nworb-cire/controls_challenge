import pytorch_lightning as pl
import torch
from torch import nn

from . import BaseController

FUTURE_PLAN_LENGTH = 10


class ControlsModel(pl.LightningModule):
    def __init__(
        self,
        state_dim: int = 64,
        hidden_dim: int = 64,
        out_dim: int = 1,
    ):
        super().__init__()
        # target lataccel, current lataccel, state, future plan lataccel
        self.input_size = 2 + 3 + FUTURE_PLAN_LENGTH
        self.state_dim = state_dim
        self.network = nn.Sequential(
            nn.Linear(self.input_size + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim + state_dim),
            nn.Tanh(),
        )

    def forward(self, x, st):
        x = torch.cat([x, st.to(x.device)], dim=-1)  # fixme
        x = self.network(x)
        x, st = x.split([1, self.state_dim], dim=-1)
        return x, st

    def save(self):
        torch.onnx.export(
            self,
            (
                torch.randn(2, self.input_size, device=self.device),
                torch.randn(2, self.state_dim, device=self.device),
            ),
            "models/tinyphysics_controls.onnx",
            verbose=False,
            input_names=["input", "state"],
            output_names=["output", "state1"],
            dynamic_axes={
                "input": {0: "b"},
                "state": {0: "b"},
                "output": {0: "b"},
                "state1": {0: "b"},
            }
        )


class Controller(BaseController):
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.model = ControlsModel()
        self.state = torch.zeros((self.batch_size, self.model.state_dim), device=self.model.device)

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan):
        future_plan = future_plan.lataccel[:, :FUTURE_PLAN_LENGTH]
        future_plan = torch.cat([
            future_plan,
            torch.zeros((future_plan.shape[0], FUTURE_PLAN_LENGTH - future_plan.shape[1]), device=future_plan.device)
        ], dim=-1)
        inp = torch.cat((
            target_lataccel.unsqueeze(-1),
            current_lataccel.unsqueeze(-1),
            state,
            future_plan
        ), dim=-1)
        out, self.state = self.model(inp, self.state)
        return out[:, 0]
