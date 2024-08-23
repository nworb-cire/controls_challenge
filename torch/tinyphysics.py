import importlib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import urllib.request
import zipfile

from io import BytesIO
from collections import namedtuple
from pathlib import Path

import torch
from onnx2torch import convert
from torch import nn
import pytorch_lightning as pl

from controllers import BaseController

sns.set_theme()
signal.signal(signal.SIGINT, signal.SIG_DFL)  # Enable Ctrl-C on plot windows

ACC_G = 9.81
FPS = 10
CONTROL_START_IDX = 100
COST_END_IDX = 500
CONTEXT_LENGTH = 20
VOCAB_SIZE = 1024
LATACCEL_RANGE = [-5, 5]
STEER_RANGE = [-2, 2]
MAX_ACC_DELTA = 0.5
DEL_T = 0.1
LAT_ACCEL_COST_MULTIPLIER = 50.0

FUTURE_PLAN_STEPS = FPS * 5  # 5 secs

State = namedtuple('State', ['roll_lataccel', 'v_ego', 'a_ego'])
FuturePlan = namedtuple('FuturePlan', ['lataccel', 'roll_lataccel', 'v_ego', 'a_ego'])

DATASET_URL = "https://huggingface.co/datasets/commaai/commaSteeringControl/resolve/main/data/SYNTHETIC_V0.zip"
DATASET_PATH = Path(__file__).resolve().parent.parent / "data"


class LataccelTokenizer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.vocab_size = VOCAB_SIZE
        self.bins = nn.Parameter(
            torch.tensor(np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size), dtype=torch.float32),
            requires_grad=False
        )

    def encode(self, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])
        return torch.bucketize(value, self.bins, right=True)

    def decode(self, token: torch.Tensor) -> torch.Tensor:
        return self.bins[token.to(torch.long)]


class TinyPhysicsModel(pl.LightningModule):
    def __init__(self, model_path: str, debug: bool) -> None:
        super().__init__()
        self.tokenizer = LataccelTokenizer()
        self.model = convert(model_path)

    def predict(self, states: torch.Tensor, tokens: torch.Tensor, temperature=1.) -> torch.Tensor:
        assert states.ndim == 3
        assert tokens.ndim == 2
        assert tokens.size(0) == states.size(0)
        assert tokens.size(1) == states.size(1)
        logits = self.model(states, tokens)  # B x T x V
        logits = logits[:, -1, :] / temperature  # B x V
        probs = torch.softmax(logits, dim=-1)  # B x V
        sample = torch.multinomial(probs, num_samples=1)  # B x 1
        assert sample.size(0) == states.size(0)
        assert sample.size(1) == 1
        return sample[:, 0]

    def get_current_lataccel(self, sim_states: torch.Tensor, actions: torch.Tensor, past_preds: torch.Tensor) -> torch.Tensor:
        pred = self.predict(
            states=torch.cat((actions.unsqueeze(1), sim_states[:, :, -CONTEXT_LENGTH:]), dim=1).permute(0, 2, 1),
            tokens=self.tokenizer.encode(past_preds[:, -CONTEXT_LENGTH:]),
            temperature=0.8
        )
        return self.tokenizer.decode(pred)

IDX = {
    "roll_lataccel": 0,
    "v_ego": 1,
    "a_ego": 2,
    "target_lataccel": 3,
    "steer_command": 4
}


def get_data(data_path: str) -> torch.Tensor:
    df = pd.read_csv(data_path)
    processed_df = pd.DataFrame({
        'roll_lataccel': np.sin(df['roll'].values) * ACC_G,
        'v_ego': df['vEgo'].values,
        'a_ego': df['aEgo'].values,
        'target_lataccel': df['targetLateralAcceleration'].values,
        'steer_command': -df['steerCommand'].values  # steer commands are logged with left-positive convention but this simulator uses right-positive
    })
    return torch.tensor(processed_df.values, dtype=torch.float32).unsqueeze(0)


class TinyPhysicsSimulator(pl.LightningModule):
    def __init__(self, model: TinyPhysicsModel, controller: BaseController, debug: bool = False) -> None:
        super().__init__()
        self.sim_model = model
        self.controller = controller
        self.debug = debug

    def reset(self) -> None:
        self.step_idx = CONTEXT_LENGTH
        state_target_futureplans = [self.get_state_target_futureplan(i) for i in range(self.step_idx)]
        self.state_history = torch.cat([x[0].unsqueeze(-1) for x in state_target_futureplans], dim=-1)
        self.action_history = self.data[:, :self.step_idx, IDX['steer_command']]
        self.current_lataccel_history = torch.cat([x[1].unsqueeze(-1) for x in state_target_futureplans], dim=-1)
        self.target_lataccel_history = torch.cat([x[1].unsqueeze(-1) for x in state_target_futureplans], dim=-1)
        self.target_future = None
        self.current_lataccel = self.current_lataccel_history[:, -1]

    def sim_step(self, step_idx: int) -> None:
        pred = self.sim_model.get_current_lataccel(
            sim_states=self.state_history[-CONTEXT_LENGTH:],
            actions=self.action_history[:, -CONTEXT_LENGTH:],
            past_preds=self.current_lataccel_history[-CONTEXT_LENGTH:]
        )
        pred = torch.clamp(pred, self.current_lataccel - MAX_ACC_DELTA, self.current_lataccel + MAX_ACC_DELTA)
        if step_idx >= CONTROL_START_IDX:
            self.current_lataccel = pred
        else:
            self.current_lataccel = self.get_state_target_futureplan(step_idx)[1]

        self.current_lataccel_history = torch.cat((self.current_lataccel_history, pred.unsqueeze(-1)), dim=-1)

    def control_step(self, step_idx: int) -> None:
        action = self.controller.update(
            self.target_lataccel_history[:, step_idx],
            self.current_lataccel,
            self.state_history[:, :, step_idx],
            future_plan=self.futureplan
        )
        if step_idx < CONTROL_START_IDX:
            action = self.data[:, step_idx, IDX['steer_command']]
        action = torch.clamp(action, STEER_RANGE[0], STEER_RANGE[1])
        self.action_history = torch.cat((self.action_history, action.unsqueeze(-1)), dim=-1)

    def get_state_target_futureplan(self, step_idx: int) -> tuple[torch.Tensor, torch.Tensor, FuturePlan]:
        state = self.data[:, step_idx, :]
        return (
            state[:, [IDX['roll_lataccel'], IDX['v_ego'], IDX['a_ego']]],
            state[:, IDX['target_lataccel']],
            FuturePlan(
                lataccel=self.data[:, step_idx + 1:step_idx + FUTURE_PLAN_STEPS, IDX['target_lataccel']],
                roll_lataccel=self.data[:, step_idx + 1:step_idx + FUTURE_PLAN_STEPS, IDX['roll_lataccel']],
                v_ego=self.data[:, step_idx + 1:step_idx + FUTURE_PLAN_STEPS, IDX['v_ego']],
                a_ego=self.data[:, step_idx + 1:step_idx + FUTURE_PLAN_STEPS, IDX['a_ego']]
            )
        )

    def step(self) -> None:
        state, target, futureplan = self.get_state_target_futureplan(self.step_idx)
        self.state_history = torch.cat((self.state_history, state.unsqueeze(-1)), dim=-1)
        self.target_lataccel_history = torch.cat((self.target_lataccel_history, target.unsqueeze(-1)), dim=-1)
        self.futureplan = futureplan
        self.control_step(self.step_idx)
        self.sim_step(self.step_idx)
        self.step_idx += 1

    def plot_data(self, ax, lines, axis_labels, title) -> None:
        ax.clear()
        for line, label in lines:
            ax.plot(line, label=label)
        ax.axline((CONTROL_START_IDX, 0), (CONTROL_START_IDX, 1), color='black', linestyle='--', alpha=0.5, label='Control Start')
        ax.legend()
        ax.set_title(f"{title} | Step: {self.step_idx}")
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    def compute_cost(self) -> dict[str, float]:
        target = self.target_lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]
        pred = self.current_lataccel_history[:, CONTROL_START_IDX:COST_END_IDX]

        lat_accel_cost = torch.mean((target - pred)**2, dim=-1) * 100
        jerk_cost = torch.mean(((pred[:, 1:] - pred[:, :-1]) / DEL_T)**2, dim=-1) * 100
        total_cost = (lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER) + jerk_cost
        return {'lataccel_cost': lat_accel_cost, 'jerk_cost': jerk_cost, 'total_cost': total_cost}

    def rollout(self) -> dict[str, float]:
        if self.debug:
            plt.ion()
            fig, ax = plt.subplots(4, figsize=(12, 14), constrained_layout=True)

        for _ in range(CONTEXT_LENGTH, self.data.size(1)):
            self.step()
            if self.debug and self.step_idx % 10 == 0:
                print(f"Step {self.step_idx:<5}: Current lataccel: {self.current_lataccel:>6.2f}, Target lataccel: {self.target_lataccel_history[-1]:>6.2f}")
                self.plot_data(ax[0], [(self.target_lataccel_history, 'Target lataccel'), (self.current_lataccel_history, 'Current lataccel')], ['Step', 'Lateral Acceleration'], 'Lateral Acceleration')
                self.plot_data(ax[1], [(self.action_history, 'Action')], ['Step', 'Action'], 'Action')
                self.plot_data(ax[2], [(np.array(self.state_history)[:, 0], 'Roll Lateral Acceleration')], ['Step', 'Lateral Accel due to Road Roll'], 'Lateral Accel due to Road Roll')
                self.plot_data(ax[3], [(np.array(self.state_history)[:, 1], 'v_ego')], ['Step', 'v_ego'], 'v_ego')
                plt.pause(0.01)

        if self.debug:
            plt.ioff()
            plt.show()
        return self.compute_cost()

    def get_loss(self, batch):
        self.data = batch
        self.reset()
        for _ in range(CONTEXT_LENGTH, self.data.size(1)):
            self.step()
        return self.compute_cost()["total_cost"]

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.controller.model.parameters(), lr=1e-3)


def get_available_controllers():
    return [f.stem for f in Path('controllers').iterdir() if f.is_file() and f.suffix == '.py' and f.stem != '__init__']


def run_rollout(data_paths, controller_type, model_path, debug=False):
    tinyphysicsmodel = TinyPhysicsModel(model_path, debug=debug)
    controller = importlib.import_module(f'controllers.{controller_type}').Controller()
    sim = TinyPhysicsSimulator(tinyphysicsmodel, controller=controller, debug=debug)
    sim.data = get_data(data_paths[0])
    sim.reset()
    return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history


def download_dataset():
    print("Downloading dataset (0.6G)...")
    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATASET_URL) as resp:
        with zipfile.ZipFile(BytesIO(resp.read())) as z:
            for member in z.namelist():
                if not member.endswith('/'):
                    with z.open(member) as src, open(DATASET_PATH / os.path.basename(member), 'wb') as dest:
                        dest.write(src.read())


class DataModule(pl.LightningDataModule):
    def setup(self, stage: str) -> None:
        files = list(Path(DATASET_PATH).glob('000*.csv'))
        ts = []
        for f in files:
            tensor = get_data(f)[:, :600, :]
            # pad to 600
            pad = torch.zeros((1, 600 - tensor.size(1), tensor.size(2)))
            ts.append(torch.cat([tensor, pad], dim=1))
        self.data = torch.cat(ts, dim=0)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data, batch_size=16, shuffle=True)


if __name__ == "__main__":
    data_module = DataModule()
    controller = importlib.import_module('controllers.nn_train').Controller(batch_size=16)
    controller.model.save()
    controller.model.to("mps")  # fixme

    tinyphysicsmodel = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    sim = TinyPhysicsSimulator(tinyphysicsmodel, controller=controller)

    trainer = pl.Trainer(
        max_epochs=1,
        fast_dev_run=True,
    )
    trainer.fit(sim, datamodule=data_module)
    controller.model.save()
