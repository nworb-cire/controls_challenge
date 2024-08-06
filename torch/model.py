import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from onnx2torch import convert

from tinyphysics import DEL_T, LAT_ACCEL_COST_MULTIPLIER


class LataccelTokenizer:
  def __init__(self):
    self.vocab_size = VOCAB_SIZE
    self.bins = np.linspace(LATACCEL_RANGE[0], LATACCEL_RANGE[1], self.vocab_size)

  def encode(self, value: Union[float, np.ndarray, List[float]]) -> Union[int, np.ndarray]:
    value = self.clip(value)
    return np.digitize(value, self.bins, right=True)

  def decode(self, token: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    return self.bins[token]

  def clip(self, value: Union[float, np.ndarray, List[float]]) -> Union[float, np.ndarray]:
    return np.clip(value, LATACCEL_RANGE[0], LATACCEL_RANGE[1])


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
        self.tokenizer = LataccelTokenizer()

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
        logits = self.state_model(states[-self.CONTEXT_WINDOW:, :], tokens[-self.CONTEXT_WINDOW:])
        logits = logits[:, -1, :]
        v, _ = torch.topk(logits, top_k, dim=-1)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, 1)
        return token

    def controls_step(self, target_lataccel, current_lataccel, state, future_plan):
        input_ = torch.cat([target_lataccel, current_lataccel, *state])
        return self.controls_model(input_)

    def loss_fn(self, preds, targets):
        lat_accel_cost = torch.mean((preds - targets) ** 2)
        jerk_cost = torch.mean(((preds[1:] - preds[:-1]) / DEL_T) ** 2)
        return lat_accel_cost * LAT_ACCEL_COST_MULTIPLIER + jerk_cost


if __name__ == "__main__":
    states = torch.randn(1, 20, 4)
    tokens = torch.randint(0, 4, (1, 20))
    future_plan = torch.randn(10, 2)

    model = LightningModel("models/tinyphysics.onnx", torch.nn.Linear(4, 4))
    for _ in range(100):
        x = model.get_current_lataccel(states, tokens)
        print(x)
