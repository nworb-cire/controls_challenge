import numpy as np
import onnxruntime

from . import BaseController


class Controller(BaseController):
    def __init__(self, path="models/tinyphysics_controls.onnx"):
        self.sess = onnxruntime.InferenceSession(path)
        self.d_state = self.sess.get_inputs()[1].shape[1]
        self.state = np.zeros((1, self.d_state), dtype=np.float32)

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan):
        inp = np.array([target_lataccel, *state], dtype=np.float32)
        inp = inp[np.newaxis, :]
        out, self.state = self.sess.run(["output", "state1"], {"input": inp, "state": self.state})
        return out[0, 0]
