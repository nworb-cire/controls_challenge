import numpy as np
import onnxruntime

from . import BaseController


class Controller(BaseController):
    def __init__(self, path="models/tinyphysics_controls.onnx"):
        self.sess = onnxruntime.InferenceSession(path)

    def update(self, target_lataccel: float, current_lataccel: float, state, future_plan):
        inp = np.array([target_lataccel, *state], dtype=np.float32)
        inp = inp[np.newaxis, :]
        out = self.sess.run(None, {"input": inp})[0]
        return out[0, 0]
