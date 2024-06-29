import ONNX

include("./onnx_ops.jl")

b = 314
A = rand(Float32, b, 20, 4);
B = rand(1:30, b, 20);

model = ONNX.load(open("models/tmp_tinyphysics_extracted.onnx"), A, B)