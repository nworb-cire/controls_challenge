import ONNX

include("./onnx_ops.jl")

b = 3
# A = zeros(Float32, b, 20, 4);
# B = zeros(Int64, b, 20);

# ONNX.jl requires the dimensions to be reversed from how python onnxruntime does it
A = ones(Float32, 4, 20, b);
B = ones(Int64, 20, b);

tape = ONNX.load("models/tinyphysics.onnx", A, B)
model = ONNX.compile(tape)
outputs = model(A, B)
if !isa(outputs, Tuple)
    outputs = (outputs,)
end
for (i, o) in enumerate(outputs)
    println("Output $i: $(size(o))")
end
@assert outputs[1][1:3, 1:3, 1] ≈ [
    6.91149  6.82687  6.51002
    3.14077  3.61317  3.34305
    3.32208  3.93678  3.60959
]

import PyCall

ort = PyCall.pyimport("onnxruntime")
sess = ort.InferenceSession("models/tinyphysics.onnx")

inputs = Dict("states" => permutedims(A, [3, 2, 1]), "tokens" => permutedims(B, [2, 1]))
py_outputs = sess.run(["output"], inputs)
for (i, o) in enumerate(py_outputs)
    println("Output $i: $(size(o))")
end
@assert permutedims(py_outputs[1], [3, 2, 1]) ≈ outputs[1]
