import ONNX

include("./onnx_ops.jl")

b = 3
# A = zeros(Float32, b, 20, 4);
# B = zeros(Int64, b, 20);

# ONNX.jl requires the dimensions to be reversed from how python onnxruntime does it
A = ones(Float32, 4, 20, b);
B = ones(Int64, 20, b);

tape = ONNX.load(open("models/tinyphysics.onnx"), A, B)
model = ONNX.compile(tape)
outputs = model(A, B)
if !isa(outputs, Tuple)
    outputs = (outputs,)
end
for (i, o) in enumerate(outputs)
    println("Output $i: $(size(o))")
end
