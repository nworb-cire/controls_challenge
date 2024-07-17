using Test
import ONNX
import PyCall

include("../onnx_ops.jl")

@testset "ONNX model tests" begin
    A = ones(Float32, 4, 20, 3)
    B = ones(Int64, 20, 3)
    tape = ONNX.load("models/tinyphysics.onnx", A, B)
    model = ONNX.compile(tape)

    outputs = model(A, B)
    @test size(outputs) == (1024, 20, 3)
    @test outputs[1:3, 1:3, 1] ≈ [
        6.91149  6.82687  6.51002
        3.14077  3.61317  3.34305
        3.32208  3.93678  3.60959
    ]

    ort = PyCall.pyimport("onnxruntime")
    sess = ort.InferenceSession("models/tinyphysics.onnx")

    inputs = Dict("states" => permutedims(A, [3, 2, 1]), "tokens" => permutedims(B, [2, 1]))
    py_outputs = sess.run(["output"], inputs)[1]
    @test size(py_outputs) == (3, 20, 1024)
    @test permutedims(py_outputs, [3, 2, 1]) ≈ outputs
end
