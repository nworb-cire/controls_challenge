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


include("../model_runners.jl")

@testset "Model runner tests" begin
    onnx_model = load_model()
    states = Float32[
        -3.3502060e-01  3.6294633e-01  3.3763535e+01 -3.9068904e-02
        -3.3270627e-01  3.5840231e-01  3.3755951e+01 -6.7940064e-02
        -3.3619869e-01  3.5876277e-01  3.3758335e+01 -2.2276435e-02
        -3.5266215e-01  3.5947475e-01  3.3752903e+01 -5.2032389e-02
        -3.5280910e-01  3.6034307e-01  3.3753941e+01 -1.3069964e-02
        -3.5334441e-01  3.6139569e-01  3.3748669e+01 -4.2920411e-02
        -3.5633713e-01  3.6244828e-01  3.3744259e+01 -4.2917959e-02
        -3.5798737e-01  3.6014545e-01  3.3743309e+01 -8.9320792e-03
        -3.5814551e-01  3.5758153e-01  3.3744141e+01  3.4218599e-04
        -3.5846645e-01  3.5563615e-01  3.3745693e+01  5.5419151e-02
        -3.5989290e-01  3.5449091e-01  3.3733711e+01 -4.8498090e-02
        -3.6355871e-01  3.5334563e-01  3.3722851e+01 -8.6824968e-02
        -3.6567196e-01  3.5110566e-01  3.3723366e+01 -1.4445494e-02
        -3.6522558e-01  3.4880039e-01  3.3717831e+01 -2.1486232e-02
        -3.6024863e-01  3.4771112e-01  3.3709713e+01 -6.5351754e-02
        -3.5633996e-01  3.4817177e-01  3.3710892e+01 -3.5019763e-02
        -3.3960471e-01  3.4863243e-01  3.3696770e+01 -1.0302839e-01
        -3.3905825e-01  3.5576937e-01  3.3691761e+01 -7.6228797e-02
        -3.3745250e-01  3.6326677e-01  3.3681850e+01 -1.0262138e-01
        -3.2244164e-01  3.7218466e-01  3.3666298e+01 -1.4273894e-01
    ]
    tokens = Int64[
        615 619 620 620 620 624 624 624 625 625 625 625 625 626 626 626 624 624 621 621
    ] |> vec

    states = permutedims(states)
    states = reshape(states, size(states)..., 1)
    tokens = reshape(tokens, size(tokens)..., 1)
    res = onnx_model(states, tokens)
    res = permutedims(res, [3, 2, 1])
    @test size(res) == (1, 20, 1024)
    @test res[1, 1, 1] ≈ -3.7462592
    @test res[end, end, end] ≈ -4.9458365
end
