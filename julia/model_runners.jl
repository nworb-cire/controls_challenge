import ONNX

include("./onnx_ops.jl")

const VOCAB_SIZE = 1024
const LATACCEL_RANGE = (-5, 5)

const tokenizer_bins = LinRange(LATACCEL_RANGE[1], LATACCEL_RANGE[2], VOCAB_SIZE)

function encode(value)
    value = clamp.(value, LATACCEL_RANGE[1], LATACCEL_RANGE[2])
    digitized = searchsortedlast.(Ref(tokenizer_bins), value)
    return digitized
end

function decode(token::Union{Int, Vector{Int}})
    return tokenizer_bins[token]
end

function predict(model, states::Matrix, tokens::Vector; temperature=1.0)
    states = permutedims(states)
    states = reshape(states, size(states)..., 1)
    tokens = reshape(tokens, size(tokens)..., 1)
    res = model(states, tokens)
    res = permutedims(res, [3, 2, 1])
    res = res[1:1, :, :]  # fixme
    probs = softmax(res ./ temperature; dims=size(res, 3))
    @assert size(probs, 1) == 1
    @assert size(probs, 3) == VOCAB_SIZE
    weights = Weights(probs[1, 1, :])
    return sample(1:VOCAB_SIZE, weights)
end

function load_model(model_path = "./models/tinyphysics.onnx")
    b = 3
    A = ones(Float32, 4, 20, b)
    B = ones(Int64, 20, b)
    tape = ONNX.load(model_path, A, B)
    return ONNX.compile(tape)
end
