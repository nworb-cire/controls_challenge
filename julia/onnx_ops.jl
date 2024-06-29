import ONNX
using ONNX: Tape, VarVec, OpConfig, AttrDict, push_call!, onnx_concat
using Statistics: mean
import NNlib

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Identity}, args::VarVec, attrs::AttrDict)
    return args[1]
end

onnx_datatypes = Dict{UInt16, DataType}(
    # 0 => Bfloat16,
    1 => Bool,
    2 => ComplexF64,
    3 => ComplexF32,
    4 => Float64,
    5 => Float32,
    6 => Float16,
    7 => Int16,
    8 => Int32,
    9 => Int64,
    10 => String,
    11 => UInt16,
    12 => UInt32,
    13 => UInt64,
    14 => UInt8,
)

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Cast}, args::VarVec, attrs::AttrDict)
    return args[1]
    # dtype = onnx_datatypes[attrs[:to]]
    # return push_call!(tape, dtype, tape[args[1]].val)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Range}, args::VarVec, attrs::AttrDict)
    start = tape[args[1]].val[1]
    stop = tape[args[2]].val[1] - 1
    step = tape[args[3]].val[1]
    return push_call!(tape, collect ∘ range; start, stop, step)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Reshape}, args::VarVec, attrs::AttrDict)
    dims = Tuple(vec(tape[args[2]].val))
    # replace -1 with :
    dims = map(x -> x == -1 ? Colon() : x, dims)
    if length(dims) == 2
        dims = (dims[2], dims[1])
    elseif length(dims) > 2
        dims = (dims[2], dims[1], dims[3:end]...)
    end
    return push_call!(tape, reshape, tape[args[1]].val, dims)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :ReduceMean}, args::VarVec, attrs::AttrDict)
    dims = attrs[:axes]
    # replace -1 with size
    N = ndims(tape[args[1]].val)
    dims = map(x -> x < 0 ? x + N + 1 : x, dims)
    return push_call!(tape, mean, tape[args[1]].val; dims)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Pow}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, broadcast, ^, tape[args[1]].val, tape[args[2]].val)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Sqrt}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, broadcast, √, tape[args[1]].val)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Div}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, broadcast, /, tape[args[1]].val, tape[args[2]].val)
end
