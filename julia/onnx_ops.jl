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
    stop = tape[args[2]].val[1] + 1
    step = tape[args[3]].val[1]
    return push_call!(tape, range; start, stop, step)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Reshape}, args::VarVec, attrs::AttrDict)
    dims = Tuple(vec(tape[args[2]].val))
    # replace -1 with :
    dims = map(x -> x == -1 ? Colon() : x, dims)
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

# function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :MatMul}, args::VarVec, attrs::AttrDict)
#     A_ndims = ndims(args[1]._op.val)
#     B_ndims = ndims(args[2]._op.val)
#     @show size(args[1]._op.val), size(args[2]._op.val)
#     if A_ndims == 2 && B_ndims == 2
#         return push_call!(tape, *, args[2], args[1])
#     # elseif A_ndims == 3 && B_ndims in (2, 3)
#     #     # batch is first, but it should be last
#     #     permute = push_call!(tape, permutedims, args[1], (3, 2, 1))
#     #     mul = push_call!(tape, NNlib.batched_mul, args[2], permute._op.val)
#     #     return push_call!(tape, permutedims, mul._op.val, (3, 2, 1))
#     # elseif A_ndims == 2 && B_ndims in (2, 3)
#     elseif A_ndims in (2, 3) && B_ndims in (2, 3)
#         return push_call!(tape, NNlib.batched_mul, args[2], args[1])
#     else
#         error("MatMul with arrays of $A_ndims and $B_ndims is not implemented yet")
#     end
# end

function my_batched_mul(A, B)
    B = permutedims(B, [3, 2, 1])
    C = NNlib.batched_mul(A, B)
    return permutedims(C, [3, 2, 1])
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :MatMul}, args::VarVec, attrs::AttrDict)
    A_ndims = ndims(args[1]._op.val)
    B_ndims = ndims(args[2]._op.val)
    if A_ndims == 2 && B_ndims == 2
        return push_call!(tape, *, args[2], args[1])
    elseif A_ndims in (2, 3) && B_ndims in (2, 3)
        return push_call!(tape, my_batched_mul, args[2], args[1])
    else
        error("MatMul with arrays of $A_ndims and $B_ndims is not implemented yet")
    end
end