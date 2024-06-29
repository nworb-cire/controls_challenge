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
    start = tape[args[1]].val[1] |> only
    stop = tape[args[2]].val[1] |> only
    stop = stop - 1
    step = tape[args[3]].val[1] |> only
    return push_call!(tape, collect ∘ range; start, stop, step)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Reshape}, args::VarVec, attrs::AttrDict)
    dims = Tuple(vec(tape[args[2]].val))
    # replace -1 with :
    dims = map(x -> x == -1 ? (:) : Integer(x), dims)
    dims = dims[end:-1:1]
    return push_call!(tape, reshape, tape[args[1]].val, dims)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :ReduceMean}, args::VarVec, attrs::AttrDict)
    dims = attrs[:axes]
    # replace -1 with size
    N = ndims(tape[args[1]].val)
    dims = map(x -> x >= 0 ? N - x : -x, dims)
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

function NNlib.unsqueeze(x::AbstractArray, dims)
    new_shape::Vector{Integer} = collect(size(x))
    for d in sort(collect(dims))
        insert!(new_shape, d, 1)
    end
    return reshape(x, new_shape...)
end

function ONNX.onnx_unsqueeze(x::Real, dims)
    return ONNX.onnx_unsqueeze(fill(x, ()), dims)
end

function ONNX.take(
    data::AbstractArray{T, N}, idxs::AbstractArray{Int, M};
    dim=ndims(data)) where {T, N, M}
    if length(idxs) == 1 && data isa ONNX.SVector
        # we use SVector to represent array size, Gather(arr_sz, idx)
        # works as size(arr, idx); but since dimensions are reversed,
        # we need to reverse the index as well
        # see https://github.com/FluxML/ONNX.jl/issues/62 for details
        idx = length(data) .- idxs .+ 1
        dat = data[idx] |> only
        return fill(dat, ())
    end
    if length(idxs) == 1
        # special case, works as getindex
        # this needs to be a zero-dimensional array
        dat = data[idxs] |> only
        return fill(dat, ())
    end
    # we will take slices of data of this size
    size_before = (size(data)[1:dim-1]...,)
    size_after = (size(data)[dim+1:ndims(data)]...,)
    # and put them into output array at out[:, :, ..., idxs[i, j, ...]]
    out = similar(data, (size_before..., size(idxs)..., size_after...))
    colons_before = [(:) for _=1:dim-1]
    colons_after = [(:) for _=dim+1:ndims(data)]
    # iteration over idxs doesn't depend on data or dimension
    # we iterate over the last index purely due to memory layout
    for i=1:size(idxs, ndims(idxs))
        # R - slice of idxs (not slice of data!)
        R = [[(:) for _=1:ndims(idxs)-1]..., i]
        # ensure I = idxs[R...] is itself an array and not a scalar
        I = [idxs[R...]...,]
        slice = data[colons_before..., I, colons_after...]
        out[colons_before..., R..., colons_after...] = slice
    end
    return out
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Transpose}, args::VarVec, attrs::AttrDict)
    # this is gross, figure out why this is necessary
    perm = attrs[:perm] .+ 1
    if length(perm) == 4
        if size(tape[args[1]].val, 1) == 3
            perm_ = [1, 2, 3, 4]
        elseif size(tape[args[1]].val, 2) == 3
            perm_ = [2, 1, 3, 4]
        elseif size(tape[args[1]].val, 3) == 3
            perm_ = [3, 1, 2, 4]
        elseif size(tape[args[1]].val, 4) == 3
            perm_ = [4, 3, 2, 1]
        else
            perm_ = [1, 2, 3, 4]
        end
        perm_ = perm_[perm]
        perm = perm_[[4, 3, 2, 1]]  # put batch dimension last  # [4, 3, 1, 2] ?
    end
    return push_call!(tape, permutedims, tape[args[1]].val, perm)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :MatMul}, args::VarVec, attrs::AttrDict)
    A_ndims = ndims(args[1]._op.val)
    B_ndims = ndims(args[2]._op.val)
    if A_ndims == 2 && B_ndims == 2
        return push_call!(tape, *, args[2], args[1])
    else
        return push_call!(tape, NNlib.batched_mul, args[2], args[1])
    end
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Not}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, .!, tape[args[1]].val)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Where}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, (x, y, z) -> ifelse.(x, y, z), tape[args[1]].val, tape[args[2]].val, tape[args[3]].val)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Softmax}, args::VarVec, attrs::AttrDict)
    axis = attrs[:axis]
    dims = axis >= 0 ? ndims(input) - axis  : -axis 
    return push_call!(tape, NNlib.softmax, tape[args[1]].val; dims)
end
