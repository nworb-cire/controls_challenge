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
    function _range(start, stop, step)
        start = only(start)
        stop = only(stop) - 1
        step = only(step)
        return collect(start:step:stop)
    end
    return push_call!(tape, _range, args[1], args[2], args[3])
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Reshape}, args::VarVec, attrs::AttrDict)
    function _reshape(arr, dims)
        dims = Tuple(vec(dims))
        # replace -1 with :
        dims = map(x -> x == -1 ? (:) : Integer(x), dims)
        dims = dims[end:-1:1]
        return reshape(arr, dims...)
    end
    return push_call!(tape, _reshape, args[1], args[2])
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :ReduceMean}, args::VarVec, attrs::AttrDict)
    dims = attrs[:axes]
    # replace -1 with size
    N = ndims(tape[args[1]].val)
    dims = map(x -> x >= 0 ? N - x : -x, dims)
    return push_call!(tape, mean, args[1]; dims)
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Pow}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, broadcast, ^, args[1], args[2])
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Sqrt}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, broadcast, √, args[1])
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Div}, args::VarVec, attrs::AttrDict)
    return push_call!(tape, broadcast, /, args[1], args[2])
end

function NNlib.unsqueeze(x::AbstractArray, dims)
    # new_shape::Vector{Integer} = collect(size(x))
    # for d in sort(collect(dims))
    #     # insert!(new_shape, d, 1)
    #     new_shape = new_shape[1:d-1] ∪ [1] ∪ new_shape[d:end]
    # end
    # return reshape(x, new_shape...)
    return NNlib.unsqueeze(x)
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
    out_size = (size_before..., size(idxs)..., size_after...)
    colons_before = ntuple(_ -> :, dim-1)
    colons_after = ntuple(_ -> :, ndims(data) - dim)

    # Collect slices and build the output array
    slices = [
        data[colons_before..., idxs[R]..., colons_after...]
        for R in CartesianIndices(idxs)
    ]
    
    # Reshape the collected slices into the desired output shape
    new_out = reshape(vcat(slices...), out_size)
    
    return new_out
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Transpose}, args::VarVec, attrs::AttrDict)
    # this is gross, figure out why this is necessary
    function _permutedims(arr, perm)
        perm = perm .+ 1
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
        return permutedims(arr, perm)
    end
    return push_call!(tape, _permutedims, args[1], attrs[:perm])
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
    return push_call!(tape, (x, y, z) -> ifelse.(x, y, z), args[1], args[2], args[3])
end

function ONNX.load_node!(tape::Tape, ::OpConfig{:ONNX, :Softmax}, args::VarVec, attrs::AttrDict)
    axis = attrs[:axis]
    dims = axis >= 0 ? ndims(input) - axis  : -axis 
    return push_call!(tape, NNlib.softmax, args[1]; dims)
end
