throw_dimserror(dims::Integer, N) = throw(ArgumentError("dims = $dims does not satisfy 1 <= dims <= $N"))
throw_dimserror(dims, N) = throw(ArgumentError("dims = $dims does not satisfy 1 <= dims <= $N for all elements"))

throw_axesmismatcherror(dim, axexp, axrcv) = throw(
	DimensionMismatch("axes mismatch in dimension $dim, expected $axexp but received $axrcv"))

function _checkdims(A, dim, ax_exp)
	for a in A
		axadim = axes(a, dim)
		if axadim != ax_exp
			throw_axesmismatcherror(dim, ax_exp, axadim)
		end
	end
end

function checkdims(A, dims)
	for (dim, ax_exp) in enumerate(axes(first(A)))
		if dim ∉ dims
			_checkdims(A, dim, ax_exp)
		end
	end
end

function checkdims(A, d::Integer)
	for (dim, ax_exp) in enumerate(axes(first(A)))
		dim == d && continue
		_checkdims(A, dim, ax_exp)
	end
end

"""
	sumcat_aligned(A::AbstractArray{T,N}...; dims) where {T,N}

Concatenate the arrays along the dimensions `dims` according to their axes, 
with overlapping sections being summed over. Returns an `OffsetArray` with the minimal 
axis span encompassing all the arrays.

`dims` may be an `Integer` or a collection of `Integer`s, but all elements of `dims` must be from the range `1:N`.

# Examples
```jldoctest
julia> ParallelUtilities.sumcat_aligned(ones(1:2), ones(4:5), dims=1)
5-element OffsetArray(::Array{Float64,1}, 1:5) with eltype Float64 with indices 1:5:
 1.0
 1.0
 0.0
 1.0
 1.0

julia> ParallelUtilities.sumcat_aligned(ones(1:2, 1:2), ones(2:3, 2:3), dims=(1,2))
3×3 OffsetArray(::Array{Float64,2}, 1:3, 1:3) with eltype Float64 with indices 1:3×1:3:
 1.0  1.0  0.0
 1.0  2.0  1.0
 0.0  1.0  1.0

julia> ParallelUtilities.sumcat_aligned(ones(1:2, 1:2), ones(3:4, 3:4), dims=(1,2))
4×4 OffsetArray(::Array{Float64,2}, 1:4, 1:4) with eltype Float64 with indices 1:4×1:4:
 1.0  1.0  0.0  0.0
 1.0  1.0  0.0  0.0
 0.0  0.0  1.0  1.0
 0.0  0.0  1.0  1.0
```

See also: [`sumhcat_aligned`](@ref), [`sumvcat_aligned`](@ref)
"""
function sumcat_aligned(A::AbstractArray{T,N}...; dims) where {T,N}

	all(x -> 1 <= x <= N, dims) || throw_dimserror(dims, N)

	checkdims(A, dims)

	ax = Vector{UnitRange{Int}}(undef, N)
	ax .= axes(first(A))

	for d in dims
		axmin = minimum(minimum.(axes.(A, d)))
		axmax = maximum(maximum.(axes.(A, d)))
		ax[d] = axmin:axmax
	end
	
	arr = OffsetArray{T,N}(undef, ax...)
	fill!(arr, zero(T))

	for a in A
		arr[axes(a)...] .+= a
	end
	arr
end

sumcat_aligned(A1::AbstractArray; dims) = (all(x -> 1 <= x <= ndims(A1), dims) || throw_dimserror(dims); A1)

"""
	sumvcat_aligned(A::AbstractArray{T,N}...) where {T,N}

Concatenate the arrays along the first dimension according to their axes, 
with overlapping sections being summed over. Returns an `OffsetArray` with the minimal 
axis span encompassing all the arrays.

The input arrays must be at least one-dimensional.

# Examples
```jldoctest
julia> ParallelUtilities.sumvcat_aligned(ones(1:2), ones(4:5))
5-element OffsetArray(::Array{Float64,1}, 1:5) with eltype Float64 with indices 1:5:
 1.0
 1.0
 0.0
 1.0
 1.0

julia> ParallelUtilities.sumvcat_aligned(ones(1:2, 1:2), ones(2:3, 1:2))
3×2 OffsetArray(::Array{Float64,2}, 1:3, 1:2) with eltype Float64 with indices 1:3×1:2:
 1.0  1.0
 2.0  2.0
 1.0  1.0
```

See also: [`sumcat_aligned`](@ref), [`sumhcat_aligned`](@ref)
"""
function sumvcat_aligned(A::AbstractArray{T,N}...) where {T,N}

	N >= 1 || throw(ArgumentError("all the arrays need to have at least 1 dimension"))
	checkdims(A, 1)

	axmin = minimum(minimum.(axes.(A, 1)))
	axmax = maximum(maximum.(axes.(A, 1)))
	
	axcat = axmin:axmax

	trailing_axes = Base.tail(axes(first(A)))
	
	arr = OffsetArray{T,N}(undef, axcat, trailing_axes...)
	fill!(arr, zero(T))

	for axt in CartesianIndices(trailing_axes)
		for a in A, ind1 in axes(a,1)
			arr[ind1, axt] += a[ind1, axt]
		end
	end

	arr
end

function sumvcat_aligned(A::AbstractArray)
	ndims(A) >= 1 || throw(ArgumentError("the array needs to have at least 1 dimension"))
	A
end

"""
	sumhcat_aligned(A::AbstractArray{T,N}...) where {T,N}

Concatenate the arrays along the second dimension according to their axes, 
with overlapping sections being summed over. Returns an `OffsetArray` with the minimal 
axis span encompassing all the arrays. 

The input arrays must be at least two-dimensional.

# Examples
```jldoctest
julia> ParallelUtilities.sumhcat_aligned(ones(2, 1:2), ones(2, 4:5))
2×5 OffsetArray(::Array{Float64,2}, 1:2, 1:5) with eltype Float64 with indices 1:2×1:5:
 1.0  1.0  0.0  1.0  1.0
 1.0  1.0  0.0  1.0  1.0

julia> ParallelUtilities.sumhcat_aligned(ones(1:2, 1:2), ones(1:2, 2:3))
2×3 OffsetArray(::Array{Float64,2}, 1:2, 1:3) with eltype Float64 with indices 1:2×1:3:
 1.0  2.0  1.0
 1.0  2.0  1.0
```

See also: [`sumcat_aligned`](@ref), [`sumvcat_aligned`](@ref)
"""
function sumhcat_aligned(A::AbstractArray{T,N}...) where {T,N}

	N >= 2 || throw(ArgumentError("all the arrays need to have at least 2 dimensions"))
	checkdims(A, 2)

	axmin = minimum(minimum.(axes.(A, 2)))
	axmax = maximum(maximum.(axes.(A, 2)))
	
	axcat = axmin:axmax

	trailing_axes = Base.tail(Base.tail(axes(first(A))))
	
	arr = OffsetArray{T,N}(undef, axes(first(A),1), axcat, trailing_axes...)
	fill!(arr, zero(T))

	for axt in CartesianIndices(trailing_axes)
		for a in A, ind2 in axes(a,2), ind1 in axes(a,1)
			arr[ind1, ind2, axt] += a[ind1, ind2, axt]
		end
	end

	arr
end

function sumhcat_aligned(A::AbstractArray)
	ndims(A) >= 2 || throw(ArgumentError("the array needs to have at least 2 dimensions"))
	A
end