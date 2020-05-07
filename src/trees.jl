const RemoteChannelContainer{T} = NamedTuple{(:out, :err),Tuple{RemoteChannel{Channel{T}},RemoteChannel{Channel{Bool}}}}

@inline Base.eltype(::RemoteChannelContainer{T}) where {T} = T

function RemoteChannelContainer{T}(n::Int,p::Int) where {T}
	out = RemoteChannel(()->Channel{T}(n),p)
    err = RemoteChannel(()->Channel{Bool}(n),p)
    RemoteChannelContainer{T}((out,err))
end
RemoteChannelContainer{T}(n::Int) where {T} = RemoteChannelContainer{T}(n,myid())
RemoteChannelContainer(n::Int,p::Int) = RemoteChannelContainer{Any}(n,p)
RemoteChannelContainer(n::Int) = RemoteChannelContainer{Any}(n,myid())

abstract type Tree end
abstract type BinaryTree <: Tree end

struct SequentialBinaryTree{T} <: BinaryTree
	#= Tree of the form 
					1
			2				3
		4		5		6		7
	8		9
	=#
	N :: Int # total number of nodes
	twochildendind :: Int
	onechildendind :: Int
	procs :: T

	function SequentialBinaryTree(procs::T) where {T<:AbstractVector{Int}}

		N = length(procs)
		(N >=1) || throw(DomainError(N,"need at least 1 node to create a binary tree"))

		h = floor(Int,log2(N)) # Number of levels of the tree (starting from zero)
		Ninternalnodes = 2^h - 1
		Nleaf = N - Ninternalnodes
		Nonechildinternalnodes = (Ninternalnodes > 0) ? rem(Nleaf,2) : 0
		twochildendind = div(N-1,2)
		onechildstartind = twochildendind + 1
		onechildendind = onechildstartind + Nonechildinternalnodes - 1

		new{T}(N,twochildendind,onechildendind,procs)
	end
end

struct OrderedBinaryTree{T<:AbstractVector{<:Integer}} <: BinaryTree
	#= Tree of the form

							8
				4						9
		2				6
	1		3		5		7
					
	The left branch has smaller numbers than the node, and the right
	branch has larger numbers
	=#

	N :: Int
	procs :: T

	function OrderedBinaryTree(procs::T) where {T<:AbstractVector{<:Integer}}
		N = length(procs)
		N >= 1 || 
			throw(DomainError(N,"Need at least one node to create a BinaryTree"))
		new{T}(N,procs)
	end
end

@inline Base.length(tree::BinaryTree) = tree.N
levels(tree::BinaryTree) = floor(Int,log2(length(tree))) + 1

Base.summary(io::IO,b::Tree) = print(io,length(b),"-node ",typeof(b))

function levelfromtop(tree::OrderedBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))
	
	top = topnoderank(tree)
	if i == top
		return 1
	elseif i < top
		subrange = 1:top - 1
	else
		subrange = top+1:length(tree)
	end
	subtree = OrderedBinaryTree(subrange)
	subindex = searchsortedfirst(subrange,i)
	1 + levelfromtop(subtree,subindex)
end
function levelfromtop(tree::SequentialBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))
	floor(Int,log2(i)) + 1
end

function parentnoderank(tree::OrderedBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	# The topmost node is its own parent
	length(tree) == 1 && return 1
	top = topnoderank(tree)
	length(tree) > 1 && i == top && return top

	if i < top
		# left branch, fully formed
		level = trailing_zeros(i)
		ired = i >> level # i / 2^level
		# ired is necessarily an odd number
		pow2level = 1 << level # 2^level

		# sgn is +1 if mod(ired,4) = 1, -1 if mod(ired,4) = 3
		sgn = 2 - mod(ired,4)
		return i + sgn * pow2level
	elseif i > top
		# right branch, possibly partially formed
		# Carry out a recursive search
		subtreeprocs = top+1:length(tree)
		subtree = OrderedBinaryTree(subtreeprocs)
		subind = searchsortedfirst(subtreeprocs,i)
		if subind == topnoderank(subtree)
			# This catches the case of there only being a leaf node
			# in the sub-tree
			return top
		elseif length(subtreeprocs) == 3
			# don't subdivide to 1-node trees
			# this lets us avoid confusing this with the case of 
			# the entire tree having only 1 node
			return subtreeprocs[2]
		end
		pid = parentnoderank(subtree,subind)
		return subtreeprocs[pid]
	end
end
function parentnoderank(tree::SequentialBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	# only one node
	i == 1 && return 1
	i >> 1 # div(i,2)
end

function nchildren(tree::OrderedBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))
	
	if isodd(i)
		0
	elseif i == length(tree)
		1
	else
		2
	end
end
function nchildren(tree::SequentialBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	if i <= tree.twochildendind
		2
	elseif i <= tree.onechildendind
		1
	else
		0
	end
end

function topnoderank(tree::OrderedBinaryTree)
	levels = floor(Int,log2(length(tree)))
	1 << levels # 2^levels
end
@inline topnoderank(tree::SequentialBinaryTree) = 1

# Indexing into a OrderedBinaryTree produces a BinaryTreeNode
struct BinaryTreeNode
	p :: Int
	parent :: Int
	nchildren :: Int

	function BinaryTreeNode(p::Int,p_parent::Int,nchildren::Int)
		(0 <= nchildren <= 2) || throw(BinaryTreeError(nchildren))
		new(p,p_parent,nchildren)
	end
end

@inline nchildren(b::BinaryTreeNode) = b.nchildren

function Base.getindex(tree::BinaryTree,i::Int)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	procs = tree.procs
	
	p = procs[i]
	pr = parentnoderank(tree,i)
	p_parent = procs[pr]
	n = nchildren(tree,i)

	BinaryTreeNode(p,p_parent,n)
end

# Branches between nodes

struct BranchChannel{Tmap,Tred}
	p :: Int
	selfchannels :: RemoteChannelContainer{Tmap}
	parentchannels :: RemoteChannelContainer{Tred}
	childrenchannels :: RemoteChannelContainer{Tred}
	nchildren :: Int

	function BranchChannel(p::Int,selfchannels::RemoteChannelContainer{Tmap},
		parentchannels::RemoteChannelContainer{Tred},
		childrenchannels::RemoteChannelContainer{Tred},
		nchildren::Int) where {Tmap,Tred}

		(0 <= nchildren <= 2) || throw(BinaryTreeError(nchildren))
	
		new{Tmap,Tred}(p,selfchannels,parentchannels,childrenchannels,nchildren)
	end
end
@inline nchildren(b::BranchChannel) = b.nchildren

function BranchChannel(p::Int,::Type{Tmap},
	parentchannels::RemoteChannelContainer{Tred},
	nchildren::Int) where {Tmap,Tred}

	(0 <= nchildren <= 2) || throw(BinaryTreeError(nchildren))
	selfchannels = RemoteChannelContainer{Tmap}(1,p)
	childrenchannels = RemoteChannelContainer{Tred}(nchildren,p)
	BranchChannel(p,selfchannels,parentchannels,childrenchannels,nchildren)
end

function BranchChannel{Tmap,Tred}(p::Int,nchildren::Int) where {Tmap,Tred}
	(0 <= nchildren <= 2) || throw(BinaryTreeError(nchildren))
	parentchannels = RemoteChannelContainer{Tred}(1,p)
	BranchChannel(p,Tmap,parentchannels,nchildren)
end
function BranchChannel{Tmap,Tred}(nchildren::Int) where {Tmap,Tred}
	BranchChannel{Tmap,Tred}(myid(),nchildren)
end

function finalize_except_wherewhence(r::RemoteChannel)
	if (myid() != r.where) && (myid() != r.whence)
		finalize(r)
	end
end
function finalize_except_wherewhence(r::RemoteChannelContainer)
	finalize_except_wherewhence.((r.out, r.err))
end

function Base.finalize(r::RemoteChannelContainer)
	finalize.((r.out, r.err))
end

function Base.finalize(bc::BranchChannel)
	finalize.((bc.selfchannels, bc.childrenchannels))
	finalize_except_wherewhence(bc.parentchannels)
end

function createbranchchannels!(branches,::Type{Tmap},::Type{Tred},
	tree::OrderedBinaryTree,topbranch::BranchChannel) where {Tmap,Tred}

	top = topnoderank(tree)
	topnode = tree[top]
	N = nchildren(topnode)
	p = topnode.p

	parentchannels = topbranch.childrenchannels
	topchannels = BranchChannel(p,Tmap,parentchannels,N)
	branches[top] = topchannels

	length(tree) == 1 && return
	
	left_inds = 1:top-1
	left_child = OrderedBinaryTree(@view tree.procs[left_inds])
	createbranchchannels!(@view(branches[left_inds]),Tmap,Tred,left_child,topchannels)

	if top < length(tree)
		right_inds = top+1:length(tree)
		right_child = OrderedBinaryTree(@view tree.procs[right_inds])
		createbranchchannels!(@view(branches[right_inds]),Tmap,Tred,right_child,topchannels)
	end
end

function createbranchchannels(::Type{Tmap},::Type{Tred},
	tree::OrderedBinaryTree) where {Tmap,Tred}

	branches = Vector{BranchChannel{Tmap,Tred}}(undef,length(tree))

	# the topmost node has to be created separately as 
	# its children will be linked to itself
	top = topnoderank(tree)
	topnode = tree[top]
	N = nchildren(topnode)
	p = topnode.p
	topmostbranch = BranchChannel{Tmap,Tred}(p,N)
	branches[top] = topmostbranch 

	length(tree) == 1 && return branches
	
	left_child = OrderedBinaryTree(@view tree.procs[1:top-1])
	createbranchchannels!(@view(branches[1:top-1]),Tmap,Tred,left_child,topmostbranch)
	
	if top < length(tree)
		right_child = OrderedBinaryTree(@view tree.procs[top+1:end])
		createbranchchannels!(@view(branches[top+1:end]),Tmap,Tred,right_child,topmostbranch)
	end

	return branches
end
function createbranchchannels(::Type{Tmap},::Type{Tred},
	tree::SequentialBinaryTree) where {Tmap,Tred}

	branches = Vector{BranchChannel{Tmap,Tred}}(undef,length(tree))

	# the topmost node has to be created separately as 
	# its children will be linked to itself
	noderank = topnoderank(tree)
	topnode = tree[noderank]
	N = nchildren(topnode)
	p = topnode.p
	branches[noderank] = BranchChannel{Tmap,Tred}(p,N)

	for noderank = 2:length(tree)
		node = tree[noderank]
		p = node.p
		parentnodebranches = branches[parentnoderank(tree,noderank)]
		parentchannels = parentnodebranches.childrenchannels
		b = BranchChannel(p,Tmap,parentchannels,nchildren(node))
		branches[noderank] = b
	end

	return branches
end

function createbranchchannels(::Type{Tmap},::Type{Tred},
	iterators::Tuple,::Type{T}) where {Tmap,Tred,T<:Tree}

	tree = T(workersactive(iterators))
	branches = createbranchchannels(Tmap,Tred,tree)
	tree,branches
end
function createbranchchannels(iterators::Tuple,::Type{T}) where {T<:Tree}
	createbranchchannels(Any,Any,iterators,T)
end

function topnode(tree::Tree,branches::Vector{<:BranchChannel})
	branches[topnoderank(tree)]
end