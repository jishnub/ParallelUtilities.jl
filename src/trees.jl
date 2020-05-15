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

function leavesateachlevelfulltree(Nleaves)
	Nnodes = 2Nleaves-1
	Nlevels = floor(Int,log2(Nnodes)) + 1
	Nleaves_lowestlevel = Nnodes - ((1 << (Nlevels - 1)) - 1)

	return Nleaves_lowestlevel, Nnodes, Nlevels
end

function leafrankfoldedtree(Nleaves,leafno)

	@assert(leafno <= Nleaves,"leafno needs to be ⩽ Nleaves")
	
	Nleaves_lowestlevel, Nnodes, Nlevels = 
		leavesateachlevelfulltree(Nleaves)

	if leafno <= Nleaves_lowestlevel
		leafrank = (1 << (Nlevels - 1)) - 1 + leafno
	else
		leafrank = Nnodes - Nleaves + leafno - Nleaves_lowestlevel
	end

	return leafrank
end

function foldedbinarytreefromleaves(T,leaves::AbstractVector{<:Integer})
	
	Nleaves = length(leaves)
	Nleaves_lowestlevel,Nnodes = leavesateachlevelfulltree(Nleaves)

	treeprocs = Vector{Int}(undef,Nnodes)

	# fill in the leaves
	@views treeprocs[end - Nleaves_lowestlevel + 1:end] .= 
		leaves[1:Nleaves_lowestlevel]
	@views treeprocs[end - Nleaves + 1:end - Nleaves_lowestlevel] .= 
		leaves[Nleaves_lowestlevel+1:end]

	# fill in the parent nodes
	for rank in Nnodes-1:-2:2
		p = treeprocs[rank]
		parentrank = rank >> 1
		treeprocs[parentrank] = p
	end

	T(treeprocs)
end

struct SequentialBinaryTree{T<:AbstractVector{<:Integer}} <: BinaryTree
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
		(N >=1) || throw(DomainError(N,
			"need at least one node to create a binary tree"))

		h = floor(Int,log2(N)) # Number of levels of the tree (starting from zero)
		Ninternalnodes = (1 << h) - 1
		Nleaf = N - Ninternalnodes
		Nonechildinternalnodes = (Ninternalnodes > 0) ? rem(Nleaf,2) : 0
		twochildendind = (N-1) >> 1
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
		throw(DomainError(N,
			"need at least one node to create a BinaryTree"))

		new{T}(N,procs)
	end
end

struct SegmentedSequentialBinaryTree{T<:AbstractVector{<:Integer},
	D<:AbstractDict} <: BinaryTree
	#=
		Each node on the cluster will have its own tree that carries out
		a local reduction. There will  be one master node on the cluster that
		will acquire the reduced value on each node. This will be followed 
		by a tree to carry out reduction among the master nodes. The 
		eventual reduced result will be returned to the calling process.
	=#
	N :: Int
	procs :: T
	workersonhosts :: D
	toptree :: SequentialBinaryTree{Vector{Int}}
	nodetreestartindices :: Vector{Int}
end

# Tree with a distribution of hosts specified by workersonhosts
# workersonhosts is a Dict that maps (host=>workers)
function SegmentedSequentialBinaryTree(procs::AbstractVector{<:Integer},
	workersonhosts::AbstractDict{String,<:AbstractVector{<:Integer}})
	
	Np = length(procs)
	Np >= 1 || throw(DomainError(Np,
		"need at least one node to create a BinaryTree"))

	nodes = collect(keys(workersonhosts))
	masternodes = Vector{Int}(undef,length(nodes))
	for (nodeind,node) in enumerate(nodes)
		workersnode = workersonhosts[node]
		nodetree = SequentialBinaryTree(workersnode)
		masternodes[nodeind] = topnode(nodetree).p
	end
	Nleaves = length(masternodes)
	toptree = foldedbinarytreefromleaves(SequentialBinaryTree,masternodes)

	toptreenonleafnodes = length(toptree) - Nleaves
	Nnodestotal = toptreenonleafnodes + length(procs)

	nodetreestartindices = Vector{Int}(undef,length(nodes))
	nodetreestartindices[1] = toptreenonleafnodes + 1
	for (nodeno,node) in enumerate(nodes)
		nodeno == 1 && continue
		prevnode = nodes[nodeno-1]
		nodetreestartindices[nodeno] = nodetreestartindices[nodeno-1] + 
										length(workersonhosts[prevnode])
	end

	SegmentedSequentialBinaryTree(Nnodestotal,procs,workersonhosts,
		toptree,nodetreestartindices)
end

function SegmentedSequentialBinaryTree(procs::AbstractVector{<:Integer})
	workersonhosts = procs_node(procs)
	SegmentedSequentialBinaryTree(procs,workersonhosts)
end

@inline Base.length(tree::BinaryTree) = tree.N
function levels(tree::Union{SequentialBinaryTree,OrderedBinaryTree})
	floor(Int,log2(length(tree))) + 1
end

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

function subtree_rank(tree::SegmentedSequentialBinaryTree,i::Integer)
	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	# node on a subtree at a host
	subnodeno = i - toptreenonleafnodes

	@assert(subnodeno > 0,"i needs to be greater than $(toptreenonleafnodes)")

	# find out which node this lies on
	nptotalprevhosts = 0
	for (host,procs) in tree.workersonhosts
		np = length(procs)
		if subnodeno <= nptotalprevhosts + np
			rankinsubtree = subnodeno - nptotalprevhosts
			subtree = SequentialBinaryTree(tree.workersonhosts[host])
			return subtree,rankinsubtree,nptotalprevhosts
		end
		nptotalprevhosts += np
	end
end

function masternodeindex(tree, p)
	leafno = 0
	for (ind,w) in enumerate(values(tree.workersonhosts))
		if w[1] == p
			leafno = ind
			break
		end
	end
	return leafno
end

function parentnoderank(tree::SegmentedSequentialBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes == 0
		# equivalent to a SequentialBinaryTree
		SBT = SequentialBinaryTree(tree.procs)
		pr = parentnoderank(SBT,i)

	elseif i <= toptreenonleafnodes
		p = tree.toptree.procs[i]
		pr = parentnoderank(tree.toptree,i)
		
	else
		subtree,rankinsubtree,nptotalprevhosts = subtree_rank(tree,i)

		if rankinsubtree == 1
			# masternode
			# parent will be on the top-tree
			p = subtree[rankinsubtree].p
			leafno = masternodeindex(tree,p)
			Nmasternodes = length(keys(tree.workersonhosts))
			leafrank = leafrankfoldedtree(Nmasternodes,leafno)
			pr = parentnoderank(tree.toptree,leafrank)
		else
			# node on a sub-tree
			pr = parentnoderank(subtree,rankinsubtree)
			pr += nptotalprevhosts + toptreenonleafnodes
		end		
	end

	return pr
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
function nchildren(tree::SegmentedSequentialBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes == 0
		SBT = SequentialBinaryTree(tree.procs)
		n = nchildren(SBT,i)

	elseif i <= toptreenonleafnodes
		n = nchildren(tree.toptree,i)

	else
		subtree,rankinsubtree = subtree_rank(tree,i)
		n = nchildren(subtree,rankinsubtree)
	end

	return n
end

topnoderank(::BinaryTree) = 1
function topnoderank(tree::OrderedBinaryTree)
	levels = floor(Int,log2(length(tree)))
	1 << levels # 2^levels
end

topnode(tree::Tree) = tree[topnoderank(tree)]

# Indexing into a OrderedBinaryTree produces a BinaryTreeNode
struct BinaryTreeNode
	p :: Int
	parent :: Int
	nchildren :: Int

	function BinaryTreeNode(p::Int,p_parent::Int,nchildren::Int)
		(0 <= nchildren <= 2) || 
		throw(DomainError(nchildren,
			"attempt to construct a binary tree with $nchildren children"))

		new(p,p_parent,nchildren)
	end
end

@inline nchildren(b::BinaryTreeNode) = b.nchildren

function Base.getindex(tree::BinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	procs = tree.procs
	
	p = procs[i]
	pr = parentnoderank(tree,i)
	p_parent = procs[pr]
	n = nchildren(tree,i)

	BinaryTreeNode(p,p_parent,n)
end

function Base.getindex(tree::SegmentedSequentialBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes == 0
		# equivalent to a SequentialBinaryTree
		SBT = SequentialBinaryTree(tree.procs)
		return SBT[i]
	elseif i <= toptreenonleafnodes
		p = tree.toptree.procs[i]
		pr = parentnoderank(tree.toptree,i)
		p_parent = tree.toptree.procs[pr]
		n = nchildren(tree.toptree,i)
		return BinaryTreeNode(p,p_parent,n)
	else
		subtree,rankinsubtree = subtree_rank(tree,i)

		p = subtree.procs[rankinsubtree]
		n = nchildren(subtree,rankinsubtree)

		if rankinsubtree == 1
			# masternode
			# parent will be on the top tree
			Nmasternodes = length(keys(tree.workersonhosts))
			leafno = masternodeindex(tree,p)
			leafrank = leafrankfoldedtree(Nmasternodes,leafno)
			pr = parentnoderank(tree.toptree,leafrank)
			p_parent = tree.toptree.procs[pr]
		else
			# node on a sub-tree
			pr = parentnoderank(subtree,rankinsubtree)
			p_parent = subtree.procs[pr]
		end
		return BinaryTreeNode(p,p_parent,n)
	end
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

		(0 <= nchildren <= 2) || 
		throw(DomainError(nchildren,
			"attempt to construct a binary tree with $nchildren children"))
	
		new{Tmap,Tred}(p,selfchannels,parentchannels,childrenchannels,nchildren)
	end
end
@inline nchildren(b::BranchChannel) = b.nchildren

function BranchChannel(p::Int,::Type{Tmap},
	parentchannels::RemoteChannelContainer{Tred},
	nchildren::Int) where {Tmap,Tred}

	(0 <= nchildren <= 2) || 
	throw(DomainError(nchildren,
		"attempt to construct a binary tree with $nchildren children"))

	selfchannels = RemoteChannelContainer{Tmap}(1,p)
	childrenchannels = RemoteChannelContainer{Tred}(nchildren,p)
	BranchChannel(p,selfchannels,parentchannels,childrenchannels,nchildren)
end

function BranchChannel{Tmap,Tred}(p::Int,nchildren::Int) where {Tmap,Tred}
	(0 <= nchildren <= 2) || 
	throw(DomainError(nchildren,
		"attempt to construct a binary tree with $nchildren children"))

	parentchannels = RemoteChannelContainer{Tred}(1,p)
	BranchChannel(p,Tmap,parentchannels,nchildren)
end

function Base.show(io::IO, b::BranchChannel)
	N = nchildren(b)
	p_parent = b.parentchannels.out.where
	p = b.p

	if N == 2
		str = "Branch: "*string(p_parent)*" ← "*string(p)*" ⇇ 2 children"
	elseif N == 1
		str = "Branch: "*string(p_parent)*" ← "*string(p)*" ← 1 child"
	else
		str = "Leaf  : "*string(p_parent)*" ← "*string(p)
	end

	print(io,str)
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

function permuteworkers(w,p,toprank)
	(length(w) == 1) && return w

	rank = findfirst(isequal(p),w)
	if !isnothing(rank) && rank != toprank
		# Move the calling worker to the top of the tree
		w_rest = w[w .!= p]
		insert!(w_rest,toprank,p)
		return w_rest
	else
		return w
	end
end

function createbranchchannels!(branches,::Type{Tmap},::Type{Tred},tree::OrderedBinaryTree,
	superbranch::BranchChannel) where {Tmap,Tred}

	top = topnoderank(tree)
	topnode = tree[top]
	N = nchildren(topnode)
	p = topnode.p

	topbranchchannels = BranchChannel(p,Tmap,superbranch.childrenchannels,N)
	branches[top] = topbranchchannels

	length(tree) == 1 && return
	
	left_inds = 1:top-1
	left_child = OrderedBinaryTree(@view tree.procs[left_inds])
	createbranchchannels!(@view(branches[left_inds]),Tmap,Tred,left_child,topbranchchannels)

	if top < length(tree)
		right_inds = top+1:length(tree)
		right_child = OrderedBinaryTree(@view tree.procs[right_inds])
		createbranchchannels!(@view(branches[right_inds]),Tmap,Tred,right_child,topbranchchannels)
	end
end
function createbranchchannels(::Type{Tmap},::Type{Tred},tree::OrderedBinaryTree) where {Tmap,Tred}

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
function createbranchchannels!(branches,::Type{Tmap},
	::Type{Tred},tree::SequentialBinaryTree, 
	finalnoderank = length(tree)) where {Tmap,Tred}

	length(branches) < 2 && return

	for noderank = 2:finalnoderank
		node = tree[noderank]
		p = node.p
		parentnodebranches = branches[parentnoderank(tree,noderank)]
		parentchannels = parentnodebranches.childrenchannels
		b = BranchChannel(p,Tmap,parentchannels,nchildren(node))
		branches[noderank] = b
	end
end
function createbranchchannels(::Type{Tmap},::Type{Tred},tree::SequentialBinaryTree) where {Tmap,Tred}

	branches = Vector{BranchChannel{Tmap,Tred}}(undef,length(tree))

	# the topmost node has to be created separately as 
	# it is its own parent
	top = topnoderank(tree)
	topnode = tree[top]
	N = nchildren(topnode)
	p = topnode.p
	topmostbranch = BranchChannel{Tmap,Tred}(p,N)
	branches[top] = topmostbranch

	createbranchchannels!(branches,Tmap,Tred,tree)

	return branches
end
function createbranchchannels(::Type{Tmap},::Type{Tred},tree::SegmentedSequentialBinaryTree) where {Tmap,Tred}

	nodes = collect(keys(tree.workersonhosts))
	toptree = tree.toptree
	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(toptree) - Nmasternodes

	branches = Vector{BranchChannel{Tmap,Tred}}(undef,length(tree))

	# populate the top tree other than the masternodes
	# This is only run if there are multiple hosts
	if toptreenonleafnodes > 0
		top = topnoderank(toptree)
		topbranch = toptree[top]
		N = nchildren(topbranch)
		p = topbranch.p
		topmostbranch = BranchChannel{Tmap,Tred}(p,N)
		branches[top] = topmostbranch
		createbranchchannels!(branches,Tmap,Tred,toptree,
			toptreenonleafnodes)
	end

	for (nodeno,node) in enumerate(nodes)
		# Top node for each subtree (a masternode)
		workersnode = tree.workersonhosts[node]
		nodetree = SequentialBinaryTree(workersnode)
		topbranch = nodetree[topnoderank(nodetree)]
		p = topbranch.p

		if toptreenonleafnodes > 0
			# inherit from the parent node
			leafno = masternodeindex(tree,p)
			leafrank = leafrankfoldedtree(Nmasternodes,leafno)
			parentrank = parentnoderank(toptree,leafrank)
			parentnodebranches = branches[parentrank]
			parentchannels = parentnodebranches.childrenchannels
		else
			# This happens if there is only one host, 
			# in which case there's nothing to inherit.
			# In this case there's no difference between a 
			# SegmentedSequentialBinaryTree and a SequentialBinaryTree
			# The top node is created separately as it is its own parent
			parentchannels = RemoteChannelContainer{Tred}(1,p)
		end

		b = BranchChannel(p,Tmap,parentchannels,nchildren(topbranch))
		nodetreestartindex = tree.nodetreestartindices[nodeno]
		branches[nodetreestartindex] = b

		# Populate the rest of the tree
		subtreeinds = StepRangeLen(nodetreestartindex,1,length(nodetree))
		branchesnode = @view branches[subtreeinds]

		createbranchchannels!(branchesnode,Tmap,Tred,nodetree)		
	end

	return branches
end

function createbranchchannels(::Type{Tmap},::Type{Tred},
	iterators::Tuple,::Type{T}) where {Tmap,Tred,T<:Tree}

	w = workersactive(iterators)
	tree = T(w)
	branches = createbranchchannels(Tmap,Tred,tree)
	tree,branches
end
function createbranchchannels(iterators::Tuple,::Type{T}) where {T<:Tree}
	createbranchchannels(Any,Any,iterators,T)
end

function topbranch(tree::Tree,branches::Vector{<:BranchChannel})
	branches[topnoderank(tree)]
end