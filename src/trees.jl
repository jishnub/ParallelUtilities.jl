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

		Ninternalnodes = prevpow(2,N) - 1
		Nleaf = N - Ninternalnodes
		Nonechildinternalnodes = (Ninternalnodes > 0) ? rem(Nleaf,2) : 0
		twochildendind = div(N-1, 2)
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

abstract type SegmentedBinaryTree <: BinaryTree end

struct SegmentedSequentialBinaryTree{T<:AbstractVector{<:Integer},
	D<:AbstractDict} <: SegmentedBinaryTree
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

struct SegmentedOrderedBinaryTree{T<:AbstractVector{<:Integer},
	D<:AbstractDict} <: SegmentedBinaryTree
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
	toptree :: OrderedBinaryTree{Vector{Int}}
	nodetreestartindices :: Vector{Int}
end

function leavesateachlevelfulltree(::Type{<:SequentialBinaryTree},Nleaves)
	Nnodes = 2Nleaves-1
	Nlevels = levels(Nnodes)
	Nleaves_lowestlevel = Nnodes - ((1 << (Nlevels - 1)) - 1)

	return Nleaves_lowestlevel, Nnodes, Nlevels
end

function leafrankfoldedtree(::SequentialBinaryTree,Nleaves,leafno)

	@assert(leafno <= Nleaves,"leafno needs to be ⩽ Nleaves")
	
	Nleaves_lowestlevel, Nnodes, Nlevels = 
		leavesateachlevelfulltree(SequentialBinaryTree,Nleaves)

	if leafno <= Nleaves_lowestlevel
		leafrank = (1 << (Nlevels - 1)) - 1 + leafno
	else
		leafrank = Nnodes - Nleaves + leafno - Nleaves_lowestlevel
	end

	return leafrank
end

function leafrankfoldedtree(::OrderedBinaryTree,Nleaves,leafno)
	@assert(leafno <= Nleaves,"leafno needs to be ⩽ Nleaves")
	leafrank = 2leafno - 1
end

function foldedbinarytreefromleaves(::Type{SequentialBinaryTree},leaves)
	Nleaves = length(leaves)
	Nleaves_lowestlevel,Nnodes = 
	leavesateachlevelfulltree(SequentialBinaryTree,Nleaves)

	treeprocs = Vector{Int}(undef,Nnodes)

	# fill in the leaves
	@views treeprocs[end - Nleaves_lowestlevel + 1:end] .= 
		leaves[1:Nleaves_lowestlevel]
	@views treeprocs[end - Nleaves + 1:end - Nleaves_lowestlevel] .= 
		leaves[Nleaves_lowestlevel+1:end]

	# fill in the parent nodes
	for rank in Nnodes-1:-2:2
		p = treeprocs[rank]
		parentrank = div(rank,2)
		treeprocs[parentrank] = p
	end

	SequentialBinaryTree(treeprocs)
end

function foldedbinarytreefromleaves(::Type{OrderedBinaryTree},leaves)
	Nleaves = length(leaves)
	Nnodes = 2Nleaves-1

	allnodes = Vector{Int}(undef,Nnodes)
	foldedbinarytreefromleaves!(OrderedBinaryTree,allnodes,leaves)

	OrderedBinaryTree(allnodes)
end

function foldedbinarytreefromleaves!(::Type{OrderedBinaryTree},allnodes,leaves)
	top = topnoderank(OrderedBinaryTree(1:length(allnodes)))
	allnodes[top] = first(leaves)

	length(allnodes) == 1 && return

	Nnodes_left = top - 1
	Nleaves_left = div( Nnodes_left + 1 , 2)
	Nleaves_right = length(leaves) - Nleaves_left

	if Nleaves_left > 0
		leaves_left = @view leaves[1:Nleaves_left]
		leftnodes = @view allnodes[1:Nnodes_left]
		foldedbinarytreefromleaves!(OrderedBinaryTree,leftnodes,leaves_left)
	end

	if Nleaves_right > 0
		leaves_right = @view leaves[end - Nleaves_right + 1:end]
		rightnodes = @view allnodes[top + 1:end]
		foldedbinarytreefromleaves!(OrderedBinaryTree,rightnodes,leaves_right)
	end
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

function SegmentedOrderedBinaryTree(procs::AbstractVector{<:Integer},
	workersonhosts::AbstractDict{String,<:AbstractVector{<:Integer}})
	
	Np = length(procs)
	Np >= 1 || throw(DomainError(Np,
		"need at least one node to create a BinaryTree"))

	nodes = collect(keys(workersonhosts))
	masternodes = Vector{Int}(undef,length(nodes))
	for (nodeind,node) in enumerate(nodes)
		workersnode = workersonhosts[node]
		nodetree = OrderedBinaryTree(workersnode)
		masternodes[nodeind] = topnode(nodetree).p
	end
	Nleaves = length(masternodes)
	toptree = foldedbinarytreefromleaves(OrderedBinaryTree,masternodes)

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

	SegmentedOrderedBinaryTree(Nnodestotal,procs,workersonhosts,
		toptree,nodetreestartindices)
end

function SegmentedOrderedBinaryTree(procs::AbstractVector{<:Integer})
	workersonhosts = procs_node(procs)
	SegmentedOrderedBinaryTree(procs,workersonhosts)
end

# for a single host there are no segments
function unsegmentedtree(::Type{<:SegmentedSequentialBinaryTree})
	SequentialBinaryTree
end
function unsegmentedtree(::Type{<:SegmentedOrderedBinaryTree})
	OrderedBinaryTree
end
function unsegmentedtree(tree::SegmentedBinaryTree)
	T = unsegmentedtree(typeof(tree))
	T(tree.procs)
end

@inline Base.length(tree::BinaryTree) = tree.N
function levels(tree::Union{SequentialBinaryTree,OrderedBinaryTree})
	levels(length(tree))
end
levels(n::Integer) = floor(Int,log2(n)) + 1

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
	div(i,2)
end

function subtree_rank(tree::SegmentedBinaryTree,i::Integer)
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
			T = unsegmentedtree(typeof(tree))
			subtree = T(tree.workersonhosts[host])
			return subtree,rankinsubtree,nptotalprevhosts
		end
		nptotalprevhosts += np
	end
end

function masternodeindex(tree::SegmentedBinaryTree, p)
	leafno = 0
	T = unsegmentedtree(typeof(tree))
	for (ind,w) in enumerate(values(tree.workersonhosts))
		subtree = T(w)
		top = topnoderank(subtree)
		if w[top] == p
			leafno = ind
			break
		end
	end
	return leafno
end

toptree_to_fulltree_index(::SequentialBinaryTree, i) = i
toptree_to_fulltree_index(::OrderedBinaryTree, i) = div(i,2)

fulltree_to_toptree_index(::SequentialBinaryTree, i) = i
fulltree_to_toptree_index(::OrderedBinaryTree, i) = 2i

function parentnoderank(tree::SegmentedBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes == 0
		pr = parentnoderank(unsegmentedtree(tree),i)

	elseif i <= toptreenonleafnodes
		#= In a SegmentedSequentialBinaryTree the leading indices
		are the parent nodes of the top tree, so ind = i
		In a SegmentedOrderedBinaryTree, the leaves are removed 
		from the top tree, so only even numbers are left.
		In this case, index i of the full tree refers to index 2i of the 
		top tree, so ind = 2i
		=#
		ind = fulltree_to_toptree_index(tree.toptree,i)
		p = tree.toptree[ind].p
		#= Compute the parent of the node with rank ind on the top tree.
		In a SegmentedSequentialBinaryTree this is what we want.
		In a SegmentedOrderedBinaryTree, we need to convert this back to 
		the index of the full tree, that is div(pr,2)
		=# 
		pr_top = parentnoderank(tree.toptree,ind)
		pr = toptree_to_fulltree_index(tree.toptree, pr_top)
		
	else
		subtree,rankinsubtree,nptotalprevhosts = subtree_rank(tree,i)

		if rankinsubtree == topnoderank(subtree)
			# masternode
			# parent will be on the top-tree
			p = subtree[rankinsubtree].p
			leafno = masternodeindex(tree,p)
			Nmasternodes = length(keys(tree.workersonhosts))
			leafrank = leafrankfoldedtree(tree.toptree, Nmasternodes,leafno)
			pr_top = parentnoderank(tree.toptree, leafrank)
			# Convert back to the rank on the full tree where the 
			# leaves of the top tree aren't stored.
			pr = toptree_to_fulltree_index(tree.toptree, pr_top)
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
function nchildren(tree::SegmentedBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes == 0
		n = nchildren(unsegmentedtree(tree),i)

	elseif i <= toptreenonleafnodes
		# The top-tree is a full binary tree.
		# Since the leaves aren't stored, every parent node
		# has 2 children
		n = 2
	else
		subtree,rankinsubtree = subtree_rank(tree,i)
		n = nchildren(subtree,rankinsubtree)
	end

	return n
end

topnoderank(::BinaryTree) = 1
function topnoderank(tree::OrderedBinaryTree)
	1 << (levels(tree) - 1)
end
function topnoderank(tree::SegmentedOrderedBinaryTree)
	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes > 0
		tnr_top = topnoderank(tree.toptree)
		tnr = toptree_to_fulltree_index(tree.toptree, tnr_top)
	else
		tnr = topnoderank(OrderedBinaryTree(tree.procs))
	end
	return tnr
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

function Base.show(io::IO,b::BinaryTreeNode)
	print(io,
		"BinaryTreeNode(p = $(b.p),"*
		" parent = $(b.parent), nchildren = $(b.nchildren))")
end

@inline nchildren(b::BinaryTreeNode) = b.nchildren

function Base.getindex(tree::Tree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	procs = tree.procs
	
	p = procs[i]
	pr = parentnoderank(tree,i)
	p_parent = procs[pr]
	n = nchildren(tree,i)

	BinaryTreeNode(p,p_parent,n)
end

function Base.getindex(tree::SegmentedBinaryTree,i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree,i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes == 0
		return unsegmentedtree(tree)[i]

	elseif i <= toptreenonleafnodes
		#= In a SegmentedSequentialBinaryTree the leading indices
		are the parent nodes of the top tree, so ind = i
		In a SegmentedOrderedBinaryTree, the leaves are removed 
		from the top tree, so only even numbers are left.
		In this case, index i of the full tree refers to index 2i of the 
		top tree, so ind = 2i
		=#
		ind = fulltree_to_toptree_index(tree.toptree,i)
		p = tree.toptree[ind].p
		pr_top = parentnoderank(tree.toptree,ind)
		p_parent = tree.toptree[pr_top].p
		n = 2
		return BinaryTreeNode(p,p_parent,n)
	else
		subtree,rankinsubtree = subtree_rank(tree,i)

		p = subtree[rankinsubtree].p
		n = nchildren(subtree,rankinsubtree)

		if rankinsubtree == topnoderank(subtree)
			# masternode
			# parent will be on the top tree
			Nmasternodes = length(keys(tree.workersonhosts))
			leafno = masternodeindex(tree,p)
			leafrank = leafrankfoldedtree(tree.toptree, Nmasternodes,leafno)
			pr_top = parentnoderank(tree.toptree, leafrank)
			p_parent = tree.toptree[pr_top].p
		else
			# node on a sub-tree
			pr = parentnoderank(subtree,rankinsubtree)
			p_parent = subtree[pr].p
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

function BranchChannel(p::Integer,Tmap,
	parentchannels::RemoteChannelContainer{Tred},
	nchildren::Int) where {Tred}

	(0 <= nchildren <= 2) || 
	throw(DomainError(nchildren,
		"attempt to construct a binary tree with $nchildren children"))

	Texp = Tuple{RemoteChannelContainer{Tmap},
		RemoteChannelContainer{Tred}}

	selfchannels, childrenchannels = @sync begin
		selftask = @async RemoteChannelContainer{Tmap}(1,p)
		childtask = @async RemoteChannelContainer{Tred}(nchildren,p)
		fetch.((selftask,childtask)) :: Texp
	end
	BranchChannel(p,selfchannels,parentchannels,childrenchannels,nchildren)
end

function BranchChannel{Tmap,Tred}(p::Integer,nchildren::Integer) where {Tmap,Tred}
	(0 <= nchildren <= 2) || 
	throw(DomainError(nchildren,
		"attempt to construct a binary tree with $nchildren children"))

	Texp = Tuple{RemoteChannelContainer{Tred},
		RemoteChannelContainer{Tmap},
		RemoteChannelContainer{Tred}}

	parentchannels, selfchannels, childrenchannels = 
	@sync begin
		parenttask = @async RemoteChannelContainer{Tred}(1,p)
		selftask = @async RemoteChannelContainer{Tmap}(1,p)
		childtask = @async RemoteChannelContainer{Tred}(nchildren,p)
		fetch.((parenttask,selftask,childtask)) :: Texp
	end
	BranchChannel(p,selfchannels,parentchannels,childrenchannels,nchildren)
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

function createbranchchannels!(branches,Tmap,Tred,tree::OrderedBinaryTree,
	superbranch::BranchChannel)

	top = topnoderank(tree)
	topnode = tree[top]
	N = nchildren(topnode)
	p = topnode.p

	topbranchchannels = BranchChannel(p,Tmap,superbranch.childrenchannels,N)
	branches[top] = topbranchchannels

	length(tree) == 1 && return
	
	left_inds = 1:top-1
	right_inds = top+1:length(tree)

	@sync begin
		@async if !isempty(left_inds)
			left_child = OrderedBinaryTree(@view tree.procs[left_inds])
			createbranchchannels!(@view(branches[left_inds]),
				Tmap,Tred,left_child,topbranchchannels)
		end
		@async if !isempty(right_inds)
			right_child = OrderedBinaryTree(@view tree.procs[right_inds])
			createbranchchannels!(@view(branches[right_inds]),Tmap,Tred,right_child,topbranchchannels)
		end
	end
	nothing 
end
function createbranchchannels(Tmap,Tred,tree::OrderedBinaryTree)

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
	
	left_inds = 1:top-1
	right_inds = top+1:length(tree)

	@sync begin
		@async if !isempty(left_inds)
			left_child = OrderedBinaryTree(@view tree.procs[left_inds])
			createbranchchannels!(@view(branches[left_inds]),
				Tmap,Tred,left_child,topmostbranch)
		end
		@async if !isempty(right_inds)
			right_child = OrderedBinaryTree(@view tree.procs[right_inds])
			createbranchchannels!(@view(branches[right_inds]),
				Tmap,Tred,right_child,topmostbranch)
		end
	end

	return branches
end

function createbranchchannels!(branches,Tmap,Tred,tree::SequentialBinaryTree, 
	finalnoderank = length(tree))

	length(branches) < 2 && return

	# make sure that the parent nodes are populated
	parentfilled = [Base.Event() for i=1:tree.onechildendind]

	@sync for noderank in 2:finalnoderank
		@async begin
			node = tree[noderank]
			p = node.p
			pnr = parentnoderank(tree,noderank)
			# The first node is filled, no need to wait for it
			if pnr > 1
				# Wait otherwise for the parent to get filled
				wait(parentfilled[pnr])
			end
			parentnodebranches = branches[pnr]
			parentchannels = parentnodebranches.childrenchannels
			b = BranchChannel(p,Tmap,parentchannels,nchildren(node))
			branches[noderank] = b
			# If this is a parent node then notify that it's filled
			if noderank <= tree.onechildendind
				notify(parentfilled[noderank])
			end
		end
	end
end
function createbranchchannels(Tmap,Tred,tree::SequentialBinaryTree)

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

function createbranchchannels(Tmap,Tred,tree::SegmentedSequentialBinaryTree)

	nodes = keys(tree.workersonhosts)
	toptree = tree.toptree
	Nmasternodes = length(nodes)
	toptreenonleafnodes = length(toptree) - Nmasternodes

	branches = Vector{BranchChannel{Tmap,Tred}}(undef,length(tree))

	# populate the top tree other than the masternodes
	# This is only run if there are multiple hosts
	if toptreenonleafnodes > 0
		top = topnoderank(toptree)
		topnode_toptree = toptree[top]
		N = nchildren(topnode_toptree)
		topmostbranch = BranchChannel{Tmap,Tred}(topnode_toptree.p,N)
		branches[top] = topmostbranch
		createbranchchannels!(branches,Tmap,Tred,toptree,
			toptreenonleafnodes)
	end

	@sync for (nodeno,node) in enumerate(nodes)
		@async begin
			# Top node for each subtree (a masternode)
			workersnode = tree.workersonhosts[node]
			nodetree = SequentialBinaryTree(workersnode)
			topnode_nodetree = topnode(nodetree)
			p = topnode_nodetree.p

			if toptreenonleafnodes > 0
				# inherit from the parent node
				leafno = masternodeindex(tree,p)
				leafrank = leafrankfoldedtree(tree.toptree, Nmasternodes,leafno)
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

			b = BranchChannel(p,Tmap,parentchannels,nchildren(topnode_nodetree))
			nodetreestartindex = tree.nodetreestartindices[nodeno]
			branches[nodetreestartindex] = b

			# Populate the rest of the tree
			subtreeinds = StepRangeLen(nodetreestartindex,1,length(nodetree))
			branchesnode = @view branches[subtreeinds]

			createbranchchannels!(branchesnode,Tmap,Tred,nodetree)	
		end
	end

	return branches
end

function createbranchchannels(Tmap,Tred,tree::SegmentedOrderedBinaryTree)

	nodes = keys(tree.workersonhosts)
	toptree = tree.toptree
	Nmasternodes = length(nodes)
	toptreenonleafnodes = length(toptree) - Nmasternodes

	branches = Vector{BranchChannel{Tmap,Tred}}(undef,length(tree))

	# populate the top tree other than the masternodes
	# This is only run if there are multiple hosts
	if toptreenonleafnodes > 0
		topnoderank_toptree = topnoderank(toptree)
		topnode_toptree = toptree[topnoderank_toptree]
		N = nchildren(topnode_toptree)
		topmostbranch = BranchChannel{Tmap,Tred}(topnode_toptree.p,N)
		branches[topnoderank_toptree] = topmostbranch
		
		left_inds = 1:topnoderank_toptree-1
		right_inds = topnoderank_toptree+1:length(toptree)

		@sync begin
			@async if !isempty(left_inds)
				left_child = OrderedBinaryTree(@view toptree.procs[left_inds])
				createbranchchannels!(@view(branches[left_inds]),
					Tmap,Tred,left_child,topmostbranch)
			end

			@async if !isempty(right_inds)
				right_child = OrderedBinaryTree(@view toptree.procs[right_inds])
				createbranchchannels!(@view(branches[right_inds]),
					Tmap,Tred,right_child,topmostbranch)
			end
		end

		#= Remove the leaves from the top tree (masternodes).
		They are the top nodes of the individual trees at the hosts.
		They will be created separately and linked to the top tree.
		=#
		for i = 1:toptreenonleafnodes
			branches[i] = branches[2i]
		end
	end

	@sync for (nodeno,node) in enumerate(nodes)
		@async begin
			# Top node for each subtree (a masternode)
			workersnode = tree.workersonhosts[node]
			nodetree = OrderedBinaryTree(workersnode)
			top = topnoderank(nodetree)
			topnode = nodetree[top]
			p = topnode.p

			if toptreenonleafnodes > 0
				# inherit from the parent node
				leafno = masternodeindex(tree,p)
				leafrank = leafrankfoldedtree(tree.toptree, Nmasternodes,leafno)
				parentrank = parentnoderank(toptree, leafrank)
				parentrankfulltree = toptree_to_fulltree_index(toptree, parentrank)
				parentnodebranches = branches[parentrankfulltree]
				parentchannels = parentnodebranches.childrenchannels
			else
				#= This happens if there is only one host, 
				in which case there's nothing to inherit.
				In this case there's no difference between a 
				SegmentedOrderedBinaryTree and an OrderedBinaryTree
				The top node is created separately as it is its own parent
				=#
				parentchannels = RemoteChannelContainer{Tred}(1,p)
			end

			topbranchnode = BranchChannel(p,Tmap,parentchannels,nchildren(topnode))
			nodetreestartindex = tree.nodetreestartindices[nodeno]
			branches[nodetreestartindex + top - 1] = topbranchnode

			# Populate the rest of the tree
			left_inds_nodetree = (1:top-1)
			left_inds_fulltree = (nodetreestartindex - 1) .+ left_inds_nodetree
			right_inds_nodetree = top+1:length(nodetree)
			right_inds_fulltree = (nodetreestartindex - 1) .+ right_inds_nodetree

			@async if !isempty(left_inds_nodetree)
				
				left_child = OrderedBinaryTree(
					@view nodetree.procs[left_inds_nodetree])
				
				createbranchchannels!(@view(branches[left_inds_fulltree]),
					Tmap,Tred,left_child,topbranchnode)
			end

			@async if !isempty(right_inds_nodetree)
				
				right_child = OrderedBinaryTree(
					@view nodetree.procs[right_inds_nodetree])

				createbranchchannels!(@view(branches[right_inds_fulltree]),
					Tmap,Tred,right_child,topbranchnode)
			end
		end
	end

	return branches
end

function createbranchchannels(Tmap,Tred,iterators::Tuple,T::Type{<:Tree})
	w = workersactive(iterators)
	tree = T(w)
	branches = createbranchchannels(Tmap,Tred,tree)
	tree,branches
end
function createbranchchannels(iterators::Tuple,T::Type{<:Tree})
	createbranchchannels(Any,Any,iterators,T)
end

function topbranch(tree::Tree,branches::Vector{<:BranchChannel})
	branches[topnoderank(tree)]
end