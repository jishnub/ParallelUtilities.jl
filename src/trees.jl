abstract type Tree end
abstract type BinaryTree <: Tree end

struct OrderedBinaryTree{T <: AbstractVector{<:Integer}, P <: Union{Nothing, Integer}} <: BinaryTree
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
	topnode_parent :: P

	function OrderedBinaryTree(procs::AbstractVector{<:Integer}, p = nothing)
		N = length(procs)
		N >= 1 || throw(DomainError(N, "need at least one node to create a BinaryTree"))

		new{typeof(procs), typeof(p)}(N, procs, p)
	end
end

# Special type for the top tree that correctly returns nchildren for the leaves
struct ConnectedOrderedBinaryTree{T <: AbstractVector{<:Integer}, D <: AbstractDict} <: BinaryTree
	N :: Int
	tree :: OrderedBinaryTree{T, Nothing}
	workersonhosts :: D

	function ConnectedOrderedBinaryTree(tree::OrderedBinaryTree{T, Nothing}, workersonhosts::D) where {T, D <: AbstractDict}
		N = length(tree)
		new{T, D}(N, tree, workersonhosts)
	end
end

abstract type SegmentedBinaryTree <: BinaryTree end

struct SegmentedOrderedBinaryTree{T<:AbstractVector{<:Integer}, D <: AbstractDict} <: SegmentedBinaryTree
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
	toptree :: ConnectedOrderedBinaryTree{Vector{Int}, D}
	nodetreestartindices :: Vector{Int}

	function SegmentedOrderedBinaryTree(N::Int, procs::T, workersonhosts::D,
		toptree::ConnectedOrderedBinaryTree{Vector{Int}, D}, nodetreestartindices:: Vector{Int}) where {T, D}

		mapreduce(length, +, values(workersonhosts)) == length(procs) || throw(ArgumentError("procs do not match workersonhosts"))
		# check that the reduction nodes of the top tree have children
		all(i -> nchildren(toptree[i]) == 2, 2:2:length(toptree)) || throw(ArgumentError("reduction nodes on the top tree must have 2 children each"))

		new{T, D}(N, procs, workersonhosts, toptree, nodetreestartindices)
	end
end

function leafrankfoldedtree(::OrderedBinaryTree, Nleaves, leafno)
	@assert(leafno <= Nleaves, "leafno needs to be ⩽ Nleaves")
	leafrank = 2leafno - 1
end
leafrankfoldedtree(tree::ConnectedOrderedBinaryTree, args...) = leafrankfoldedtree(tree.tree, args...)

function foldedbinarytreefromleaves(::Type{OrderedBinaryTree}, leaves)
	Nleaves = length(leaves)
	Nnodes = 2Nleaves - 1

	allnodes = Vector{Int}(undef, Nnodes)
	foldedbinarytreefromleaves!(OrderedBinaryTree, allnodes, leaves)

	OrderedBinaryTree(allnodes)
end

function foldedbinarytreefromleaves!(::Type{OrderedBinaryTree}, allnodes, leaves)
	top = topnoderank(OrderedBinaryTree(1:length(allnodes)))
	allnodes[top] = first(leaves)

	length(allnodes) == 1 && return

	Nnodes_left = top - 1
	Nleaves_left = div( Nnodes_left + 1 , 2)
	Nleaves_right = length(leaves) - Nleaves_left

	if Nleaves_left > 0
		leaves_left = @view leaves[1:Nleaves_left]
		leftnodes = @view allnodes[1:Nnodes_left]
		foldedbinarytreefromleaves!(OrderedBinaryTree, leftnodes, leaves_left)
	end

	if Nleaves_right > 0
		leaves_right = @view leaves[end - Nleaves_right + 1:end]
		rightnodes = @view allnodes[top + 1:end]
		foldedbinarytreefromleaves!(OrderedBinaryTree, rightnodes, leaves_right)
	end
end

function SegmentedOrderedBinaryTree(procs::AbstractVector{<:Integer}, workersonhosts::AbstractDict = procs_node(procs))

	Np = length(procs)
	Np >= 1 || throw(DomainError(Np, "need at least one node to create a BinaryTree"))

	nodes = collect(keys(workersonhosts))
	masternodes = Vector{Int}(undef, length(nodes))
	for (nodeind, node) in enumerate(nodes)
		workersnode = workersonhosts[node]
		nodetree = OrderedBinaryTree(workersnode)
		masternodes[nodeind] = topnode(nodetree).p
	end
	Nleaves = length(masternodes)
	toptree_inner = foldedbinarytreefromleaves(OrderedBinaryTree, masternodes)
	toptree = ConnectedOrderedBinaryTree(toptree_inner, workersonhosts)

	toptreenonleafnodes = length(toptree) - Nleaves
	Nnodestotal = toptreenonleafnodes + length(procs)

	nodetreestartindices = Vector{Int}(undef, length(nodes))
	nodetreestartindices[1] = toptreenonleafnodes + 1
	for (nodeno, node) in enumerate(nodes)
		nodeno == 1 && continue
		prevnode = nodes[nodeno - 1]
		nodetreestartindices[nodeno] = nodetreestartindices[nodeno - 1] +
										length(workersonhosts[prevnode])
	end

	SegmentedOrderedBinaryTree(Nnodestotal, procs, workersonhosts,
		toptree, nodetreestartindices)
end



# for a single host there are no segments
function unsegmentedtree(::Type{<:SegmentedOrderedBinaryTree})
	OrderedBinaryTree
end
function unsegmentedtree(tree::SegmentedBinaryTree)
	T = unsegmentedtree(typeof(tree))
	T(workers(tree))
end

Base.length(tree::BinaryTree) = tree.N
levels(tree::OrderedBinaryTree) = levels(length(tree))
levels(n::Integer) = floor(Int, log2(n)) + 1

function Base.summary(io::IO, tree::SegmentedOrderedBinaryTree)
	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(toptree(tree)) - Nmasternodes
	mapnodes = length(tree) - toptreenonleafnodes
	print(io, length(tree), "-node ", Base.nameof(typeof(tree)))
	print(io, " with ", mapnodes, " workers and ", toptreenonleafnodes, " extra reduction nodes")
end
Base.summary(io::IO, tree::Tree) = print(io, length(tree),"-element ", nameof(typeof(tree)))

function Base.show(io::IO, b::OrderedBinaryTree)
	print(io, summary(b), "(", workers(b), ") with top node = ", topnode(b))
end
function Base.show(io::IO, b::ConnectedOrderedBinaryTree)
	print(io, summary(b), "(", workers(b), ", ", b.workersonhosts, ")")
end

function Base.show(io::IO, b::SegmentedOrderedBinaryTree)
	summary(io, b)
	println(io)
	println(io, "toptree => ", toptree(b))
	println(io, "subtrees start from indices ", b.nodetreestartindices)
	tt = toptree(b)
	for (ind, (host, w)) in enumerate(b.workersonhosts)
		node = tt[2ind - 1]
		print(io, host, " => ",  OrderedBinaryTree(w, node.parent))
		if ind != length(b.workersonhosts)
			println(io)
		end
	end
end

toptree(tree::SegmentedOrderedBinaryTree) = tree.toptree

function levelfromtop(tree::OrderedBinaryTree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))

	top = topnoderank(tree)
	if i == top
		return 1
	elseif i < top
		subrange = 1:top - 1
	else
		subrange = top + 1:length(tree)
	end
	subtree = OrderedBinaryTree(subrange)
	subindex = searchsortedfirst(subrange, i)
	1 + levelfromtop(subtree, subindex)
end

function parentnoderank(tree::OrderedBinaryTree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))

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

		# sgn is +1 if mod(ired, 4) = 1, -1 if mod(ired, 4) = 3
		sgn = 2 - mod(ired, 4)
		return i + sgn * pow2level
	elseif i > top
		# right branch, possibly partially formed
		# Carry out a recursive search
		subtreeprocs = top + 1:length(tree)
		subtree = OrderedBinaryTree(subtreeprocs)
		subind = searchsortedfirst(subtreeprocs, i)
		if subind == topnoderank(subtree)
			# This catches the case of there only being a leaf node
			# in the sub - tree
			return top
		elseif length(subtreeprocs) == 3
			# don't subdivide to 1 - node trees
			# this lets us avoid confusing this with the case of
			# the entire tree having only 1 node
			return subtreeprocs[2]
		end
		pid = parentnoderank(subtree, subind)
		return subtreeprocs[pid]
	end
end
parentnoderank(tree::ConnectedOrderedBinaryTree, i::Integer) = parentnoderank(tree.tree, i)

function subtree_rank(tree::SegmentedBinaryTree, i::Integer)
	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	# node on a subtree at a host
	subnodeno = i - toptreenonleafnodes

	@assert(subnodeno > 0, "i needs to be greater than $(toptreenonleafnodes)")

	# find out which node this lies on
	nptotalprevhosts = 0
	for (host, procs) in tree.workersonhosts
		np = length(procs)
		if subnodeno <= nptotalprevhosts + np
			rankinsubtree = subnodeno - nptotalprevhosts
			T = unsegmentedtree(typeof(tree))
			subtree = T(tree.workersonhosts[host])
			return subtree, rankinsubtree, nptotalprevhosts
		end
		nptotalprevhosts += np
	end
end

"""
	masternodeindex(tree::SegmentedOrderedBinaryTree, p)

Given the top worker `p` on one node, compute the serial order of the host that it corresponds to.
"""
function masternodeindex(tree::SegmentedOrderedBinaryTree, p)
	leafno = nothing
	T = unsegmentedtree(typeof(tree))
	for (ind, w) in enumerate(values(tree.workersonhosts))
		subtree = OrderedBinaryTree(w)
		top = topnoderank(subtree)
		if w[top] == p
			leafno = ind
			break
		end
	end
	return leafno
end

toptree_to_fulltree_index(::OrderedBinaryTree, i) = div(i, 2)
toptree_to_fulltree_index(tree::ConnectedOrderedBinaryTree, i) = toptree_to_fulltree_index(tree.tree, i)

fulltree_to_toptree_index(::OrderedBinaryTree, i) = 2i
fulltree_to_toptree_index(tree::ConnectedOrderedBinaryTree, i) = fulltree_to_toptree_index(tree.tree, i)

"""
	parentnoderank(tree::SegmentedBinaryTree, i::Integer)

Given the rank `i` of a node on the tree, return the rank of its parent.
"""
function parentnoderank(tree::SegmentedBinaryTree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))

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
		ind = fulltree_to_toptree_index(tree.toptree, i)
		p = tree.toptree[ind].p
		#= Compute the parent of the node with rank ind on the top tree.
		In a SegmentedSequentialBinaryTree this is what we want.
		In a SegmentedOrderedBinaryTree, we need to convert this back to
		the index of the full tree, that is div(pr, 2)
		=#
		pr_top = parentnoderank(tree.toptree, ind)
		pr = toptree_to_fulltree_index(tree.toptree, pr_top)

	else
		subtree, rankinsubtree, nptotalprevhosts = subtree_rank(tree, i)

		if rankinsubtree == topnoderank(subtree)
			# masternode
			# parent will be on the top - tree
			p = subtree[rankinsubtree].p
			leafno = masternodeindex(tree, p)
			Nmasternodes = length(keys(tree.workersonhosts))
			leafrank = leafrankfoldedtree(tree.toptree, Nmasternodes, leafno)
			pr_top = parentnoderank(tree.toptree, leafrank)
			# Convert back to the rank on the full tree where the
			# leaves of the top tree aren't stored.
			pr = toptree_to_fulltree_index(tree.toptree, pr_top)
		else
			# node on a sub - tree
			pr = parentnoderank(subtree, rankinsubtree)
			pr += nptotalprevhosts + toptreenonleafnodes
		end
	end

	return pr
end

function nchildren(tree::OrderedBinaryTree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))

	if isodd(i)
		0
	elseif i == length(tree)
		1
	else
		2
	end
end
function nchildren(tree::SegmentedBinaryTree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(tree.toptree) - Nmasternodes

	if toptreenonleafnodes == 0
		n = nchildren(unsegmentedtree(tree),i)

	elseif i <= toptreenonleafnodes
		# The top - tree is a full binary tree.
		# Since the leaves aren't stored, every parent node
		# has 2 children
		n = 2
	else
		subtree, rankinsubtree = subtree_rank(tree, i)
		n = nchildren(subtree, rankinsubtree)
	end

	return n
end
function nchildren(tree::ConnectedOrderedBinaryTree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))
	if isodd(i)
		host = ""
		for (ind, h) in enumerate(keys(tree.workersonhosts))
			if ind == i ÷ 2 + 1
				host = h
			end
		end
		st = OrderedBinaryTree(tree.workersonhosts[host])
		nchildren(topnode(st))
	else
		2
	end
end

topnoderank(tree::ConnectedOrderedBinaryTree) = topnoderank(tree.tree)
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
		tnr = topnoderank(OrderedBinaryTree(workers(tree)))
	end
	return tnr
end

topnode(tree::Tree) = tree[topnoderank(tree)]
function topnode(tree::OrderedBinaryTree{<:Any, <:Integer})
	node = tree[topnoderank(tree)]
	BinaryTreeNode(node.p, tree.topnode_parent, node.nchildren)
end

# Indexing into a OrderedBinaryTree produces a BinaryTreeNode
struct BinaryTreeNode
	p :: Int
	parent :: Int
	nchildren :: Int

	function BinaryTreeNode(p::Int, p_parent::Int, nchildren::Int)
		(0 <= nchildren <= 2) ||
		throw(DomainError(nchildren,
			"attempt to construct a binary tree with $nchildren children"))

		new(p, p_parent, nchildren)
	end
end

function Base.show(io::IO, b::BinaryTreeNode)
	print(io,
		"BinaryTreeNode(p = $(b.p),"*
		" parent = $(b.parent), nchildren = $(b.nchildren))")
end

nchildren(b::BinaryTreeNode) = b.nchildren

Distributed.workers(tree::OrderedBinaryTree) = tree.procs
Distributed.workers(tree::ConnectedOrderedBinaryTree) = workers(tree.tree)
Distributed.workers(tree::SegmentedOrderedBinaryTree) = tree.procs

Distributed.nworkers(tree::Tree) = length(workers(tree))

function Base.getindex(tree::Tree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))

	procs = workers(tree)

	p = procs[i]
	pr = parentnoderank(tree, i)
	p_parent = procs[pr]
	n = nchildren(tree, i)

	BinaryTreeNode(p, p_parent, n)
end

function Base.getindex(tree::SegmentedBinaryTree, i::Integer)
	1 <= i <= length(tree) || throw(BoundsError(tree, i))

	Nmasternodes = length(keys(tree.workersonhosts))
	toptreenonleafnodes = length(toptree(tree)) - Nmasternodes

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
		ind = fulltree_to_toptree_index(tree.toptree, i)
		p = tree.toptree[ind].p
		pr_top = parentnoderank(tree.toptree, ind)
		p_parent = tree.toptree[pr_top].p
		n = 2
		return BinaryTreeNode(p, p_parent, n)
	else
		subtree, rankinsubtree = subtree_rank(tree, i)

		p = subtree[rankinsubtree].p
		n = nchildren(subtree, rankinsubtree)

		if rankinsubtree == topnoderank(subtree)
			# masternode
			# parent will be on the top tree
			Nmasternodes = length(keys(tree.workersonhosts))
			leafno = masternodeindex(tree, p)
			leafrank = leafrankfoldedtree(tree.toptree, Nmasternodes, leafno)
			pr_top = parentnoderank(tree.toptree, leafrank)
			p_parent = tree.toptree[pr_top].p
		else
			# node on a sub - tree
			pr = parentnoderank(subtree, rankinsubtree)
			p_parent = subtree[pr].p
		end
		return BinaryTreeNode(p, p_parent, n)
	end
end

# Branches between nodes

struct BranchChannel{R <: RemoteChannel}
	p :: Int
	parentchannels :: R
	childrenchannels :: R
	nchildren :: Int

	function BranchChannel(p::Int,
		parentchannels::R,
		childrenchannels::R,
		nchildren::Int) where {R <: RemoteChannel}

		(0 <= nchildren <= 2) ||
		throw(DomainError(nchildren,
			"attempt to construct a binary tree with $nchildren children"))

		new{R}(p, parentchannels, childrenchannels, nchildren)
	end
end
nchildren(b::BranchChannel) = b.nchildren

childrenerror(nchildren) = throw(DomainError(nchildren,
		"attempt to construct a binary tree with $nchildren children"))

function BranchChannel(p::Integer, parentchannels::R, nchildren::Integer) where {R <: RemoteChannel}

	(0 <= nchildren <= 2) || childrenerror(nchildren)

	childrenchannels = RemoteChannel(() -> Channel(nchildren), p)
	BranchChannel(p, parentchannels, childrenchannels, nchildren)
end

function BranchChannel(p::Integer, nchildren::Integer)

	(0 <= nchildren <= 2) || childrenerror(nchildren)

	parentchannels, childrenchannels = @sync begin
		parenttask = @async RemoteChannel(() -> Channel(1), p)
		childtask = @async RemoteChannel(() -> Channel(nchildren), p)
		asyncmap(fetch, (parenttask, childtask))
	end
	BranchChannel(p, parentchannels, childrenchannels, nchildren)
end

function Base.show(io::IO, b::BranchChannel)
	N = nchildren(b)
	p_parent = b.parentchannels.where
	p = b.p

	if N == 2
		str = "Branch: "*string(p_parent)*" ← "*string(p)*" ⇇ 2 children"
	elseif N == 1
		str = "Branch: "*string(p_parent)*" ← "*string(p)*" ← 1 child"
	else
		str = "Leaf  : "*string(p_parent)*" ← "*string(p)
	end

	print(io, str)
end

function finalize_except_wherewhence(r::RemoteChannel)
	if (myid() != r.where) && (myid() != r.whence)
		finalize(r)
	end
end

function Base.finalize(bc::BranchChannel)
	finalize(bc.childrenchannels)
	finalize_except_wherewhence(bc.parentchannels)
end

function createbranchchannels!(branches, tree::OrderedBinaryTree, superbranch::BranchChannel)
	top = topnoderank(tree)
	topnode = tree[top]

	topbranchchannels = BranchChannel(topnode.p, superbranch.childrenchannels, nchildren(topnode))
	branches[top] = topbranchchannels

	length(tree) == 1 && return nothing

	left_inds = 1:top - 1
	right_inds = top + 1:length(tree)

	@sync begin
		@async if !isempty(left_inds)
			left_child = OrderedBinaryTree(@view workers(tree)[left_inds])
			createbranchchannels!(@view(branches[left_inds]), left_child, topbranchchannels)
		end
		@async if !isempty(right_inds)
			right_child = OrderedBinaryTree(@view workers(tree)[right_inds])
			createbranchchannels!(@view(branches[right_inds]), right_child, topbranchchannels)
		end
	end
	return nothing
end

function createbranchchannels(tree::SegmentedOrderedBinaryTree)

	nodes = keys(tree.workersonhosts)
	toptree = tree.toptree
	Nmasternodes = length(nodes)
	toptreenonleafnodes = length(toptree) - Nmasternodes

	branches = Vector{BranchChannel{RemoteChannel{Channel{Any}}}}(undef, length(tree))

	# populate the top tree other than the masternodes
	# This is only run if there are multiple hosts
	if toptreenonleafnodes > 0
		topnoderank_toptree = topnoderank(toptree)
		topnode_toptree = toptree[topnoderank_toptree]
		N = nchildren(topnode_toptree)
		topmostbranch = BranchChannel(topnode_toptree.p, N)
		branches[topnoderank_toptree] = topmostbranch

		left_inds = 1:(topnoderank_toptree - 1)
		right_inds = (topnoderank_toptree + 1):length(toptree)

		@sync begin
			@async if !isempty(left_inds)
				left_child = OrderedBinaryTree(@view workers(toptree)[left_inds])
				createbranchchannels!(@view(branches[left_inds]), left_child, topmostbranch)
			end

			@async if !isempty(right_inds)
				right_child = OrderedBinaryTree(@view workers(toptree)[right_inds])
				createbranchchannels!(@view(branches[right_inds]), right_child, topmostbranch)
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

	@sync for (nodeno, node) in enumerate(nodes)
		@async begin
			# Top node for each subtree (a masternode)
			workersnode = tree.workersonhosts[node]
			nodetree = OrderedBinaryTree(workersnode)
			top = topnoderank(nodetree)
			topnode = nodetree[top]
			p = topnode.p

			if toptreenonleafnodes > 0
				# inherit from the parent node
				leafno = masternodeindex(tree, p)
				leafrank = leafrankfoldedtree(tree.toptree, Nmasternodes, leafno)
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
				parentchannels = RemoteChannel(() -> Channel(1), p)
			end

			topbranchnode = BranchChannel(p, parentchannels, nchildren(topnode))
			nodetreestartindex = tree.nodetreestartindices[nodeno]
			branches[nodetreestartindex + top - 1] = topbranchnode

			# Populate the rest of the tree
			left_inds_nodetree = (1:top - 1)
			left_inds_fulltree = (nodetreestartindex - 1) .+ left_inds_nodetree
			right_inds_nodetree = top + 1:length(nodetree)
			right_inds_fulltree = (nodetreestartindex - 1) .+ right_inds_nodetree

			@async if !isempty(left_inds_nodetree)
				left_child = OrderedBinaryTree(@view workers(nodetree)[left_inds_nodetree])
				createbranchchannels!(@view(branches[left_inds_fulltree]), left_child, topbranchnode)
			end

			@async if !isempty(right_inds_nodetree)
				right_child = OrderedBinaryTree(@view workers(nodetree)[right_inds_nodetree])
				createbranchchannels!(@view(branches[right_inds_fulltree]), right_child, topbranchnode)
			end
		end
	end

	return branches
end

function createbranchchannels(pool::AbstractWorkerPool, len::Integer)
	w = workersactive(pool, len)
	tree = SegmentedOrderedBinaryTree(w)
	branches = createbranchchannels(tree)
	tree, branches
end

topbranch(tree::Tree, branches::AbstractVector{<:BranchChannel}) = branches[topnoderank(tree)]

function workersactive(pool::AbstractWorkerPool, len::Integer,
	workers_on_hosts::AbstractDict = procs_node(workers(pool)))

	nw = min(nworkers(pool), len)
	chooseworkers(workers(pool), len, workers_on_hosts)
end

function chooseworkers(workerspool, n::Integer, workers_on_hosts::AbstractDict = procs_node(workerspool))
	n >= 1 || throw(ArgumentError("number of workers to choose must be >= 1"))
	length(workerspool) <= n && return workerspool
	myhost = Libc.gethostname()
	if myhost in keys(workers_on_hosts)
		if length(workers_on_hosts[myhost]) >= n
			return workers_on_hosts[myhost][1:n]
		else
			w_chosen = workers_on_hosts[myhost]
			np_left = n - length(w_chosen)
			for (host, workers_host) in workers_on_hosts
				np_left <= 0 && break
				host == myhost && continue
				workers_host_section = @view workers_host[1:min(length(workers_host), np_left)]
				w_chosen = vcat(w_chosen, workers_host_section)
				np_left -= length(workers_host_section)
			end
			return w_chosen
		end
	else
		return workerspool[1:n]
	end
end

function maybetrimmedworkerpool(workers, N)
	w = chooseworkers(workers, N)
	WorkerPool(w)
end
