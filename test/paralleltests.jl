using Distributed

@everywhere begin
    using DataStructures
    using Test
    using ParallelUtilities
    using ParallelUtilities.ClusterQueryUtils
    using OffsetArrays
    import ParallelUtilities: BinaryTreeNode, BranchChannel,
    OrderedBinaryTree, SegmentedOrderedBinaryTree,
    parentnoderank, nchildren,
    createbranchchannels,
    workersactive,
    leafrankfoldedtree,
    TopTreeNode, SubTreeNode,
    NoSplat, reducedvalue

    function parentnoderank(tree::SegmentedOrderedBinaryTree, i::Integer)
        1 <= i <= length(tree) || throw(BoundsError(tree, i))

        Nmasternodes = length(keys(ParallelUtilities.workersonhosts(tree)))
        toptreenonleafnodes = length(tree.toptree) - Nmasternodes

        if toptreenonleafnodes == 0
            pr = parentnoderank(ParallelUtilities.unsegmentedtree(tree),i)

        elseif i <= toptreenonleafnodes
            #= In a SegmentedSequentialBinaryTree the leading indices
            are the parent nodes of the top tree, so ind = i
            In a SegmentedOrderedBinaryTree, the leaves are removed
            from the top tree, so only even numbers are left.
            In this case, index i of the full tree refers to index 2i of the
            top tree, so ind = 2i
            =#
            ind = ParallelUtilities.fulltree_to_toptree_index(tree.toptree, i)
            p = tree.toptree[ind].p
            #  Compute the parent of the node with rank ind on the top tree.
            # In a SegmentedSequentialBinaryTree this is what we want.
            # In a SegmentedOrderedBinaryTree, we need to convert this back to
            # the index of the full tree, that is div(pr, 2)
            pr_top = parentnoderank(tree.toptree, ind)
            pr = ParallelUtilities.toptree_to_fulltree_index(tree.toptree, pr_top)
        else
            subtree, rankinsubtree, nptotalprevhosts = ParallelUtilities.subtree_rank(tree, i)

            if rankinsubtree == ParallelUtilities.topnoderank(subtree)
                # masternode
                # parent will be on the top - tree
                p = subtree[rankinsubtree].p
                leafno = ParallelUtilities.masternodeindex(tree, p)
                Nmasternodes = length(keys(ParallelUtilities.workersonhosts(tree)))
                leafrank = ParallelUtilities.leafrankfoldedtree(tree.toptree, Nmasternodes, leafno)
                pr_top = parentnoderank(tree.toptree, leafrank)
                # Convert back to the rank on the full tree where the
                # leaves of the top tree aren't stored.
                pr = ParallelUtilities.toptree_to_fulltree_index(tree.toptree, pr_top)
            else
                # node on a sub - tree
                pr = parentnoderank(subtree, rankinsubtree)
                pr += nptotalprevhosts + toptreenonleafnodes
            end
        end

        return pr
    end
end

macro testsetwithinfo(str, ex)
    quote
        @info "Testing "*$str
        @testset $str begin $(esc(ex)); end;
    end
end

fmap_local(x) = x^2
fred_local(x) = x
fred_local(x, y) = x + y

function showworkernumber(ind, nw)
    # Cursor starts off at the beginning of the line
    print("\u1b[K") # clear till end of line
    print("Testing on worker $ind of $nw")
    # return the cursor to the beginning of the line
    endchar = ind == nw ? "\n" : "\r"
    print(endchar)
end

@testsetwithinfo "utilities" begin
    @testset "hostnames" begin
        hosts = hostnames()
        nodes = unique(hosts)
        @test nodenames() == nodes
        @test nodenames(hosts) == nodes
        np1 = nprocs_node(hosts, nodes)
        np2 = nprocs_node(hosts)
        np3 = nprocs_node()
        @test np1 == np2 == np3
        for node in nodes
            npnode = count(isequal(node), hosts)
            @test np1[node] == npnode
        end
        p1 = procs_node(workers(), hosts, nodes)
        for node in nodes
            pnode = workers()[findall(isequal(node), hosts)]
            @test p1[node] == pnode
        end
        np4 = nprocs_node(p1)
        @test np1 == np4
    end
end;

@testset "BinaryTree" begin
    @testsetwithinfo "BinaryTreeNode" begin
        @testset "Constructor" begin
            p = workers()[1]
            b = BinaryTreeNode(p, p, 0)
            @test nchildren(b) == 0
            b = BinaryTreeNode(p, p, 1)
            @test nchildren(b) == 1
            b = BinaryTreeNode(p, p, 2)
            @test nchildren(b) == 2

            @test_throws DomainError BinaryTreeNode(p, p, 3)
            @test_throws DomainError BinaryTreeNode(p, p,-1)
        end
    end

    @testsetwithinfo "BinaryTree" begin
        @testsetwithinfo "OrderedBinaryTree" begin
            @testset "pid and parent" begin
                for imax = 1:100
                    procs = 1:imax
                    tree = OrderedBinaryTree(procs)
                    @test length(tree) == length(procs)

                    topnoderank = ParallelUtilities.topnoderank(tree)
                    @test tree[topnoderank].parent == topnoderank
                    for rank in 1:length(tree)
                        node = tree[rank]
                        @test node.p == procs[rank]
                        @test node.parent == procs[parentnoderank(tree, rank)]
                    end
                    @test_throws BoundsError(tree, 0) parentnoderank(tree, 0)
                    @test_throws BoundsError(tree, imax + 1) parentnoderank(tree, imax + 1)
                end
            end

            @testset "nchildren" begin
                tree = OrderedBinaryTree(1:1)
                @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                @test_throws BoundsError(tree, 2) nchildren(tree, 2)
                @test ParallelUtilities.topnoderank(tree) == 1

                tree = OrderedBinaryTree(1:2)
                @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 1
                @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                @test_throws BoundsError(tree, 3) nchildren(tree, 3)
                @test ParallelUtilities.topnoderank(tree) == 2

                tree = OrderedBinaryTree(1:8)
                @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 2
                @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 0
                @test nchildren(tree, 4) == nchildren(tree[4]) == tree[4].nchildren == 2
                @test nchildren(tree, 5) == nchildren(tree[5]) == tree[5].nchildren == 0
                @test nchildren(tree, 6) == nchildren(tree[6]) == tree[6].nchildren == 2
                @test nchildren(tree, 7) == nchildren(tree[7]) == tree[7].nchildren == 0
                @test nchildren(tree, 8) == nchildren(tree[8]) == tree[8].nchildren == 1
                @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                @test_throws BoundsError(tree, 9) nchildren(tree, 9)
                @test ParallelUtilities.topnoderank(tree) == 8

                tree = OrderedBinaryTree(1:11)
                @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 2
                @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 0
                @test nchildren(tree, 4) == nchildren(tree[4]) == tree[4].nchildren == 2
                @test nchildren(tree, 5) == nchildren(tree[5]) == tree[5].nchildren == 0
                @test nchildren(tree, 6) == nchildren(tree[6]) == tree[6].nchildren == 2
                @test nchildren(tree, 7) == nchildren(tree[7]) == tree[7].nchildren == 0
                @test nchildren(tree, 8) == nchildren(tree[8]) == tree[8].nchildren == 2
                @test nchildren(tree, 9) == nchildren(tree[9]) == tree[9].nchildren == 0
                @test nchildren(tree, 10) == nchildren(tree[10]) == tree[10].nchildren == 2
                @test nchildren(tree, 11) == nchildren(tree[11]) == tree[11].nchildren == 0
                @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                @test_throws BoundsError(tree, 12) nchildren(tree, 12)
                @test ParallelUtilities.topnoderank(tree) == 8

                tree = OrderedBinaryTree(1:13)
                @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 2
                @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 0
                @test nchildren(tree, 4) == nchildren(tree[4]) == tree[4].nchildren == 2
                @test nchildren(tree, 5) == nchildren(tree[5]) == tree[5].nchildren == 0
                @test nchildren(tree, 6) == nchildren(tree[6]) == tree[6].nchildren == 2
                @test nchildren(tree, 7) == nchildren(tree[7]) == tree[7].nchildren == 0
                @test nchildren(tree, 8) == nchildren(tree[8]) == tree[8].nchildren == 2
                @test nchildren(tree, 9) == nchildren(tree[9]) == tree[9].nchildren == 0
                @test nchildren(tree, 10) == nchildren(tree[10]) == tree[10].nchildren == 2
                @test nchildren(tree, 11) == nchildren(tree[11]) == tree[11].nchildren == 0
                @test nchildren(tree, 12) == nchildren(tree[12]) == tree[12].nchildren == 2
                @test nchildren(tree, 13) == nchildren(tree[13]) == tree[13].nchildren == 0
                @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                @test_throws BoundsError(tree, 14) nchildren(tree, 14)
                @test ParallelUtilities.topnoderank(tree) == 8
            end

            @testset "level" begin
                tree = OrderedBinaryTree(1:15)
                @test ParallelUtilities.levels(tree) == 4

                @test ParallelUtilities.levelfromtop.((tree,), 1:2:15) == ones(Int, 8).*4
                @test ParallelUtilities.levelfromtop.((tree,), (2, 6, 10, 14)) == (3, 3, 3, 3)
                @test ParallelUtilities.levelfromtop.((tree,), (4, 12)) == (2, 2)
                @test ParallelUtilities.levelfromtop(tree, 8) == 1
                for p in [0, length(tree) + 1]
                    @test_throws BoundsError(tree, p) ParallelUtilities.levelfromtop(tree, p)
                end

                tree = OrderedBinaryTree(1:13)
                @test ParallelUtilities.levels(tree) == 4
                @test ParallelUtilities.levelfromtop.((tree,), 1:2:11) == ones(Int, 6).*4
                @test ParallelUtilities.levelfromtop.((tree,), (2, 6, 10, 13)) == (3, 3, 3, 3)
                @test ParallelUtilities.levelfromtop.((tree,), (4, 12)) == (2, 2)
                @test ParallelUtilities.levelfromtop(tree, 8) == 1
                for p in [0, length(tree) + 1]
                    @test_throws BoundsError(tree, p) ParallelUtilities.levelfromtop(tree, p)
                end
            end
        end

        @testsetwithinfo "SegmentedOrderedBinaryTree" begin
            @testsetwithinfo "single host" begin
                @testset "pid and parent" begin
                    for imax = 1:100
                        procs = 1:imax
                        workersonhosts = Dict("host" => procs)
                        tree = SegmentedOrderedBinaryTree(procs, workersonhosts)
                        treeOBT = OrderedBinaryTree(procs)
                        @test length(tree) == length(procs) == length(treeOBT)

                        topnoderank = ParallelUtilities.topnoderank(tree)
                        # The top node is its own parent
                        @test tree[topnoderank].parent == topnoderank
                        @test tree[topnoderank] == ParallelUtilities.topnode(tree)
                        for rank in 1:length(tree)
                            node = tree[rank]
                            parentnode = tree[parentnoderank(tree, rank)]
                            @test length(procs) > 1 ? nchildren(parentnode) > 0 : nchildren(parentnode) == 0
                            @test node.p == procs[rank]
                            @test node.parent == procs[parentnoderank(treeOBT, rank)]
                            @test parentnode.p == node.parent
                        end
                    end
                end;

                @testset "nchildren" begin
                    procs = 1:1
                    tree = SegmentedOrderedBinaryTree(procs, Dict("host" => procs))
                    @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                    @test_throws BoundsError(tree, 2) nchildren(tree, 2)
                    @test ParallelUtilities.topnoderank(tree) == 1

                    procs = 1:2
                    tree = SegmentedOrderedBinaryTree(procs, Dict("host" => procs))
                    @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 1
                    @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                    @test_throws BoundsError(tree, 3) nchildren(tree, 3)
                    @test ParallelUtilities.topnoderank(tree) == 2

                    procs = 1:8
                    tree = SegmentedOrderedBinaryTree(procs, Dict("host" => procs))
                    @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test nchildren(tree, 4) == nchildren(tree[4]) == tree[4].nchildren == 2
                    @test nchildren(tree, 5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree, 6) == nchildren(tree[6]) == tree[6].nchildren == 2
                    @test nchildren(tree, 7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree, 8) == nchildren(tree[8]) == tree[8].nchildren == 1
                    @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                    @test_throws BoundsError(tree, 9) nchildren(tree, 9)
                    @test ParallelUtilities.topnoderank(tree) == 8

                    procs = 1:11
                    tree = SegmentedOrderedBinaryTree(procs, Dict("host" => procs))
                    @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test nchildren(tree, 4) == nchildren(tree[4]) == tree[4].nchildren == 2
                    @test nchildren(tree, 5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree, 6) == nchildren(tree[6]) == tree[6].nchildren == 2
                    @test nchildren(tree, 7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree, 8) == nchildren(tree[8]) == tree[8].nchildren == 2
                    @test nchildren(tree, 9) == nchildren(tree[9]) == tree[9].nchildren == 0
                    @test nchildren(tree, 10) == nchildren(tree[10]) == tree[10].nchildren == 2
                    @test nchildren(tree, 11) == nchildren(tree[11]) == tree[11].nchildren == 0
                    @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                    @test_throws BoundsError(tree, 12) nchildren(tree, 12)
                    @test ParallelUtilities.topnoderank(tree) == 8

                    procs = 1:13
                    tree = SegmentedOrderedBinaryTree(procs, Dict("host" => procs))
                    @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 0
                    @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test nchildren(tree, 4) == nchildren(tree[4]) == tree[4].nchildren == 2
                    @test nchildren(tree, 5) == nchildren(tree[5]) == tree[5].nchildren == 0
                    @test nchildren(tree, 6) == nchildren(tree[6]) == tree[6].nchildren == 2
                    @test nchildren(tree, 7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree, 8) == nchildren(tree[8]) == tree[8].nchildren == 2
                    @test nchildren(tree, 9) == nchildren(tree[9]) == tree[9].nchildren == 0
                    @test nchildren(tree, 10) == nchildren(tree[10]) == tree[10].nchildren == 2
                    @test nchildren(tree, 11) == nchildren(tree[11]) == tree[11].nchildren == 0
                    @test nchildren(tree, 12) == nchildren(tree[12]) == tree[12].nchildren == 2
                    @test nchildren(tree, 13) == nchildren(tree[13]) == tree[13].nchildren == 0
                    @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                    @test_throws BoundsError(tree, 14) nchildren(tree, 14)
                    @test ParallelUtilities.topnoderank(tree) == 8
                end;
            end;

            @testsetwithinfo "multiple hosts" begin
                @testset "length" begin
                    procs = 1:2
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict("host1" => 1:1,"host2" => 2:2))
                    @test length(tree) == 2 + 1

                    procs = 1:4
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict("host1" => 1:2,"host2" => 3:4))

                    @test length(tree) == 4 + 1

                    procs = 1:12
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict(
                            "host1" => 1:3,"host2" => 4:6,
                            "host3" => 7:9,"host4" => 10:12))

                    @test length(tree) == 12 + 3
                end;

                @testset "leafrankfoldedtree" begin
                    treeflag = OrderedBinaryTree(1:1)
                    @test leafrankfoldedtree(treeflag, 5, 1) == 1
                    @test leafrankfoldedtree(treeflag, 5, 2) == 3
                    @test leafrankfoldedtree(treeflag, 5, 3) == 5
                    @test leafrankfoldedtree(treeflag, 5, 4) == 7
                    @test leafrankfoldedtree(treeflag, 5, 5) == 9
                end;

                @testset "pid and parent" begin
                    for imax = 2:100
                        procs = 1:imax
                        mid = div(imax, 2)
                        workersonhosts = OrderedDict{String, Vector{Int}}()
                        workersonhosts["host1"] = procs[1:mid]
                        workersonhosts["host2"] = procs[mid + 1:end]
                        tree = SegmentedOrderedBinaryTree(procs, workersonhosts)

                        top = ParallelUtilities.topnoderank(tree)
                        @test tree[top] == ParallelUtilities.topnode(tree)
                        for (ind, rank) in enumerate(1:mid)
                            node = tree[rank + 1]
                            parentnode = tree[parentnoderank(tree, rank + 1)]
                            @test parentnode.p == node.parent
                            pnodes = workersonhosts["host1"]
                            @test node.p == pnodes[ind]
                            OBT = OrderedBinaryTree(pnodes)
                            if ind == ParallelUtilities.topnoderank(OBT)
                                # Special check for 2 hosts as
                                # there's only one node in the top tree
                                @test node.parent == ParallelUtilities.topnode(tree.toptree).p
                            else
                                @test node.parent == pnodes[parentnoderank(OBT, ind)]
                            end
                        end
                        for (ind, rank) in enumerate(mid + 1:imax)
                            node = tree[rank + 1]
                            parentnode = tree[parentnoderank(tree, rank + 1)]
                            @test parentnode.p == node.parent
                            pnodes = workersonhosts["host2"]
                            @test node.p == pnodes[ind]
                            OBT = OrderedBinaryTree(pnodes)
                            if ind == ParallelUtilities.topnoderank(OBT)
                                # Special check for 2 hosts as
                                # there's only one node in the top tree
                                @test node.parent == ParallelUtilities.topnode(tree.toptree).p
                            else
                                @test node.parent == pnodes[parentnoderank(OBT, ind)]
                            end
                        end
                    end
                end;

                @testset "nchildren" begin
                    procs = 1:2
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict("host1" => 1:1,"host2" => 2:2))
                    @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 2
                    @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 0
                    @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 0
                    @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                    @test_throws BoundsError(tree, 4) nchildren(tree, 4)

                    procs = 1:12
                    tree = SegmentedOrderedBinaryTree(procs,
                        OrderedDict(
                            "host1" => 1:3,"host2" => 4:6,
                            "host3" => 7:9,"host4" => 10:12))
                    @test nchildren(tree, 1) == nchildren(tree[1]) == tree[1].nchildren == 2
                    @test nchildren(tree, 2) == nchildren(tree[2]) == tree[2].nchildren == 2
                    @test nchildren(tree, 3) == nchildren(tree[3]) == tree[3].nchildren == 2
                    @test nchildren(tree, 4) == nchildren(tree[4]) == tree[4].nchildren == 0
                    @test nchildren(tree, 5) == nchildren(tree[5]) == tree[5].nchildren == 2
                    @test nchildren(tree, 6) == nchildren(tree[6]) == tree[6].nchildren == 0
                    @test nchildren(tree, 7) == nchildren(tree[7]) == tree[7].nchildren == 0
                    @test nchildren(tree, 8) == nchildren(tree[8]) == tree[8].nchildren == 2
                    @test nchildren(tree, 9) == nchildren(tree[9]) == tree[9].nchildren == 0
                    @test nchildren(tree, 10) == nchildren(tree[10]) == tree[10].nchildren == 0
                    @test nchildren(tree, 11) == nchildren(tree[11]) == tree[11].nchildren == 2
                    @test nchildren(tree, 12) == nchildren(tree[12]) == tree[12].nchildren == 0
                    @test nchildren(tree, 13) == nchildren(tree[13]) == tree[13].nchildren == 0
                    @test nchildren(tree, 14) == nchildren(tree[14]) == tree[14].nchildren == 2
                    @test nchildren(tree, 15) == nchildren(tree[15]) == tree[15].nchildren == 0
                    @test_throws BoundsError(tree, 0) nchildren(tree, 0)
                    @test_throws BoundsError(tree, 16) nchildren(tree, 16)
                end;
            end;
        end
    end
end;

@testsetwithinfo "reduction" begin

    @testset "BranchChannel" begin
        @test_throws DomainError BranchChannel(1, 3)
        parentchannel = RemoteChannel(() -> Channel(1))
        @test_throws DomainError BranchChannel(1, parentchannel, 3)
    end

    @testset "TopTreeNode" begin
        # Special test for this as this is usually not called when tests are carried out on the same machine
        parentchannel = RemoteChannel(() -> Channel(1))
        childrenchannel = RemoteChannel(() -> Channel(2))
        pipe = ParallelUtilities.BranchChannel(1, parentchannel, childrenchannel, 2)

        put!(childrenchannel, ParallelUtilities.pval(1, false, 1))
        put!(childrenchannel, ParallelUtilities.pval(2, false, 2))

        redval = reducedvalue(+, ParallelUtilities.TopTreeNode(1), pipe, nothing)
        @test redval === ParallelUtilities.pval(1, false, 3)

        put!(childrenchannel, ParallelUtilities.pval(1, false, 1))
        put!(childrenchannel, ParallelUtilities.pval(2, true, nothing))

        redval = reducedvalue(+, ParallelUtilities.TopTreeNode(1), pipe, nothing)
        @test redval === ParallelUtilities.pval(1, true, nothing)

        put!(childrenchannel, ParallelUtilities.pval(1, false, 1))
        put!(childrenchannel, ParallelUtilities.pval(2, false, 2))
        @test_throws Exception reducedvalue(x -> error(""), ParallelUtilities.TopTreeNode(1), pipe, nothing)
    end

    @testset "fake multiple hosts" begin
        tree = ParallelUtilities.SegmentedOrderedBinaryTree([1,1], OrderedDict("host1" => 1:1, "host2" => 1:1))
        branches = ParallelUtilities.createbranchchannels(tree)
        @test ParallelUtilities.pmapreduceworkers(x -> 1, +, (tree, branches), (1:4,)) == 4

        if nworkers() > 1
            p = procs_node()
            # Choose workers on the same node to avoid communication bottlenecks in testing
            w = first(values(p))
            tree = ParallelUtilities.SegmentedOrderedBinaryTree(w, OrderedDict("host1" => w[1]:w[1], "host2" => w[2]:w[end]))
            branches = ParallelUtilities.createbranchchannels(tree)
            @test ParallelUtilities.pmapreduceworkers(x -> 1, +, (tree, branches), (1:length(w),)) == length(w)
        end
    end
end

@testset "pmapreduce" begin
    @testsetwithinfo "pmapreduce" begin
        @testsetwithinfo "sum" begin
            @testsetwithinfo "comparison with mapreduce" begin
                for iterators in Any[(1:1,), (ones(2,2),), (1:10,)]
                    res_exp = mapreduce(x -> x^2, +, iterators...)
                    res = pmapreduce(x -> x^2, +, iterators...)
                    @test res_exp == res

                    res_exp = mapreduce(x -> x^2, +, iterators..., init = 100)
                    res = pmapreduce(x -> x^2, +, iterators..., init = 100)
                    @test res_exp == res
                end

                @testset "dictionary" begin
                    res = pmapreduce(x -> Dict(x => x), merge, 1:1)
                    res_exp = mapreduce(x -> Dict(x => x), merge, 1:1)
                    @test res == res_exp

                    res = pmapreduce(x -> Dict(x => x), merge, 1:200)
                    res_exp = mapreduce(x -> Dict(x => x), merge, 1:200)
                    @test res == res_exp

                    res = pmapreduce(x -> OrderedDict(x => x), merge, 1:20)
                    res_exp = mapreduce(x -> OrderedDict(x => x), merge, 1:20)
                    @test res == res_exp
                end

                iterators = (1:10, 2:2:20)
                res_exp = mapreduce((x, y) -> x*y, +, iterators...)
                res = pmapreduce((x, y) -> x*y, +, iterators...)
                @test res_exp == res

                res_exp = mapreduce((x, y) -> x*y, +, iterators..., init = 100)
                res = pmapreduce((x, y) -> x*y, +, iterators..., init = 100)
                @test res_exp == res

                iterators = (1:10, 2:2:20)
                iterators_product = Iterators.product(iterators...)
                res_exp = mapreduce(((x, y),) -> x*y, +, iterators_product)
                res = pmapreduce(((x, y),) -> x*y, +, iterators_product)
                @test res_exp == res

                res_exp_2itp = mapreduce(((x, y), (a, b)) -> x*a + y*b, +, iterators_product, iterators_product)
                res_2itp = pmapreduce(((x, y), (a, b)) -> x*a + y*b, +, iterators_product, iterators_product)
                @test res_2itp == res_exp_2itp

                iterators_product_putil = ParallelUtilities.product(iterators...)
                res_exp2 = mapreduce(((x, y),) -> x*y, +, iterators_product_putil)
                res2 = pmapreduce(((x, y),) -> x*y, +, iterators_product_putil)
                @test res_exp2 == res2
                @test res_exp2 == res_exp

                res_exp_2pup = mapreduce(((x, y), (a, b)) -> x*a + y*b, +, iterators_product_putil, iterators_product_putil)
                res_2pup = pmapreduce(((x, y), (a, b)) -> x*a + y*b, +, iterators_product_putil, iterators_product_putil)
                @test res_2pup == res_exp_2pup
                @test res_2pup == res_2itp
            end

            @testsetwithinfo "pmapreduce_productsplit" begin
                res_exp = sum(workers())
                @test pmapreduce_productsplit(x -> myid(), +, 1:nworkers()) == res_exp
                @test pmapreduce_productsplit(NoSplat(x -> myid()), +, 1:nworkers()) == res_exp
                @test pmapreduce_productsplit(x -> myid(), +, 1:nworkers(), 1:1) == res_exp
            end
        end;

        @testsetwithinfo "inplace assignment" begin
            res = pmapreduce_productsplit(x -> ones(2), ParallelUtilities.elementwisesum!, 1:10)
            resexp = mapreduce(x -> ones(2), +, 1:min(10, nworkers()))
            @test res == resexp

            res = pmapreduce_productsplit(x -> ones(2), ParallelUtilities.elementwiseproduct!, 1:4)
            resexp = mapreduce(x -> ones(2), (x,y) -> x .* y, 1:min(4, nworkers()))
            @test res == resexp

            res = pmapreduce_productsplit(x -> ones(2), ParallelUtilities.elementwisemin!, 1:4)
            resexp = mapreduce(x -> ones(2), (x,y) -> min.(x,y), 1:min(4, nworkers()))
            @test res == resexp

            res = pmapreduce_productsplit(x -> ones(2), ParallelUtilities.elementwisemax!, 1:4)
            resexp = mapreduce(x -> ones(2), (x,y) -> max.(x,y), 1:min(4, nworkers()))
            @test res == resexp
        end

        @testsetwithinfo "concatenation" begin
            @testsetwithinfo "comparison with mapreduce" begin
                resexp_vcat = mapreduce(identity, vcat, 1:nworkers())
                resexp_hcat = mapreduce(identity, hcat, 1:nworkers())
                res_vcat = pmapreduce(identity, vcat, 1:nworkers())
                res_hcat = pmapreduce(identity, hcat, 1:nworkers())
                @test res_vcat == resexp_vcat
                @test res_hcat == resexp_hcat
            end

            @testsetwithinfo "pmapreduce_productsplit" begin
                res_vcat = mapreduce(identity, vcat, ones(2) for i in 1:nworkers())
                res_hcat = mapreduce(identity, hcat, ones(2) for i in 1:nworkers())

                @test pmapreduce_productsplit(x -> ones(2), vcat, 1:nworkers()) == res_vcat
                @test pmapreduce_productsplit(x -> ones(2), hcat, 1:nworkers()) == res_hcat
            end
        end;

        @testsetwithinfo "run elsewhere" begin
            @testsetwithinfo "sum" begin
                res_exp = sum(workers())
                c = Channel(nworkers())
                tasks = Vector{Task}(undef, nworkers())
                @sync begin
                    for (ind, p) in enumerate(workers())
                        tasks[ind] = @async begin
                            try
                                res = @fetchfrom p pmapreduce_productsplit(x -> myid(), +, 1:nworkers())
                                put!(c,(ind, res, false))
                            catch
                                put!(c,(ind, 0, true))
                                rethrow()
                            end
                        end
                    end
                    for i = 1:nworkers()
                        ind, res, err = take!(c)
                        err && wait(tasks[ind])
                        @test res == res_exp
                        showworkernumber(i, nworkers())
                    end
                end
            end
            # concatenation where the rank is used in the mapping function
            # Preserves order of the iterators
            @testsetwithinfo "concatenation using rank" begin
                c = Channel(nworkers())
                tasks = Vector{Task}(undef, nworkers())
                @sync begin
                    for (ind, p) in enumerate(workers())
                        tasks[ind] = @async begin
                            try
                                res = @fetchfrom p (pmapreduce_productsplit(x -> x[1][1], vcat, 1:nworkers()) == mapreduce(identity, vcat, 1:nworkers()))
                                put!(c,(ind, res, false))
                            catch
                                put!(c,(ind, false, true))
                                rethrow()
                            end
                        end
                    end
                    for i = 1:nworkers()
                        ind, res, err = take!(c)
                        err && wait(tasks[ind])
                        @test res
                        showworkernumber(i, nworkers())
                    end
                end
            end
        end;

        @testsetwithinfo "errors" begin
            @test_throws Exception pmapreduce(x -> error("map"), +, 1:10)
            @test_throws Exception pmapreduce(identity, x -> error("reduce"), 1:10)
            @test_throws Exception pmapreduce(x -> error("map"), x -> error("reduce"), 1:10)

            @test_throws Exception pmapreduce(fmap, +, 1:10)
            @test_throws Exception pmapreduce(identity, fred, 1:10)
            @test_throws Exception pmapreduce(fmap, fred, 1:10)

            if nworkers() != nprocs()
                @test_throws Exception pmapreduce(fmap_local, +, 1:10)
                @test_throws Exception pmapreduce(identity, fred_local, 1:10)
                @test_throws Exception pmapreduce(fmap_local, fred, 1:10)
                @test_throws Exception pmapreduce(fmap_local, fred_local, 1:10)
            end
        end;
    end;
    @testsetwithinfo "pmapbatch" begin
        for (iterators, fmap) in Any[
            ((1:1,), x -> 1),
            ((1:10,), x -> 1),
            ((1:5,), x -> ones(1) * x),
            ((1:10, 1:10), (x,y) -> ones(3) * (x+y))]

            res = pmapbatch(fmap, iterators...)
            res_exp = pmap(fmap, iterators...)
            @test res == res_exp
        end

        v = pmapbatch_productsplit(x -> sum(sum(i) for i in x) * ones(2), 1:1, 1:1)
        @test v == [[2.0, 2.0]]
        v = pmapbatch_productsplit(x -> ParallelUtilities.workerrank(x), 1:nworkers(), 1:nworkers())
        @test v == [1:nworkers();]
    end
end;
