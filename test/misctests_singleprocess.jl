using DataStructures
using Test
using Aqua
using ParallelUtilities
using Documenter
using OffsetArrays
import ParallelUtilities: pval, value, BinaryTreeNode, BranchChannel,
ProductSplit, SegmentedOrderedBinaryTree
import ParallelUtilities.ClusterQueryUtils: chooseworkers

@testset "Project quality" begin
    Aqua.test_all(ParallelUtilities)
end

DocMeta.setdocmeta!(ParallelUtilities, :DocTestSetup, :(using ParallelUtilities); recursive=true)

@testset "doctest" begin
    doctest(ParallelUtilities, manual = false)
end

@testset "pval" begin
    p1 = pval{Float64}(1, false, 2.0)
    p2 = pval{Int}(1, false, 2)

    @test pval{Float64}(p1) === p1
    @test pval{Int}(p1) === p2

    @test value(p1) === 2.0
    @test value(p2) === 2
    @test value(2) === 2
    @test value(nothing) === nothing
end

@testset "chooseworkers" begin
    workers = 1:8
    workers_on_hosts = OrderedDict("host1" => 1:4, "host2" => 5:8)
    @test chooseworkers(workers, 3, workers_on_hosts) == 1:3
    @test chooseworkers(workers, 5, workers_on_hosts) == 1:5

    workers_on_hosts = OrderedDict(Libc.gethostname() => 1:4, "host2" => 5:8)
    @test chooseworkers(workers, 3, workers_on_hosts) == 1:3
    @test chooseworkers(workers, 5, workers_on_hosts) == 1:5

    workers_on_hosts = OrderedDict("host1" => 1:4, Libc.gethostname() => 5:8)
    @test chooseworkers(workers, 3, workers_on_hosts) == 5:7
    @test chooseworkers(workers, 5, workers_on_hosts) == [5:8; 1]
end

@testset "Reduction functions" begin
    # BroadcastStack with OffsetArrays
    @testset "BroadcastStack" begin
        arr = ParallelUtilities.BroadcastStack(+, 1)(ones(2:4), ones(3:5))
        @test arr == OffsetArray([1, 2, 2, 1], 2:5)

        arr = ParallelUtilities.BroadcastStack(+, 1:2)(ones(1:2, 2:4), ones(2:3, 3:5))
        arr_exp = OffsetArray([1.0  1.0  1.0  0.0
                                1.0  2.0  2.0  1.0
                                0.0  1.0  1.0  1.0], 1:3, 2:5)
        @test arr == arr_exp
    end

    @testset "BroadcastFunction" begin
        x = ones(3); y = ones(3);
        b = ParallelUtilities.BroadcastFunction{1}(+)
        @test b(x, y) == ones(3) * 2
        @test x == ones(3) * 2
        @test y == ones(3)

        b = ParallelUtilities.BroadcastFunction{2}(+)
        x = ones(3); y = ones(3);
        @test b(x, y) == ones(3) * 2
        @test x == ones(3)
        @test y == ones(3) * 2
    end

    @testset "Flip" begin
        x = ones(3); y = ones(3);
        f = ParallelUtilities.Flip(ParallelUtilities.elementwisesum!)
        @test f(x,y) == ones(3) * 2
        @test x == ones(3)
        @test y == ones(3) * 2

        x = ones(3); y = ones(3);
        f = ParallelUtilities.Flip(ParallelUtilities.broadcastinplace(+, Val(2)))
        @test f(x,y) == ones(3) * 2
        @test x == ones(3) * 2
        @test y == ones(3)
    end
end

@testset "show" begin

    @testset "ProductSplit" begin
        io = IOBuffer()
        ps = ProductSplit((1:20, 1:30), 4, 1)
        show(io, ps)
        showstr = String(take!(io))
        startstr = string(length(ps))*"-element ProductSplit"
        @test startswith(showstr, startstr)
    end

    @testset "error" begin
        io = IOBuffer()

        showerror(io, ParallelUtilities.TaskNotPresentError((1:4,), (5,)))
        strexp = "could not find the task $((5,)) in the list $((1:4,))"
        @test String(take!(io)) == strexp
    end;

    @testset "BranchChannel" begin
        io = IOBuffer()

        b = BranchChannel(1, 0)
        show(io, b)
        strexp = "Leaf  : 1 ← 1"
        @test String(take!(io)) == strexp

        b = BranchChannel(1, 1)
        show(io, b)
        strexp = "Branch: 1 ← 1 ← 1 child"
        @test String(take!(io)) == strexp

        b = BranchChannel(1, 2)
        show(io, b)
        strexp = "Branch: 1 ← 1 ⇇ 2 children"
        @test String(take!(io)) == strexp
    end;

    @testset "BinaryTreeNode" begin
        io = IOBuffer()
        b = BinaryTreeNode(2, 3, 1)
        show(io, b)
        strexp = "BinaryTreeNode(p = 2, parent = 3, nchildren = 1)"
        @test String(take!(io)) == strexp
    end;

    @testset "BinaryTree" begin
        # check that show is working
        io = IOBuffer()
        tree = SegmentedOrderedBinaryTree(1:8, OrderedDict("host1" => 1:4, "host2" => 5:8))
        show(io, tree)
        show(io, ParallelUtilities.toptree(tree))
        show(io, ParallelUtilities.toptree(tree).tree)
    end
end;
