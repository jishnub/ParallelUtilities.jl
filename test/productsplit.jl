using Distributed
using Test
using ParallelUtilities
import ParallelUtilities: ProductSplit, ProductSection,
minimumelement, maximumelement, extremaelement, nelements, dropleading, indexinproduct,
extremadims, localindex, extrema_commonlastdim, whichproc, procrange_recast, whichproc_localindex,
getiterators, _niterators

macro testsetwithinfo(str, ex)
    quote
        @info "Testing "*$str
        @testset $str begin $(esc(ex)); end;
    end
end

@testsetwithinfo "AbstractConstrainedProduct" begin

    various_iters = Any[(1:10,), (1:1:10,), (1:10, 4:6), (1:1:10, 4:6), (1:10, 4:6, 1:4), (1:2:9,), (1:2:9, 4:1:6),
                    (1:2, Base.OneTo(4), 1:3:10), (1:0.5:3, 2:4)]

    @testsetwithinfo "ProductSplit" begin

        function split_across_processors_iterators(arr::Iterators.ProductIterator, num_procs, proc_id)

            num_tasks = length(arr);

            num_tasks_per_process, num_tasks_leftover = divrem(num_tasks, num_procs)

            num_tasks_on_proc = num_tasks_per_process + (proc_id <= mod(num_tasks, num_procs) ? 1 : 0 );
            task_start = num_tasks_per_process*(proc_id-1) + min(num_tasks_leftover, proc_id-1) + 1;

            Iterators.take(Iterators.drop(arr, task_start-1), num_tasks_on_proc)
        end

        function split_product_across_processors_iterators(arrs_tuple, num_procs, proc_id)
            split_across_processors_iterators(Iterators.product(arrs_tuple...), num_procs, proc_id)
        end

        @testset "Constructor" begin

            function checkPSconstructor(iters, npmax = 10)
                ntasks_total = prod(length, iters)
                for np = 1:npmax, p = 1:np
                    ps = ProductSplit(iters, np, p)
                    @test eltype(ps) == Tuple{map(eltype, iters)...}
                    @test _niterators(ps) == length(iters)
                    if !isempty(ps)
                        @test collect(ps) == collect(split_product_across_processors_iterators(iters, np, p))
                    end
                    @test prod(length, getiterators(ps)) == ntasks_total
                    @test ParallelUtilities.workerrank(ps) == p
                    @test nworkers(ps) == np
                end

                @test_throws ArgumentError ProductSplit(iters, npmax, npmax + 1)
            end

            @testset "0D" begin
                @test_throws ArgumentError ProductSplit((), 2, 1)
            end

            @testset "cumprod" begin
                @test ParallelUtilities._cumprod(1,()) == ()
                @test ParallelUtilities._cumprod(1,(2,)) == (1,)
                @test ParallelUtilities._cumprod(1,(2, 3)) == (1, 2)
                @test ParallelUtilities._cumprod(1,(2, 3, 4)) == (1, 2, 6)
            end

            @testset "1D" begin
                iters = (1:10,)
                checkPSconstructor(iters)
            end
            @testset "2D" begin
                iters = (1:10, 4:6)
                checkPSconstructor(iters)
            end
            @testset "3D" begin
                iters = (1:10, 4:6, 1:4)
                checkPSconstructor(iters)
            end
            @testset "steps" begin
                iters = (1:2:10, 4:1:6)
                checkPSconstructor(iters)
            end
            @testset "mixed" begin
                for iters in [(1:2, 4:2:6), (1:2, Base.OneTo(4), 1:3:10)]
                    checkPSconstructor(iters)
                end
            end

            @testset "empty" begin
                iters = (1:1,)
                ps = ProductSplit(iters, 10, 2)
                @test isempty(ps)
                @test length(ps) == 0
            end

            @testset "first and last ind" begin
                for iters in Any[(1:10,), (1:2, Base.OneTo(4), 1:3:10)]
                    ps = ProductSplit(iters, 2, 1)
                    @test firstindex(ps) == 1
                    @test ParallelUtilities.firstindexglobal(ps) == 1
                    @test ParallelUtilities.lastindexglobal(ps) == div(prod(length, iters), 2)
                    @test lastindex(ps) == div(prod(length, iters), 2)
                    @test lastindex(ps) == length(ps)
                    ps = ProductSplit(iters, 2, 2)
                    @test ParallelUtilities.firstindexglobal(ps) == div(prod(length, iters), 2) + 1
                    @test firstindex(ps) == 1
                    @test ParallelUtilities.lastindexglobal(ps) == prod(length, iters)
                    @test lastindex(ps) == length(ps)

                    for np in prod(length, iters) + 1:prod(length, iters) + 10,
                        p in prod(length, iters) + 1:np

                        ps = ProductSplit(iters, np, p)
                        @test ParallelUtilities.firstindexglobal(ps) == prod(length, iters) + 1
                        @test ParallelUtilities.lastindexglobal(ps) == prod(length, iters)
                    end
                end
            end
        end

        @testset "firstlast" begin
            @testset "first" begin

                @test ParallelUtilities._first(()) == ()

                for iters in various_iters, np = 1:prod(length, iters)

                    ps = ProductSplit(iters, np, 1)
                    @test first(ps) == map(first, iters)
                end
            end
            @testset "last" begin

                @test ParallelUtilities._last(()) == ()

                for iters in various_iters, np = 1:prod(length, iters)

                    ps = ProductSplit(iters, np, np)
                    @test last(ps) == map(last, iters)
                end
            end
        end

        @testset "extrema" begin

            @testset "min max extrema" begin
                function checkPSextrema(iters, (fn_el, fn), npmax = 10)
                    for np = 1:npmax, p = 1:np
                        ps = ProductSplit(iters, np, p)
                        if isempty(ps)
                            continue
                        end
                        pcol = collect(ps)
                        for dims in 1:length(iters)
                            @test begin
                                res = fn_el(ps, dims = dims) == fn(x[dims] for x in pcol)
                                if !res
                                    show(ps)
                                end
                                res
                            end
                        end
                        if _niterators(ps) == 1
                            @test begin
                                res = fn_el(ps) == fn(x[1] for x in pcol)
                                if !res
                                    show(ps)
                                end
                                res
                            end
                        end
                    end
                end

                for iters in various_iters,
                    fntup in [(maximumelement, maximum), (minimumelement, minimum), (extremaelement, extrema)]
                    checkPSextrema(iters, fntup)
                end

                @test minimumelement(ProductSplit((1:5,), 2, 1)) == 1
                @test maximumelement(ProductSplit((1:5,), 2, 1)) == 3
                @test extremaelement(ProductSplit((1:5,), 2, 1)) == (1, 3)

                @test minimumelement(ProductSplit((1:5,), 2, 2)) == 4
                @test maximumelement(ProductSplit((1:5,), 2, 2)) == 5
                @test extremaelement(ProductSplit((1:5,), 2, 2)) == (4, 5)
            end

            @testset "extremadims" begin
                ps = ProductSplit((1:10,), 2, 1)
                @test ParallelUtilities._extremadims(ps, 1,()) == ()
                for iters in various_iters
                    dims = length(iters)
                    for np = 1:prod(length, iters) + 1, proc_id = 1:np
                        ps = ProductSplit(iters, np, proc_id)
                        if isempty(ps)
                            @test_throws ArgumentError extremadims(ps)
                        else
                            ext = Tuple(map(extrema, zip(collect(ps)...)))
                            @test extremadims(ps) == ext
                        end
                    end
                end
            end

            @testset "extrema_commonlastdim" begin
                iters = (1:10, 4:6, 1:4)
                ps = ProductSplit(iters, 37, 8)
                @test extrema_commonlastdim(ps) == ([(9, 1), (6, 1)], [(2, 2), (4, 2)])
                ps = ProductSplit(iters, prod(length, iters) + 1, prod(length, iters) + 1)
                @test extrema_commonlastdim(ps) === nothing
            end
        end

        @testset "in" begin

            function checkifpresent(iters, npmax = 10)
                for np = 1:npmax, p = 1:np
                    ps = ProductSplit(iters, np, p)
                    if isempty(ps)
                        continue
                    end
                    pcol = collect(ps)

                    for el in pcol
                        # It should be contained in this iterator
                        @test el in ps
                        for p2 in 1:np
                            # It should not be contained anywhere else
                            p2 == p && continue
                            ps2 = ProductSplit(iters, np, p2)
                            @test !(el in ps2)
                        end
                    end
                end
            end

            for iters in various_iters
                checkifpresent(iters)
            end

            @test ParallelUtilities._infullrange((), ())
        end

        @testset "whichproc + procrange_recast" begin
            np, proc_id = 5, 5
            iters = (1:10, 4:6, 1:4)
            ps = ProductSplit(iters, np, proc_id)
            @test whichproc(iters, first(ps), 1) == 1
            @test whichproc(ps, first(ps)) == proc_id
            @test whichproc(ps, last(ps)) == proc_id
            @test whichproc(iters,(100, 100, 100), 1) === nothing
            @test procrange_recast(iters, ps, 1) == 1:1
            @test procrange_recast(ps, 1) == 1:1

            smalleriter = (1:1, 1:1, 1:1)
            err = ParallelUtilities.TaskNotPresentError(smalleriter, first(ps))
            @test_throws err procrange_recast(smalleriter, ps, 1)
            smalleriter = (7:9, 4:6, 1:4)
            err = ParallelUtilities.TaskNotPresentError(smalleriter, last(ps))
            @test_throws err procrange_recast(smalleriter, ps, 1)

            iters = (1:1, 2:2)
            ps = ProductSplit(iters, np, proc_id)
            @test procrange_recast(iters, ps, 2) == nothing
            @test procrange_recast(ps, 2) == nothing

            iters = (1:1, 2:2)
            ps = ProductSplit(iters, 1, 1)
            @test procrange_recast(iters, ps, 2) == 1:1
            @test procrange_recast(ps, 2) == 1:1

            iters = (Base.OneTo(2), 2:4)
            ps = ProductSplit(iters, 2, 1)
            @test procrange_recast(iters, ps, 1) == 1:1
            @test procrange_recast(iters, ps, 2) == 1:1
            @test procrange_recast(iters, ps, prod(length, iters)) == 1:length(ps)

            for np_new in 1:prod(length, iters)
                for proc_id_new = 1:np_new
                    ps_new = ProductSplit(iters, np_new, proc_id_new)

                    for val in ps_new
                        # Should loop only if ps_new is non-empty
                        @test whichproc(iters, val, np_new) == proc_id_new
                    end
                end
                @test procrange_recast(iters, ps, np_new) == (isempty(ps) ? nothing : (whichproc(iters, first(ps), np_new):whichproc(iters, last(ps), np_new)))
                @test procrange_recast(ps, np_new) == (isempty(ps) ? nothing : (whichproc(iters, first(ps), np_new):whichproc(iters, last(ps), np_new)))
            end

            @testset "different set" begin
                iters = (1:100, 1:4000)
                ps = ProductSplit((20:30, 1:1), 2, 1)
                @test procrange_recast(iters, ps, 700) == 1:1
                ps = ProductSplit((20:30, 1:1), 2, 2)
                @test procrange_recast(iters, ps, 700) == 1:1

                iters = (1:1, 2:2)
                ps = ProductSplit((20:30, 2:2), 2, 1)
                @test_throws ParallelUtilities.TaskNotPresentError procrange_recast(iters, ps, 3)
                ps = ProductSplit((1:30, 2:2), 2, 1)
                @test_throws ParallelUtilities.TaskNotPresentError procrange_recast(iters, ps, 3)
            end
        end

        @testset "indexinproduct" begin
            @test indexinproduct((1:4, 2:3:8), (3, 5)) == 7
            @test indexinproduct((1:4, 2:3:8), (3, 6)) === nothing
            @test_throws ArgumentError indexinproduct((), ())
        end

        @testset "localindex" begin
            for iters in various_iters
                for np = 1:prod(length, iters), proc_id = 1:np
                    ps = ProductSplit(iters, np, proc_id)
                    for (ind, val) in enumerate(ps)
                        @test localindex(ps, val) == ind
                    end
                end
            end
        end

        @testset "whichproc_localindex" begin
            for iters in various_iters
                iters isa Tuple{AbstractUnitRange, Vararg{AbstractUnitRange}} || continue
                for np = 1:prod(length, iters), proc_id = 1:np
                    ps_col = collect(ProductSplit(iters, np, proc_id))
                    ps_col_rev = [reverse(t) for t in ps_col]
                    for val in ps_col
                        p, ind = whichproc_localindex(iters, val, np)
                        @test p == proc_id
                        ind_in_arr = searchsortedfirst(ps_col_rev, reverse(val))
                        @test ind == ind_in_arr
                    end
                end
            end
        end

        @testset "getindex" begin

            @test ParallelUtilities._getindex((), 1) == ()
            @test ParallelUtilities._getindex((), 1, 2) == ()

            @test ParallelUtilities.childindex((), 1) == (1,)

            for iters in various_iters
                for np = 1:prod(length, iters), p = 1:np
                    ps = ProductSplit(iters, np, p)
                    ps_col = collect(ps)
                    for i in 1:length(ps)
                        @test ps[i] == ps_col[i]
                    end
                    @test ps[end] == ps[length(ps)]
                    for ind in [0, length(ps) + 1]
                       @test_throws ParallelUtilities.BoundsError(ps, ind) ps[ind]
                    end
                end
            end
        end
    end
    @testsetwithinfo "ProductSection" begin
        @testset "Constructor" begin
            function testPS(iterators)
                itp = collect(Iterators.product(iterators...))
                l = length(itp)
                for startind in 1:l, endind in startind:l
                    ps = ProductSection(iterators, startind:endind)
                    @test eltype(ps) == Tuple{map(eltype, iterators)...}
                    for (psind, ind) in enumerate(startind:endind)
                        @test ps[psind] == itp[ind]
                    end
                end
            end

            for iter in various_iters
                testPS(iter)
            end

            @test_throws ArgumentError ProductSection((), 2:3)
        end
    end
    @testset "dropleading" begin
        ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);
        @test dropleading(ps) isa ProductSection
        @test collect(dropleading(ps)) == [(4, 1), (2, 2), (3, 2)]
        @test collect(dropleading(dropleading(ps))) == [(1,), (2,)]

        ps = ProductSection((1:5, 2:4, 1:3), 5:8);
        @test dropleading(ps) isa ProductSection
        @test collect(dropleading(ps)) == [(2, 1), (3, 1)]
        @test collect(dropleading(dropleading(ps))) == [(1,)]
    end
    @testset "nelements" begin
        ps = ProductSplit((1:5, 2:4, 1:3), 7, 3);
        @test nelements(ps, dims = 1) == 5
        @test nelements(ps, dims = 2) == 3
        @test nelements(ps, dims = 3) == 2
        @test_throws ArgumentError nelements(ps, dims = 0)
        @test_throws ArgumentError nelements(ps, dims = 4)

        ps = ProductSection((1:5, 2:4, 1:3), 5:8);
        @test nelements(ps, dims =1) == 4
        @test nelements(ps, dims =2) == 2
        @test nelements(ps, dims =3) == 1

        ps = ProductSection((1:5, 2:4, 1:3), 5:11);
        @test nelements(ps, dims = 1) == 5
        @test nelements(ps, dims = 2) == 3
        @test nelements(ps, dims = 3) == 1

        ps = ProductSection((1:5, 2:4, 1:3), 4:8);
        @test nelements(ps, dims = 1) == 5
        @test nelements(ps, dims = 2) == 2
        @test nelements(ps, dims = 3) == 1

        ps = ProductSection((1:5, 2:4, 1:3), 4:9);
        @test nelements(ps, dims = 1) == 5
        @test nelements(ps, dims = 2) == 2
        @test nelements(ps, dims = 3) == 1
    end

    @test ParallelUtilities._checknorollover((), (), ())
end;

@testset "ReverseLexicographicTuple" begin
    @testset "isless" begin
        a = ParallelUtilities.ReverseLexicographicTuple((1, 2, 3))
        b = ParallelUtilities.ReverseLexicographicTuple((2, 2, 3))
        @test a < b
        @test a <= b
        b = ParallelUtilities.ReverseLexicographicTuple((1, 1, 3))
        @test b < a
        @test b <= a
        b = ParallelUtilities.ReverseLexicographicTuple((2, 1, 3))
        @test b < a
        @test b <= a
        b = ParallelUtilities.ReverseLexicographicTuple((2, 1, 4))
        @test a < b
        @test a <= b
    end
    @testset "equal" begin
        a = ParallelUtilities.ReverseLexicographicTuple((1, 2, 3))
        @test a == a
        @test isequal(a, a)
        @test a <= a
        b = ParallelUtilities.ReverseLexicographicTuple(a.t)
        @test a == b
        @test isequal(a, b)
        @test a <= b
    end
end;
