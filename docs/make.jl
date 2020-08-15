using Documenter
using ParallelUtilities

DocMeta.setdocmeta!(ParallelUtilities, :DocTestSetup, :(using ParallelUtilities); recursive=true)

makedocs(;
    modules=[ParallelUtilities],
    authors="Jishnu Bhattacharya",
    repo="https://github.com/jishnub/ParallelUtilities.jl/blob/{commit}{path}#L{line}",
    sitename="ParallelUtilities.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jishnub.github.io/ParallelUtilities.jl",
        assets=String[],
    ),
    pages=[
        "Reference" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jishnub/ParallelUtilities.jl",
)
