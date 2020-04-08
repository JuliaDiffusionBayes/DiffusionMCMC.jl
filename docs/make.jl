using Documenter, DiffusionMCMC

makedocs(;
    modules=[DiffusionMCMC],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl/blob/{commit}{path}#L{line}",
    sitename="DiffusionMCMC.jl",
    authors="Sebastiano Grazzi, Frank van der Meulen, Marcin Mider, Moritz Schauer",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaDiffusionBayes/DiffusionMCMC.jl",
)
