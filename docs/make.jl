using Documenter, DiffusionMCMC

makedocs(;
    modules=[DiffusionMCMC],
    format=Documenter.HTML(
        mathengine = Documenter.MathJax(
            Dict(
                :TeX => Dict(
                    :equationNumbers => Dict(
                        :autoNumber => "AMS"
                    ),
                    :Macros => Dict(
                        :dd => "{\\textrm d}",
                        :RR => "\\mathbb{R}",
                        :wt => ["\\widetilde{#1}", 1]
                    ),
                ),
            ),
        ),
        collapselevel = 1,
    ),
    pages=[
        "Home" => "index.md",
        "Get started" => joinpath("get_started", "overview.md"),
        "User manual" => Any[
            "Extension of ExtensibleMCMC.jl" => joinpath("manual", "how_extend.md"),
            "Updates & Decorators" => joinpath("manual", "updates_and_decorators.md"),
            "(TODO) Blocking" => joinpath("manual", "blocking.md"),
            "(TODO) Callbacks" => joinpath("manual", "callbacks.md"),
            "Diffusion Workspaces" => joinpath("manual", "workspaces.md"),
            "Conjugate updates" => joinpath("manual", "conjugate_updates.md"),
        ],
        "How to..." => Any[
            "(TODO) Do smoothing" => joinpath("how_to_guides", "smoothing.md"),
            "(TODO) Do blocking" => joinpath("how_to_guides", "blocking.md"),
        ],
        "Tutorials" => Any[
            "Inference & smoothing for the FitzHugh–Nagumo model" => joinpath("tutorials", "simple_inference.md"),
            "Conjugate updates for the FitzHugh–Nagumo model" => joinpath("tutorials", "conjugate_updates.md"),
            "(TODO) Inference from first-passage time observations" => joinpath("tutorials", "inference_fpt.md"),
            "(TODO) Mixed-effect models" => joinpath("tutorials", "mixed_effects_models.md"),
        ],
        "Index" => "module_index.md",
    ],
    repo="https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl/blob/{commit}{path}#L{line}",
    sitename="DiffusionMCMC.jl",
    authors="Sebastiano Grazzi, Frank van der Meulen, Marcin Mider, Moritz Schauer",
    assets=String[],
)

deploydocs(;
    repo="github.com/JuliaDiffusionBayes/DiffusionMCMC.jl",
)
