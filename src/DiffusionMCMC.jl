module DiffusionMCMC

    using DiffusionDefinition
    using DiffObservScheme
    using GuidedProposals
    using ExtensibleMCMC
    const eMCMC = ExtensibleMCMC
    const DOS = DiffObservScheme

    const _DEFAULT_TIME_TRANSFORM = identity

    include("types.jl") # ✔
    include("callbacks.jl") # ✗
    include("blocking.jl") # ✗/✔ (TODO some additional functions to be written)
    include("workspaces.jl") # ✗
    include("updates.jl") # ✗

    export DiffusionMCMCBackend
    export PathImputation, StartingPointsUpdate, StartingPointsLangevinUpdate
end # module
