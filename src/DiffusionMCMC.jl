module DiffusionMCMC

    using DiffusionDefinition
    using ObservationSchemes
    using GuidedProposals
    using ExtensibleMCMC
    import ExtensibleMCMC: ll, ll°, state, state°, accepted
    const eMCMC = ExtensibleMCMC
    const OBS = ObservationSchemes

    const _DEFAULT_TIME_TRANSFORM = identity

    include("types.jl") # ✔
    include("callbacks.jl") # ✗
    include("blocking.jl") # ✗/✔ (TODO some additional functions to be written)
    include("workspaces.jl") # ✗
    include("updates.jl") # ✗
    include("schedule.jl")
    include("run!_alterations.jl")

    export DiffusionMCMCBackend
    export PathImputation, StartingPointsUpdate, StartingPointsLangevinUpdate
end # module
