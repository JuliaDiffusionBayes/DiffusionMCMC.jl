module DiffusionMCMC

    using DiffusionDefinition, ObservationSchemes, GuidedProposals
    using ExtensibleMCMC
    using OrderedCollections

    import ExtensibleMCMC: ll, ll°, state, state°, accepted, set_accepted!

    const eMCMC = ExtensibleMCMC
    const OBS = ObservationSchemes
    const GP = GuidedProposals
    const DD = DiffusionDefinition

    OBS.var_parameter_names(P::DD.DiffusionProcess) = DD.var_parameter_names(P)

    const _DEFAULT_TIME_TRANSFORM = identity
    const _LL = Val(:ll)
    const _NEXT = Val(:next)

    include("types.jl") # ✔
    include("callbacks.jl") # ✗
    include("blocking.jl") # ✗/✔ (TODO some additional functions to be written)
    include("workspaces.jl") # ✗
    include("path_saving_buffer.jl")
    include("updates.jl") # ✗
    include("schedule.jl")
    include("run!_alterations.jl")

    export DiffusionMCMCBackend
    export PathImputation, StartingPointsUpdate, StartingPointsLangevinUpdate
end # module
