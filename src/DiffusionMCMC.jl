module DiffusionMCMC

    using DiffusionDefinition
    using DiffObservScheme
    using GuidedProposals
    using ExtensibleMCMC
    const eMCMC = ExtensibleMCMC

    include("types.jl") # ✔
    include("callbacks.jl") # ✗
    include("blocking.jl") # ✗/✔ (TODO some additional functions to be written)
    #include("workspaces.jl") # ✗
    #include("updates.jl") # ✗
end # module
