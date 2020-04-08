module DiffusionMCMC

    using DiffusionDefinition
    using DiffObservScheme
    using GuidedProposals
    using StochasticProcessMCMC
    
    greet() = print("Hello World!")

    #include("types.jl") # ✗
    #include("callbacks.jl") # ✗
    #include("blocking.jl") # ✗
    #include("workspaces.jl") # ✗
    #include("updates.jl") # ✗
end # module
