#===============================================================================
                    New definitions of introduced updates
===============================================================================#
"""
    struct PathImputation{T} <: MCMCDiffusionImputation
        ρs::Vector{Vector{T}}
        aux_laws
    end

Update type. An indicator for imputation of unobserved path segments. `ρs` are
the memory parameters for the preconditioned Crank–Nicolson scheme and
`aux_laws` are the laws for the auxiliary diffusions.

    PathImputation(ρ::T, P) where T<:Number

Base constructor v1. Initialize all `ρs` with the same value `ρ` and set
auxiliary laws to `P`.

    PathImputation(ρ::T, P) where T<:Vector
Base constructor v2. Initialize each recording with its own value of `ρ` and set
auxiliary laws to `P`.
"""
struct PathImputation{T} <: MCMCDiffusionImputation
    ρs::Vector{Vector{T}} #TODO change to simply Vector{T} as each recording may have only one ρ
    aux_laws
    #adpt::
    PathImputation(ρ::T, P) where T<:Number = new{T}([[ρ]], P)

    PathImputation(ρ::T, P) where T<:Vector = new{eltype(ρ)}([ρ], P)

    function PathImputation(ρs::T, P) where T<:Vector{<:Vector} #TODO remove
        new{eltype(ρs[1])}(ρs, P)
    end
end

init_update!(updt::eMCMC.MCMCUpdate, block_layout) = nothing

function init_update!(updt::PathImputation, block_layout)
    Δ = length(block_layout) - length(updt.ρs)
    if Δ > 0
        for i in 1:Δ
            push!(updt.ρs, [0.8])
        end
    end
    for i in 1:length(updt.ρs)
        update_ρ!(updt.ρs[i], block_layout[i])
    end
end

function update_ρ!(ρs::Vector, block_layout)
    N = length(block_layout[2])
    if length(ρs) < N
        for i in 1:(N-length(ρs))
            push!(ρs, ρs[1])
        end
    end
end

struct StartingPointsImputation <: MCMCDiffusionImputation end

struct StartingPointsLangevinImputation <: MCMCDiffusionImputation end
