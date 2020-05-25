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
struct PathImputation{T,A} <: MCMCDiffusionImputation
    ρs::Vector{Vector{T}}
    aux_laws
    adpt::A

    function PathImputation(ρ::T, P; adpt=NoAdaptation()) where T<:Number
        _adpt, A = init_path_imp_adpt(adpt)
        new{T,A}([[ρ]], P, _adpt)
    end

    function PathImputation(ρ::T, P; adpt=NoAdaptation()) where T<:Vector
        _adpt, A = init_path_imp_adpt(adpt)
        new{eltype(ρ),A}([ρ], P, _adpt)
    end

    function PathImputation(
            ρs::T, P; adpt=NoAdaptation()
        ) where T<:Vector{<:Vector}
        _adpt, A = init_path_imp_adpt(adpt)
        new{eltype(ρs[1]),A}(ρs, P, _adpt)
    end
end

init_update!(updt::eMCMC.MCMCUpdate, block_layout) = nothing

function init_update!(updt::PathImputation, block_layout)
    num_recordings = length(block_layout)
    Δ = num_recordings - length(updt.ρs)
    if Δ > 0
        for i in 1:Δ
            push!(updt.ρs, [updt.ρs[1][1]])
        end
    end
    for i in 1:length(updt.ρs)
        update_ρ!(updt.ρs[i], block_layout[i])
    end
    resize_adpt!(updt.adpt, num_recordings)
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
