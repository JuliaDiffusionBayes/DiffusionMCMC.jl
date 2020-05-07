#===============================================================================
                    New definitions of introduced updates
===============================================================================#

struct PathImputation{T} <: MCMCDiffusionImputation
    ρs::Vector{Vector{T}}
    aux_laws
    #adpt::
    PathImputation(ρ::T, P) where T<:Number = new{T}([[ρ]], P)

    PathImputation(ρ::T, P) where T<:Vector = new{eltype(ρ)}([ρ], P)

    function PathImputation(ρs::T, P) where T<:Vector{<:Vector}
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
