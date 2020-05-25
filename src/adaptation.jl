"""
    mutable struct AdaptationPathImputation <: eMCMC.Adaptation
        proposed::Int64
        accepted::Int64
        target_accpt_rate::Float64
        adapt_every_k_steps::Int64
        scale::Float64
        min::Float64
        max::Float64
        offset::Float64
    end

A struct containing information about the way in which to adapt the memory
parameter of the preconditioned Crank–Nicolson scheme. `proposed` and `accepted`
are the internal counters that keep track of the number of proposed and accepted
samples. `target_accpt_rate` is the acceptance rate of the Metropolis-Hastings
steps that is supposed to be targetted. `min` is used to enforce the minimal
allowable range that the random walker can sample from, `max` enforces the
maximum. `offset` introduces some delay in the start of decelerting the
adaptation extent and `scale` is a scaling parameter for adaptation speed.
"""
mutable struct AdaptationPathImputation <: eMCMC.Adaptation
    proposed::Int64
    accepted::Int64
    target_accpt_rate::Float64
    adapt_every_k_steps::Int64
    scale::Float64
    min::Float64
    max::Float64
    offset::Float64

    function AdaptationPathImputation(; kwargs...)
        trgt_ar = get(kwargs, :target_accpt_rate, 0.234)
        steps = get(kwargs, :adapt_every_k_steps, 100)
        scale = get(kwargs, :scale, 1.0)
        min = get(kwargs, :min, 1e-12)
        max = get(kwargs, :max, 1.0 - 1e-12)
        offset = get(kwargs, :offset, 1e2)
        new(0, 0, trgt_ar, steps, scale, min, max, offset)
    end
end

"""
    acceptance_rate(adpt::AdaptationPathImputation)

Compute current acceptance rate of the Metropolis-Hastings update step
"""
eMCMC.acceptance_rate(adpt::AdaptationPathImputation) = (
    (adpt.proposed == 0) ? 0.0 : adpt.accepted/adpt.proposed
)

"""
    acceptance_rate!(adpt::AdaptationPathImputation)

Destructive computation of a current acceptance rate that also resets the
number of proposals and accepted samples to zeros.
"""
function eMCMC.acceptance_rate!(adpt::AdaptationPathImputation)
    a_r = eMCMC.acceptance_rate(adpt)
    reset!(adpt)
    a_r
end

"""
    reset!(adpt::AdaptationPathImputation)

Reset the number of proposals and accepted samples to zero.
"""
function DataStructures.reset!(adpt::AdaptationPathImputation)
    adpt.proposed = 0
    adpt.accepted = 0
end

"""
    readjust!(
        imp::PathImputation, adpt::AdaptationPathImputation, mcmc_iter, i=1
    )

Adaptive readjustment for the `i`th recording of the preconditioned
Crank–Nicolson scheme's memory parameter.
"""
function eMCMC.readjust!(
        imp::PathImputation, adpt::AdaptationPathImputation, mcmc_iter, i=1
    )
    δ = eMCMC.compute_δ(adpt, mcmc_iter)
    a_r = eMCMC.acceptance_rate!(adpt)
    ϵ = eMCMC.compute_ϵ(first(imp.ρs[i]), adpt, a_r, δ, -1.0, logit, sigmoid)
    imp.ρs[i] .= ϵ
    ϵ
end

sigmoid(x, a=1.0) = 1.0 / (1.0 + exp(-a*x))
logit(x, a=1.0) = (log(x) - log(1-x))/a

"""
    register!(updt, adpt::AdaptationPathImputation, accepted::Bool, ::Any)

Register the result of the acceptance decision in the Metropolis-Hastings step.
"""
function eMCMC.register!(updt, adpt::AdaptationPathImputation, accepted::Bool, ::Any)
    adpt.accepted += accepted
    adpt.proposed += 1
end

"""
    time_to_update(::Val{true}, adpt::AdaptationPathImputation)

Return true if it's the time to update the memory parameter of the
preconditioned Crank–Nicolson scheme.
"""
function eMCMC.time_to_update(::Val{true}, adpt::AdaptationPathImputation)
    adpt.proposed >= adpt.adapt_every_k_steps
end
#===============================================================================

                        AdaptationMultiPathImputation

        A struct that gathers multiple AdaptationPathImputatation—one
        for each recording
===============================================================================#

struct AdaptationMultiPathImputation <: eMCMC.Adaptation
    v::Vector{AdaptationPathImputation}

    AdaptationMultiPathImputation(adpt::AdaptationPathImputation) = new([adpt])
end

function eMCMC.register!(
        updt,
        adpt::AdaptationMultiPathImputation,
        accepted::Vector{Bool},
        θ
    )
    for i in eachindex(accepted)
        eMCMC.register!(updt, adpt.v[i], accepted[i], θ)
    end
end

function eMCMC.time_to_update(v::Val{true}, adpt::AdaptationMultiPathImputation)
    eMCMC.time_to_update(v, adpt.v[1])
end

function eMCMC.readjust!(
        imp::PathImputation, adpt::AdaptationMultiPathImputation, mcmc_iter
    )
    for i in eachindex(adpt.v)
        eMCMC.readjust!(imp, adpt.v[i], mcmc_iter, i)
    end
end

init_path_imp_adpt(adpt::NoAdaptation) = (adpt, NoAdaptation)

init_path_imp_adpt(adpt::AdaptationMultiPathImputation) = (adpt, typeof(adpt))

function init_path_imp_adpt(adpt::AdaptationPathImputation)
    _adpt = AdaptationMultiPathImputation(adpt)
    _adpt, typeof(_adpt)
end

resize_adpt!(adpt::NoAdaptation, num_recordings) = nothing

function resize_adpt!(adpt::AdaptationMultiPathImputation, num_recordings)
    Δ = num_recordings - length(adpt.v)
    for i in 1:Δ
        push!(adpt.v, deepcopy(adpt.v[1]))
    end
end
