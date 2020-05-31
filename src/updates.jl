#===============================================================================
                    New definitions of introduced updates
===============================================================================#


#===============================================================================
                            Imputation of paths
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


#===============================================================================
                        Conjugate Gaussian updates
===============================================================================#

struct DiffusionConjugGsnUpdate{K,TP} <: eMCMC.MCMCConjugateParamUpdate
    coords::K
    prior::TP
    adpt::NoAdaptation

    function DiffusionConjugGsnUpdate(coords, prior)
        if typeof(prior) <: Normal
            prior = Gaussian( [prior.μ], reshape(prior.σ, (1,1)) )
        elseif typeof(prior) <: MvNormal
            prior = Gaussian(prior.μ, prior.Σ.mat)
        else
            @assert typeof(prior) <: Gaussian
        end

        if typeof(coords) <: Number
            coords = [coords]
        end
        new{typeof(coords), typeof(prior)}(coords, prior, NoAdaptation())
    end
end

struct ConjugGsnHelper{T,N,K}
    relevant_recordings::Vector{Int64}
    sub_μ_idx::Vector{K}
    θ_names::Vector{Val}
    θᶜ_names::Vector{Val}

    function ConjugGsnHelper(
            updt::DiffusionConjugGsnUpdate,
            global_ws,
            PPP,
            lpn::LocalUpdtParamNames;
            computation_type=:outofplace
        )
        _glob_pnames = global_ws.pnames[updt.coords]
        relevant_recordings = Int64[]
        sub_μ_idx = Vector{Vector{Int64}}(undef, 0)
        θᶜ_names = Val[]
        θ_names = Val[]
        all_θ_names = Symbol[]

        for (i, PP) in enumerate(PPP)
            P = first(PP).P_target
            if length(lpn.θ_local_names) > 0
                push!(relevant_recordings, i)

                # sort by the local index of θ
                θ_loc_names = sort(collect(lpn.θ_local_names[i]), by=x->x[1])
                sub_idx = getindex.(θ_loc_names, 1)
                push!(sub_μ_idx, sub_idx)

                sub_name = getindex.(θ_loc_names, 2)
                push!(
                    θᶜ_names,
                    Val(
                        Tuple(
                            filter(
                                pn->(
                                    !(pn in sub_name) &&
                                    !DD.ignore_for_cu(Val(pn), P)
                                ),
                                DD.parameter_names(P)
                            )
                        )
                    )
                )
                push!( θ_names, Val(Tuple(sub_name)) )
                append!(all_θ_names, sub_name)
            else
                push!(sub_μ_idx, Int64[])
                push!(θᶜ_names, Val[])
                push!(θ_names, Val[])
            end
        end
        u_θ_names = unique(all_θ_names)
        sub_μ_idx = transform_μ_idx(Val(computation_type), sub_μ_idx)
        new{computation_type, length(u_θ_names), eltype(sub_μ_idx)}(
            relevant_recordings,
            sub_μ_idx,
            θ_names,
            θᶜ_names,
        )
    end

    ConjugGsnHelper() = new{:none, 0, Nothing}()
end

transform_μ_idx(::Val{:inplace}, sub_μ_idx) = sub_μ_idx

function transform_μ_idx(v::Val{:outofplace}, sub_μ_idx)
    map(i->transform_μ_idx(v, i, Val(length(i))), sub_μ_idx)
end

function transform_μ_idx(::Val{:outofplace}, sub_μ_idx, ::Val{N}) where N
    SVector{N,Int64}(sub_μ_idx)
end

function μ_init(cu_helper::ConjugGsnHelper{:outofplace,N}) where N
    zero(MVector{N,Float64})
end

function W_init(cu_helper::ConjugGsnHelper{:outofplace,N}) where N
    zero(MMatrix{N,N,Float64})
end

function μ_init(cu_helper::ConjugGsnHelper{:outofplace}, ::SVector{N}) where N
    zero(SVector{N,Float64})
end

μ_init(cu_helper::ConjugGsnHelper{:inplace,N}) where N = zeros(Float64, N)
W_init(cu_helper::ConjugGsnHelper{:inplace,N}) where N = zeros(Float64, (N,N))

function μ_init(cu_helper::ConjugGsnHelper{:inplace,N}, v) where N
    zeros(Float64, length(v))
end


@generated function conjug_θᶜ(
        P_target, pnames::Val{T}, ::ConjugGsnHelper{:outofplace}
    ) where T
    N = length(T)
    data = Expr(
        :tuple,
        1.0,
        (:(getfield(P_target, T[$i])) for i in 1:N)...,
    )
    vec = Expr(:call, MVector{N+1}, data)
end

@generated function conjug_θᶜ(
        P_target, pnames::Val{T}, ::ConjugGsnHelper{:inplace}
    ) where T
    N = length(T)
    data = Expr(
        :tuple,
        1.0,
        (:(getfield(P_target, T[$i])) for i in 1:N)...,
    )
    vec = Expr(:call, :collect, data)
end


function eMCMC.proposal!(
        updt::DiffusionConjugGsnUpdate,
        global_ws,
        local_ws::eMCMC.LocalWorkspace,
        step,
    )
    state°(local_ws) .= conjugate_draw(
        local_ws.cu_helper,
        local_ws.sub_ws_diff.P,
        updt.prior,
        local_ws.sub_ws_diff.XX,
    )
end

function conjugate_draw(cu_helper, PP, prior, XX)
    μ = μ_init(cu_helper)
    W = W_init(cu_helper)
    for i in cu_helper.relevant_recordings
        μi = μ_init(cu_helper, cu_helper.sub_μ_idx[i])
        Wi = μi*μi'
        P = first(PP[i]).P_target
        θᶜ = conjug_θᶜ(P, cu_helper.θᶜ_names[i], cu_helper)

        μi, Wi = compute_μ_and_W(
            μi, Wi, cu_helper.θ_names[i], cu_helper.θᶜ_names[i], θᶜ, P, XX[i],
            cu_helper
        )

        sidx = cu_helper.sub_μ_idx[i]
        view(μ, sidx) .+= μi
        view(W, sidx, sidx) .+= Wi
    end
    μ_post, Σ_post = compute_posterior_μ_and_Σ(μ, W, prior)
    rand(Gaussian(μ_post, Σ_post))
end

function compute_μ_and_W(
        μ, W, θ_names, θᶜ_names, θᶜ, P_target::S, XX, ::ConjugGsnHelper{:outofplace};
        nnh = DD.num_non_hypo(S)
    ) where S
    for X in XX
        num_pts = length(X)
        for i in 1:num_pts-1
            φₜ = φ(θ_names, X.t[i], X.x[i], P_target, nnh)
            φᶜₜ = φᶜ(θᶜ_names, θᶜ, X.t[i], X.x[i], P_target, nnh)
            Γ⁻¹ = DD.hypo_a_inv(X.t[i], X.x[i], P_target)
            dt = X.t[i+1] - X.t[i]
            dy = DD.nonhypo(X.x[i+1], P_target) - DD.nonhypo(X.x[i], P_target)
            μ = μ + φₜ'*Γ⁻¹*dy - φₜ'*Γ⁻¹*φᶜₜ*dt
            W = W + φₜ'*Γ⁻¹*φₜ*dt
        end
    end
    μ, W
end

function compute_μ_and_W(
        μ, W, θ_names, P_target, XX, θᶜ, ::ConjugGsnHelper{:inplace}
    )
    error("not implemented, need buffers...")
end


function compute_posterior_μ_and_Σ(μ, W, prior)
    Σ = Symmetric(inv(W + inv(prior.Σ)))
    Σ * (μ .+ prior.Σ\prior.μ), Σ
end

const _GENERATED = Val(:gen)

function φ(v::Val, t, x, P::S, num_non_smooth_coords=DD.num_non_hypo(S)) where S
    φ(_GENERATED, v, t, x, P, num_non_smooth_coords)
end

@generated function φ(::Val{:gen}, ::Val{Ts}, t, x, P::S, ::Val{N}) where {Ts,S,N}
	args = [Expr(:call, :Val, :(Ts[$i])) for i in 1:length(Ts)]
    data = Expr(
        :call,
        :tuplejoin,#TODO try to do tuple join not in the expression
        ( :(DD.phi($a, t, x, P)) for a in args )...
    )
    mat = Expr(:call, SMatrix{N,length(Ts)}, data)
    return mat
end

"""
    φᶜ(v, θᶜ, t, x, P::S, n=DD.num_non_hypo(S)) where S

Remainder term in the drift after removing φ'θ from it.
"""
function φᶜ(v, θᶜ, t, x, P::S, n=DD.num_non_hypo(S)) where S
    φᶜlinear(v, t, x, P, n) * θᶜ
end

function φᶜlinear(v, t, x, P::S, nnh=DD.num_non_hypo(S)) where S
    φᶜlinear(_GENERATED, v, t, x, P, nnh)
end

@generated function φᶜlinear(::Val{:gen}, ::Val{Ts}, t, x, P::S, ::Val{N}) where {Ts,S,N}
    args = [Expr(:call, :Val, :(Ts[$i])) for i in 1:length(Ts)]
    data = Expr(
        :call,
        :tuplejoin,
        :( DD.phi(Val(:intercept), t, x, P) ),
        ( :(DD.phi($a, t, x, P)) for a in args )...
    )
    mat = Expr(:call, SMatrix{N,length(Ts)+1}, data)
    return mat
end

@inline tuplejoin(x) = x
@inline tuplejoin(x, y) = (x..., y...)
@inline tuplejoin(x, y, z...) = tuplejoin(tuplejoin(x, y), z...)
#TODO at the accept reject step it needs to accept a boolean not only a vec of booleans
