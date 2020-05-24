#===============================================================================

    Defines:
        - DiffusionGlobalWorkspace      : global workspace for diffusions
        - DiffusionGlobalSubworkspace   : convenience object with containers
        - DiffusionLocalWorkspace       : local workspace for diffusions
        - DiffusionLocalSubworkspace    : convenience object with containers
        - LocalUpdtParamNames           : convenience object for setting params
    and methods for them.


===============================================================================#


#===============================================================================
                        Global workspace for diffusions
===============================================================================#

"""
    struct DiffusionGlobalSubworkspace{T,TGP,TW,TWn,TX} <: eMCMC.GlobalWorkspace{T}
        P::TGP
        WW::TW
        Wnr::TWn
        XX::TX
    end

Convenience sub-workspace that holds
- `P::Vector{<:Vector{<:GuidProp}}`: the `GuidProp` laws [TODO change to PP or
                                     PPP]
- `WW::Vector{<:Vector{<:Trajectory}}`: the containers for sampling the Wiener
                                        noise
- `Wnr::Vector{<:Wiener}`: the flags for the Wiener noise
- `XX::Vector{<:Vector{<:Trajectory}}`: the containers for sampling the
                                        diffusion paths
The outer array corresponds to successive `recording`s, the inner arrays
correspond to successive observations within each recording. Each
`DiffusionSubworkspace` has two `DiffusionGlobalSubworkspace`s, one for the
accepted law/path another for the proposal.

    DiffusionGlobalSubworkspace{T}(
        aux_laws, data::AllObservations, tts, args,
    ) where T

A standard constructor. `aux_laws` is a list of auxiliary laws (one for each
recording), `data` holds the observations, `tts` are the time-grids on which
to do imputation when sampling and `args` are additional positional arguments
passed to initializers of `GuidProp`.
"""
struct DiffusionGlobalSubworkspace{T,TGP,TW,TWn,TX} <: eMCMC.GlobalWorkspace{T}
    P::TGP
    WW::TW
    Wnr::TWn
    XX::TX

    function DiffusionGlobalSubworkspace{T}(
            aux_laws, data::AllObservations, tts, args,
        ) where T
        N = num_recordings(data)
        P = map(1:N) do i
            build_guid_prop(aux_laws[i], data.recordings[i], tts[i], args[i])
        end

        traj = map(1:N) do i
            trajectory(tts[i], data.recordings[i].P)
        end
        XX = map(x->x.process, traj)
        WW = map(x->x.wiener, traj)
        Wnr = [Wiener(data.recordings[i].P) for i=1:N]
        init_paths!(P, WW, Wnr, XX, data)
        new{T,typeof(P),typeof(WW),typeof(Wnr),typeof(XX)}(P, WW, Wnr, XX)
    end
end

"""
    init_paths!(P, WW, Wnr, XX, data)

Sample paths of guided proposals without using the preconditioned Crank–Nicolson
scheme.
"""
function init_paths!(P, WW, Wnr, XX, data)
    num_recordings = length(P)

    # this loop should be entirely parallelizable, as recordings
    # are—by design—conditionally independent
    for i in 1:num_recordings
        success = false
        while !success
            y1 = rand(data.recordings[i].x0_prior)
            success = forward_guide!(P[i], XX[i], WW[i], y1; Wnr=Wnr[i])
        end
    end
end

"""
    struct DiffusionGlobalWorkspace{T,SW,SWD,TD,TX} <: eMCMC.GlobalWorkspace{T}
        sub_ws::SW
        sub_ws_diff::SWD
        sub_ws_diff°::SWD
        data::TD
        stats::GenericChainStats{T}
        XX_buffer::TX
        pnames::Vector{Symbol}
    end

Global workspace for MCMC problems concerning diffusions.

# Arguments
- `sub_ws`: `GenericGlobalSubworkspace` from `ExtensibleMCMC` that stores info
            about the parameter chain
- `sub_ws_diff`: `DiffusionGlobalSubworkspace` containers for the accepted
                 diffusion law/paths
- `sub_ws_diff°`: `DiffusionGlobalSubworkspace` containers for the proposed
                  diffusion law/paths
- `data`: observations
- `stats::GenericChainStats`: generic online statistics for the param chain
- `XX_buffer`: cache for saving a chain of paths
- `pnames`: a list of parameter names that are being updated
"""
struct DiffusionGlobalWorkspace{T,SW,SWD,TD,TX} <: eMCMC.GlobalWorkspace{T}
    sub_ws::SW
    sub_ws_diff::SWD
    sub_ws_diff°::SWD
    data::TD
    stats::GenericChainStats{T}
    XX_buffer::TX
    pnames::Vector{Symbol}
end

function eMCMC.init_global_workspace(
        ::DiffusionMCMCBackend,
        num_mcmc_steps,
        updates::Vector{<:eMCMC.MCMCUpdate},
        data,
        θinit::OrderedDict{Symbol,T};
        kwargs...
    ) where T
    dkwargs = Dict(kwargs)
    M, NU = num_mcmc_steps, length(updates)

    _θ_init = collect(values(θinit))
    _pnames = collect(keys(θinit))

    sub_ws = eMCMC.StandardGlobalSubworkspace(M, NU, data, _θ_init)

    sub_ws_diff = DiffusionGlobalSubworkspace{T}(
        extract_aux_laws(updates, data),
        data,
        OBS.setup_time_grids(
            data,
            get(dkwargs, :dt, 0.01),
            get(dkwargs, :τ, _DEFAULT_TIME_TRANSFORM),
            get(dkwargs, :Ttime, Float64),
            get(dkwargs, :tt, nothing),
        ),
        package(get(dkwargs, :guid_prop_args, tuple()), data), # from ObservationSchemes
    )

    path_saving_buffer = PathSavingBuffer(
        sub_ws_diff.XX,
        get(dkwargs, :path_buffer_size, 1)
    )

    SW, SWD, TD = typeof(sub_ws), typeof(sub_ws_diff), typeof(data)
    TX = typeof(path_saving_buffer)

    DiffusionGlobalWorkspace{T,SW,SWD,TD,TX}(
        sub_ws,
        sub_ws_diff,
        deepcopy(sub_ws_diff),
        data,
        GenericChainStats(_θ_init, NU, M),
        path_saving_buffer,
        _pnames,
    )
end

function extract_aux_laws(updates::Vector{<:eMCMC.MCMCUpdate}, data)
    imps = filter(x->typeof(x)<:PathImputation, updates)
    N = length(imps)
    @assert N > 0
    aux_laws = map(x->OBS.package(x.aux_laws, data), imps)
    @assert N == 1 || all([aux_laws[1]==aux_laws[i] for i=2:N])
    aux_laws[1]
end

#===============================================================================
                        Local workspace for diffusions
===============================================================================#
"""
    struct DiffusionLocalSubworkspace{T,TP,TW,TWn,TX,Tz0} <: eMCMC.LocalWorkspace{T}
        ll::Vector{Float64}
        ll_history::Vector{Vector{Float64}}
        P::TP
        WW::TW
        Wnr::TWn
        XX::TX
        z0s::Tz0
    end

Convenience sub-workspace that holds
- `ll::Vector{Float64}`: computational buffer for log-likelihoods (one for each
                         recording)
- `ll_history::Vector{Vector{Float64}}`:  history of all log-likelihoods
- `P`: appropriately shaped views to containers with `GuidProp`s
- `WW`: appropriately shaped views to containers for sampled Wiener process
- `Wnr`: list of Wiener flags for each recording
- `XX`: appropriately shaped views to containers for sampled diffusion paths
- `z0s`: list of white noise for starting points (one for each recording)
"""
struct DiffusionLocalSubworkspace{T,TP,TW,TWn,TX,Tz0} <: eMCMC.LocalWorkspace{T}
    ll::Vector{Float64}
    ll_history::Vector{Vector{Float64}}
    P::TP
    WW::TW
    Wnr::TWn
    XX::TX
    z0s::Tz0

    function DiffusionLocalSubworkspace{T}(
            ll, ll_hist, P::TP, WW::TW, Wnr::TWn, XX::TX, z0s::Tz0
        ) where {T,TP,TW,TWn,TX,Tz0}
        new{T,TP,TW,TWn,TX,Tz0}(ll, ll_hist, P, WW, Wnr, XX, z0s)
    end
end

function DiffusionLocalSubworkspace{T}(
        sws::DiffusionGlobalSubworkspace,
        block_layout,
        num_mcmc_steps
    ) where T
    N, M = num_mcmc_steps, length(block_layout)
    DiffusionLocalSubworkspace{T}(
        Float64[-Inf for _=1:M],
        [zeros(Float64, M) for _=1:N],
        block_view(sws.P, block_layout),
        block_view(sws.WW, block_layout),
        block_wiener(sws.Wnr, block_layout),
        block_view(sws.XX, block_layout),
        block_start_pts(sws.XX, block_layout)
    )
end

function block_view(v, layout)
    [ view(v[bl_i], bl_i_range) for (bl_i, bl_i_range) in layout ]
end

function block_view(v, layout::Nothing)
    [ view(v[i], :) for i in 1:length(v) ]
end

function block_start_pts(XX, layout::Nothing)
    [ deepcopy(X[1].x[1]) for X in XX ]
end

function block_start_pts(XX, layout)
    [ deepcopy(XX[bl_i][bl_i_range[1]].x[1]) for (bl_i, bl_i_range) in layout ]
end

block_wiener(Wnr, layout::Nothing) = Wnr

block_wiener(Wnr, layout) = [ deepcopy(Wnr[bl_i]) for (bl_i, _) in layout ]

const _SYMS = Tuple{Vararg{Pair{Int64,Symbol},N} where N}
const _OBSIDX = Tuple{Vararg{Pair{Int64,Int64},N} where N}

struct LocalUpdtParamNames
    vp_names::Vector{Tuple{Vararg{Symbol,N} where N}}
    vp_names_aux::Vector{Vector{Tuple{Vararg{Symbol,N} where N}}}
    θ_local_names::Vector{_SYMS}
    θ_local_aux_names::Vector{Vector{_SYMS}}
    θ_local_obs::Vector{Vector{_OBSIDX}}

    function LocalUpdtParamNames(
            PPP::AbstractArray{<:AbstractArray{<:GuidProp}},
            data::AllObservations,
            θinit::Vector{Symbol},
        )
        psym, oind = OBS.local_symbols(data, PPP, x->x.P_target, θinit)
        pauxsym, _ = OBS.local_symbols(data, PPP, x->x.P_aux, θinit)

        new(
            [DD.var_parameter_names(PP) for PP in PPP],
            [[DD.var_parameter_names(P.P_aux) for P in PP] for PP in PPP],
            first.(psym), # target laws are the same on each obs of a single rec
            pauxsym,
            oind,
        )
    end

    LocalUpdtParamNames() = new()
end

"""
    struct DiffusionLocalWorkspace{T,SW,SWD,Tpr} <: eMCMC.LocalWorkspace{T}
        sub_ws::SW
        sub_ws°::SW
        sub_ws_diff::SWD
        sub_ws_diff°::SWD
        xx0_priors::Tpr
        acceptance_history::Vector{Vector{Bool}}
        loc_pnames::LocalUpdtParamNames
        critical_param_change::Vector{Bool}
    end

Local workspace for MCMC problems concerning diffusions.

# Arguments
- `sub_ws`: `GenericLocalSubworkspace` from `ExtensibleMCMC` that acts as a
            local cache for some light parameter computations
- `sub_ws°`: same as `sub_ws`, but concerns the proposed parameter
- `sub_ws_diff`: `DiffusionLocalSubworkspace`, views to containers for the
                 accepted diffusion law/paths
- `sub_ws_diff°`: `DiffusionLocalSubworkspace`, views to containers for the
                  proposed diffusion law/paths
- `xx0_priors`: priors over starting points
- `acceptance_history`: history of results from the accept/reject
                        Metropolis–Hastings step
- `loc_pnames`: various lists of parameter names helping in performing the
                update in a clean way
- `critical_param_change`: a list of flags for whether a given update by itself
                           prompts for recomputation of the guiding term
"""
struct DiffusionLocalWorkspace{T,SW,SWD,Tpr} <: eMCMC.LocalWorkspace{T}
    sub_ws::SW
    sub_ws°::SW
    sub_ws_diff::SWD
    sub_ws_diff°::SWD
    xx0_priors::Tpr
    acceptance_history::Vector{Vector{Bool}}
    loc_pnames::LocalUpdtParamNames
    critical_param_change::Vector{Bool}
end

function create_workspace(
        backend::DiffusionMCMCBackend,
        mcmcupdate,
        global_ws,
        block_layout,
        num_mcmc_steps,
    )
    _state = (
        typeof(mcmcupdate) <: eMCMC.MCMCParamUpdate ?
        copy(state(global_ws, mcmcupdate)) :
        Float64[]
    )
    sub_ws = eMCMC.StandardLocalSubworkspace(_state, num_mcmc_steps)
    xx0_priors = block_setup_x0_priors(global_ws.data, block_layout, global_ws.sub_ws_diff)
    T = eltype(state(global_ws))
    sub_ws_diff = DiffusionLocalSubworkspace{T}(
        global_ws.sub_ws_diff, block_layout, num_mcmc_steps
    )
    init_ll!(sub_ws_diff)

    sub_ws_diff° = DiffusionLocalSubworkspace{T}(
        global_ws.sub_ws_diff°, block_layout, num_mcmc_steps
    )
    a_h = [similar(v, Bool) for v in sub_ws_diff.ll_history]

    init_update!(mcmcupdate, block_layout)

    lpn = (
        typeof(mcmcupdate) <: eMCMC.MCMCParamUpdate ?
        LocalUpdtParamNames(
            sub_ws_diff.P,
            global_ws.data,
            global_ws.pnames[eMCMC.coords(mcmcupdate)]
        ) :
        LocalUpdtParamNames()
    )
    crit_updt = (
        typeof(mcmcupdate) <: eMCMC.MCMCParamUpdate ?
        [
            GP.is_critical_update(
                PP, lpn.θ_local_aux_names[i], lpn.θ_local_obs[i]
            ) for (i,PP) in enumerate(sub_ws_diff.P)
        ] :
        [false for _ in eachindex(sub_ws_diff.P)]
    )

    DiffusionLocalWorkspace{T,typeof(sub_ws),typeof(sub_ws_diff), typeof(xx0_priors)}(
        sub_ws, deepcopy(sub_ws), sub_ws_diff, sub_ws_diff°, xx0_priors, a_h, lpn,
        crit_updt,
    )
end

function init_ll!(sub_ws_diff)
    XXX, PPP = sub_ws_diff.XX, sub_ws_diff.P
    for i in 1:length(sub_ws_diff.ll)
        recompute_guiding_term!(PPP[i])
        sub_ws_diff.ll[i] = loglikhd(PPP[i], XXX[i])
    end
end

function block_setup_x0_priors(data, block_layout, sub_ws_diff)
    #println(block_layout)
    map(block_layout) do b_l
        bl_i, bl_i_range = b_l
        #println(bl_i, bl_i_range)
        i1 = bl_i_range[1]
        pr = (
            i1 == 1 ?
            data.recordings[bl_i].x0_prior :
            KnownStartingPt(sub_ws_diff.XX[bl_i][i1].x[1])
        )
        pr
    end
end

function eMCMC.create_workspaces(backend::DiffusionMCMCBackend, mcmc::MCMC)
    map(1:length(mcmc.updates)) do i
        create_workspace(
            backend,
            mcmc.updates[i],
            mcmc.workspace,
            mcmc.schedule.extra_info.blocking_layout[i],
            mcmc.schedule.num_mcmc_steps
        )
    end
end

function eMCMC.update_workspaces!(
        local_updt::MCMCDiffusionImputation,
        global_ws::eMCMC.GlobalWorkspace,
        local_ws::eMCMC.LocalWorkspace,
        step,
        prev_ws,
    )
    for i in 1:length(step.same_layout)
        if !step.same_layout[i]
            change_laws_for_blocking!(local_ws, i) # TODO
            recompute_loglikhd!(local_ws, i)
        else
            prev_ws===nothing || ( ll(local_ws)[i] = ll(prev_ws)[i] )
        end
    end
    local_ws, local_updt
end

accepted(ws::DiffusionLocalWorkspace, i::Int) = ws.acceptance_history[i]
set_accepted!(ws::DiffusionLocalWorkspace, i::Int, accepted, j::Int=1) = (ws.acceptance_history[i][j] = accepted)
ll(ws::DiffusionLocalWorkspace) = ws.sub_ws_diff.ll
ll°(ws::DiffusionLocalWorkspace) = ws.sub_ws_diff°.ll
ll(ws::DiffusionLocalWorkspace, i::Int) = ws.sub_ws_diff.ll_history[i]
ll°(ws::DiffusionLocalWorkspace, i::Int) = ws.sub_ws_diff°.ll_history[i]
