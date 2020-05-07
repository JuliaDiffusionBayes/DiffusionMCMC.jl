#===============================================================================
                        Global workspace for diffusions
===============================================================================#

struct DiffusionGlobalSubworkspace{T,TGP,TW,TWn,TX} <: eMCMC.GlobalWorkspace{T}
    P::TGP
    WW::TW
    Wnr::TWn
    XX::TX

    function DiffusionGlobalSubworkspace{T}(
            aux_laws,
            data::AllObservations,
            tts,
            args,
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
        init!(P, WW, Wnr, XX, data)
        new{T,typeof(P),typeof(WW),typeof(Wnr),typeof(XX)}(P, WW, Wnr, XX)
    end
end

function init!(P, WW, Wnr, XX, data)
    num_recordings = length(P)

    # this loop should be entirely parallelizable, as recordings
    # are---by design---conditionally independent
    for i in 1:num_recordings
        success = false
        while !success
            success, _ = forward_guide!(
                WW[i], XX[i], Wnr[i], P[i], rand(data.recordings[i].x0_prior)
            )
        end
    end
end

struct DiffusionGlobalWorkspace{T,SW,SWD,TD} <: eMCMC.GlobalWorkspace{T}
    sub_ws::SW
    sub_ws_diff::SWD
    sub_ws_diff°::SWD
    data::TD
    stats::GenericChainStats{T}
end

function eMCMC.init_global_workspace(
        ::DiffusionMCMCBackend,
        num_mcmc_steps,
        updates::Vector{<:eMCMC.MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T
    dkwargs = Dict(kwargs)
    M, NU = num_mcmc_steps, length(updates)
    #TW = get(dkwargs, :TW, Float64)
    #TX = get(dkwargs, :TX, Float64)

    sub_ws = eMCMC.StandardGlobalSubworkspace(M, NU, data, θinit)

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

    DiffusionGlobalWorkspace{T,typeof(sub_ws),typeof(sub_ws_diff),typeof(data)}(
        sub_ws,
        sub_ws_diff,
        deepcopy(sub_ws_diff),
        data,
        GenericChainStats(θinit, NU, M),
    )
end

function init_XX_hist(XX, step)

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

struct DiffusionLocalWorkspace{T,SW,SWD,Tpr} <: eMCMC.LocalWorkspace{T}
    sub_ws::SW
    sub_ws°::SW
    sub_ws_diff::SWD
    sub_ws_diff°::SWD
    xx0_priors::Tpr
    acceptance_history::Vector{Vector{Bool}}
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
    init_ll!(sub_ws_diff, global_ws)

    sub_ws_diff° = DiffusionLocalSubworkspace{T}(
        global_ws.sub_ws_diff°, block_layout, num_mcmc_steps
    )
    a_h = [similar(v, Bool) for v in sub_ws_diff.ll_history]

    init_update!(mcmcupdate, block_layout)

    DiffusionLocalWorkspace{T,typeof(sub_ws),typeof(sub_ws_diff), typeof(xx0_priors)}(
        sub_ws, deepcopy(sub_ws), sub_ws_diff, sub_ws_diff°, xx0_priors, a_h
    )
end

function init_ll!(sub_ws_diff, global_ws)
    XX, P = global_ws.sub_ws_diff.XX, global_ws.sub_ws_diff.P
    for i in 1:length(sub_ws_diff.ll)
        num_segm = length(XX)
        sub_ws_diff.ll[i] = 0.0
        for j in 1:length(XX[i])
            sub_ws_diff.ll[i] += loglikhd(XX[i][j], P[i][j])
        end
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
ll(ws::DiffusionLocalWorkspace) = ws.sub_ws_diff.ll
ll°(ws::DiffusionLocalWorkspace) = ws.sub_ws_diff°.ll
ll(ws::DiffusionLocalWorkspace, i::Int) = ws.sub_ws_diff.ll_history[i]
ll°(ws::DiffusionLocalWorkspace, i::Int) = ws.sub_ws_diff°.ll_history[i]
