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
        new{T,typeof(P),typeof(WW),typeof(Wnr),typeof(XX)}(P, WW, Wnr, XX)
    end
end

struct DiffusionGlobalWorkspace{T,SW,SWD,TD} <: eMCMC.GlobalWorkspace{T}
    sub_ws::SW
    sub_ws_diff::SWD
    sub_ws_diff°::SWD
    data::TD
    stats::GenericChainStats{T}
end

function init_global_workspace(
        ::DiffusionMCMCBackend,
        schedule::eMCMC.MCMCSchedule,
        updates::Vector{<:eMCMC.MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T
    dkwargs = Dict(kwargs)
    M, NU = schedule.num_mcmc_steps, length(updates)
    #TW = get(dkwargs, :TW, Float64)
    #TX = get(dkwargs, :TX, Float64)

    sub_ws = eMCMC.StandardGlobalSubworkspace(M, NU, data, θinit)

    sub_ws_diff = DiffusionGlobalSubworkspace{T}(
        extract_aux_laws(updates, data),
        data,
        DOS.setup_time_grids(
            data,
            get(dkwargs, :dt, 0.01),
            get(dkwargs, :τ, _DEFAULT_TIME_TRANSFORM),
            get(dkwargs, :Ttime, Float64),
            get(dkwargs, :tt, nothing),
        ),
        package(get(dkwargs, :guid_prop_args, tuple()), data), # from DiffObservScheme
    )

    DiffusionGlobalWorkspace{T,typeof(sub_ws),typeof(sub_ws_diff),typeof(data)}(
        sub_ws,
        sub_ws_diff,
        deepcopy(sub_ws_diff),
        data,
        GenericChainStats(θinit, NU, M),
    )
end

function extract_aux_laws(updates::Vector{<:eMCMC.MCMCUpdate}, data)
    imps = filter(x->typeof(x)<:PathImputation, updates)
    N = length(imps)
    @assert N > 0
    aux_laws = map(x->DOS.package(x.aux_laws, data), imps)
    @assert N == 1 || all([aux_laws[1]==aux_laws[i] for i=2:N])
    aux_laws[1]
end


#===============================================================================
                        Local workspace for diffusions
===============================================================================#

struct DiffusionLocalSubworkspace{T,TP,TW,TWn,TX,Tz0} <: eMCMC.LocalWorkspace{T}
    ll::Vector{Float64}
    ll_history::Vector{Vector{Float64}}
    ll_correct::Vector{Bool}
    P::TP
    WW::TW
    Wnr::TWn
    XX::TX
    z0s::Tz0

    function DiffusionLocalSubworkspace{T}(
            ll, ll_hist, ll_correct, P::TP, WW::TW, Wnr::TWn, XX::TX, z0s::Tz0
        ) where {T,TP,TW,TWn,TX,Tz0}
        new{T,TP,TW,TWn,TX,Tz0}(ll, ll_hist, ll_correct, P, WW, Wnr, XX, z0s)
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
        zeros(Bool, M),
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
end

function create_workspace(
        backend::DiffusionMCMCBackend,
        mcmcupdate,
        global_ws,
        block_layout,
        num_mcmc_steps,
    )
    state = (
        typeof(mcmcupdate) <: MCMCParamUpdate ?
        global_ws.sub_ws.state[mcmcupdate.loc2glob_idx] :
        undef
    )
    sub_ws = StandardLocalSubworkspace(state, num_mcmc_steps)
    xx0_priors = block_setup_x0_priors(global_ws.data, block_layout) #TODO
    sub_ws_diff = DiffusionLocalSubworkspace{T}(
        global_ws.sub_ws_diff, block_layout, num_mcmc_steps
    )
    sub_ws_diff° = DiffusionLocalSubworkspace{T}(
        global_ws.sub_ws_diff°, block_layout, num_mcmc_steps
    )
    new{T,typeof(sub_ws),typeof(sub_ws_diff), typeof(xx0_priors)}(
        sub_ws, deepcopy(sub_ws), sub_ws_diff, sub_ws_diff°, xx0_priors
    )
end

function create_workspaces(backend::DiffusionMCMCBackend, mcmc::MCMC, global_ws)
    current_block_flag = check_for_blocking(mcmc.updates_and_decorators)

    current_block_layout = init_block_layout(global_ws.data, current_block_flag)
    map(mcmc.updates_and_decorators) do updt_or_decor
        if isdecorator(updt_or_decor)
            update!(current_block_layout, updt_or_decor)
            current_block_flag = updt_or_decor
        end
        create_workspace(v, updt, global_ws, current_block_layout)
    end
end

function check_for_blocking(updates_and_decorators)
    decorators = get_decorators(updates_and_decorators)
    length(decorators)==0 && return nothing

    tdecorators = typeof.(decorators)
    # currently only blocking implemented
    @assert all( map(d->(d<:BlockingType), tdecorators) )

    # only one type of blocking may be used for now
    @assert length(unique(tdecorators)) == 1

    # WLOG allow only an even number of block changes to prevent ugly code!
    @assert length(tdecorators) % 2 == 0

    # start from the blocking layout that the update loop finishes on
    decorators[end]
end
