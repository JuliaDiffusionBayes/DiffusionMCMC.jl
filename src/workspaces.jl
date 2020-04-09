
struct GlobalDiffusionWorkspace{T,Ttime,TW,TX} <: GlobalWorkspace{T}
    state::Vector{T}
    state_history::Vector{Vector{Vector{T}}}
    state_proposal_history::Vector{Vector{Vector{T}}}
    acceptance_history::Vector{Vector{Bool}}
    data::TD
    P::Vector{Vector{GuidProp{TGP}}}
    P°::Vector{Vector{GuidProp{TGP}}}
    WW::Vector{Vector{Trajectory{Ttime,TW}}}
    WW°::Vector{Vector{Trajectory{Ttime,TW}}}
    XX::Vector{Vector{Trajectory{Ttime,TX}}}
    XX°::Vector{Vector{Trajectory{Ttime,TX}}}
    xx0::Vector{TX}
    xx0°::Vector{TX}
end

function init_global_workspace(
        ::Val{:DiffusionProblem},
        schedule::MCMCSchedule,
        updates::Vector{<:MCMCUpdate},
        data,
        θinit::Vector{T};
        kwargs...
    ) where T
    dkwargs = Dict(kwargs)

    Ttime = get(dkwargs, :Ttime, Float64)
    TW = get(dkwargs, :TW, Float64)
    TX = get(dkwargs, :TX, Float64)

    tts = setup_time_grids(
        data,
        get(dkwargs, :τ, _DEFAULT_TIME_TRANSFORM),
        get(dkwargs, :dt, 0.01),
        Ttime,
    )

    GlobalDiffusionWorkspace(
        θinit,
        Vector{Vector{Vector{T}}}(undef, schedule.num_mcmc_steps),
        Vector{Vector{Vector{T}}}(undef, schedule.num_mcmc_steps),
        Vector{Vecotr{Bool}}(undef, schedule.num_mcmc_steps),
        data,
        setup_guid_prop_for_data(data, dkwargs[:auxiliary_laws]),
        setup_guid_prop_for_data(data, dkwargs[:auxiliary_laws]),
        setup_trajectory(data, tts, TW),
        setup_trajectory(data, tts, TW),
        setup_trajectory(data, tts, TX),
        setup_trajectory(data, tts, TX),
        setup_starting_points(data, TX),
        setup_starting_points(data, TX),
    )
end


const _VIEW{T} = Array{SubArray{T,1,Array{T,1},Tuple{UnitRange{Int64}},true},1} where T

struct WorkspaceDiffusion{T,Ttime,TW,TX,K} <: LocalWorkspace{T}
    state::Vector{T}
    state°::Vector{T}
    local_to_global_idx::Vector{K<:Integer}
    state_history::Vector{Vector{Vector{T}}}
    state_proposal_history::Vector{Vector{Vector{T}}}
    data::TD
    P::_VIEW{GuidProp{TGP}}
    P°::_VIEW{GuidProp{TGP}}
    WW::_VIEW{Trajectory{Ttime,TW}}
    WW°::_VIEW{Trajectory{Ttime,TW}}
    XX::_VIEW{Trajectory{Ttime,TX}}
    XX°::_VIEW{Trajectory{Ttime,TX}}
    xx0::Vector{AbstractArray{TX}}
    xx0°::Vector{AbstractArray{TX}}
    xx0_priors::Vector{}
    adpt_ws::AdaptationWorkspace... #TODO
end

function create_workspace(
        ::Val{:DiffusionProblem},
        mcmcupdate,
        global_ws,
        block_layout
    )
    P = block_view(global_ws.P, block_layout)
    P° = block_view(global_ws.P°, block_layout)
    WW = block_view(global_ws.WW, block_layout)
    WW° = block_view(global_ws.WW°, block_layout)
    XX = block_view(global_ws.XX, block_layout)
    XX° = block_view(global_ws.XX°, block_layout)
    xx0 = block_view_start_pt(global_ws.xx0, block_layout) # possibly remove
    xx0° = block_view_start_pt(global_ws.xx0, block_layout) # possibly remove
    xx0_priors = block_setup_x0_priors(global_ws.data, block_layout)
end

function block_view(v, layout)
    [ view(v[bl_i], bl_i_range) for (bl_i, bl_i_range) in layout ]
end

function block_view(v, layout::Nothing)
    [ view(v[i], :) for i in 1:length(v) ]
end

block_view_start_pt(v, layout::Nothing) = view(v, :)

function block_view_start_pt(v, layout)
    [
        ( 1 in bl_i_range ? view(v, bl_i) : deepcopy(v[bl_i]) )
        for (bl_i, bl_i_range) in layout
    ]
end

function block_setup_priors(data, layout)
    # implement this in DiffObservScheme.jl
end

function create_workspaces(v::Val{:DiffusionProblem}, mcmc::MCMC, global_ws)
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
