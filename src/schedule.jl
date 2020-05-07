function eMCMC.extra_transitions(
        iter::eMCMC.MCMCSchedule{DiffusionMCMCBackend},
        new_state
    )
    lt = iter.extra_info.layout_types
    (
        prev_mcmciter = new_state.prev_mcmciter,
        prev_pidx = new_state.prev_pidx,
        mcmciter = new_state.mcmciter,
        pidx = new_state.pidx,
        same_layout = lt[new_state.prev_pidx] .== lt[new_state.pidx]
    )
end

function eMCMC.extra_schedule_params(
        ws::DiffusionGlobalWorkspace,
        updates_and_decorators;
        kwargs...
    )
    updates_only = filter(u->!isdecorator(u), updates_and_decorators)
    no_blocking, current_block_flag = check_for_blocking(updates_and_decorators)

    extra_mcmcschedule_p = (
        start=(
            prev_mcmciter=nothing,
            prev_pidx=nothing,
            mcmciter=1,
            pidx=1,
            same_layout=Tuple(fill(no_blocking, length(ws.data.recordings)))
        ),
        backend=DiffusionMCMCBackend(),
    )

    bl_layout = []
    layout_types = []
    current_block_layout = init_block_layout(ws.data, current_block_flag)

    #append!(bl_layout, current_block_layout)
    current_layout_type = Tuple(fill(1, length(ws.data.recordings)))
    #println(current_layout_type)
    #append!(layout_types, current_layout_type)

    curated_updt_n_decor = (
        no_blocking ?
        updates_and_decorators :
        updates_and_decorators[2:end]
    )

    map(curated_updt_n_decor) do updt_or_decor
        if isdecorator(updt_or_decor)
            current_block_layout = update_layout( # TODO
                updt_or_decor,
                ws,
                current_block_layout
            )
            current_block_flag = updt_or_decor
            update_layout_type!( # TODO
                updt_or_decor,
                current_layout_type
            )
        else
            append!(bl_layout, [current_block_layout])
            append!(layout_types, [current_layout_type])
        end
    end

    (
        extra_mcmcschedule_p...,
        extra_info=(
            layout_types=Tuple(layout_types),
            no_blocking=no_blocking,
            blocking_layout=Tuple(bl_layout),
        )
    )
end

function check_for_blocking(updates_and_decorators)
    decorators = get_decorators(updates_and_decorators)
    length(decorators)==0 && return true, nothing

    tdecorators = typeof.(decorators)
    # currently only blocking implemented
    @assert all( map(d->(d<:BlockingType), tdecorators) )

    # only one type of blocking may be used for now
    @assert length(unique(tdecorators)) == 1

    # WLOG allow only an even number of block changes to prevent ugly code!
    @assert length(tdecorators) % 2 == 0

    # if there is blocking then must start from it
    @assert typeof(updates_and_decorators) <: BlockingType

    false, decorators[1]
end

function init_block_layout(data, ::Nothing)
    r = data.recordings
    Tuple([(i, 1:length(r[i].obs)) for i in 1:length(r)])
end
