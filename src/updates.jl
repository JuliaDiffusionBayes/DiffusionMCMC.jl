struct PathUpdate <: MCMCDiffusionImputation
end

struct StartingPointsUpdate <: MCMCDiffusionImputation
end

struct StartingPointsLangevinUpdate <: MCMCDiffusionImputation
end

#function proposal!(updt::PathUpdate, ws, ws_main, step)
#
#end

function update!(
        update::PathUpdate,
        ws::WorkspaceDiffusion,
        ws_global::GlobalDiffusionWorkspace
    )
    num_recordings = length(ws.P)

    # this loop should be entirely parallelizable, as recordings
    # are---by design---conditionally independent
    for i in 1:num_recordings
        num_intv = length(ws.P[i])
        ll° = sample_segments!(i, ws, update)
        ll = ll° > -Inf ?  path_log_likhd(Target(), i, ws) : 0.0 # to avoid wasting resources
        accept_reject!(updt, ws, ws_global, i, ll°-ll)
    end
end

function update!(
        update::StartingPointsUpdate,
        ws::WorkspaceDiffusion,
        prev_ws::WorkspaceDiffusion,
        ws_global::GlobalDiffusionWorkspace,
        step,
    )
    num_recordings = length(ws.P)

    # this loop should be entirely parallelizable, as recordings
    # are---by design---conditionally independent
    for i in 1:num_recordings
        z°, y° = proposal_start_pt(ws, i)

        num_interv = length(ws.XX°)
        success = compute_segments!(y°, ws.WW[i], ws.XX°[i], ws.P[i])
        ll° = (
            success ?
            (
                logpdf(ws.x0_prior, y°) +
                path_log_likhd(OS(), ws.XX°, ws.P, 1:m, ws.fpt) +
                lobslikelihood(ws.P[1], y°)
            ) :
            -Inf
        )
        ll = prev_ws.ll[i]
        accept_reject!(updt, ws, ws_global, i, ll°, ll)
    end
end

function update!(
        update::StartingPointsLangevinUpdate,
        ws::WorkspaceDiffusion,
        ws_global::GlobalDiffusionWorkspace,
        step
    )
    nothing
end
