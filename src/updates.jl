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
        global_ws::GlobalDiffusionWorkspace,
        step,
        ws_cached=nothing
    )
    num_recordings = length(ws.P)

    # this loop should be entirely parallelizable, as recordings
    # are---by design---conditionally independent
    for i in 1:num_recordings
        num_intv = length(ws.P[i])
        success, ll° = forward_guide!(ws, update, i)
        ll = path_loglikhd(ws, ws_cached, i) # use cashed values or recompute
        accept_reject!(updt, ws, ws_global, i, ll, ll°)
    end
end

function update!(::Val{:accepted}, updt::PathUpdate, ws::WorkspaceDiffusion, i, ll)
    swap!(ws.WW°[i], ws.WW[i])
    swap!(ws.XX°[i], ws.XX[i])
    ws.ll[i] = ll
    register!(updt, true)
end

function update!(::Val{:rejected}, updt::PathUpdate, ws::WorkspaceDiffusion, i, ll)
    ws.ll[i] = ll
    register!(updt, false)
end

function update!(::Val{:accepted}, updt::PathUpdate, ws::GlobalDiffusionWorkspace, i, ll)
    register_accept_reject!(ws, true, i) # this should update the acceptance history
end

function update!(::Val{:rejected}, updt::PathUpdate, ws::GlobalDiffusionWorkspace, i, ll)
    register_accept_reject!(ws, false, i) # this should update the acceptance history
end

function forward_guide!(ws::WorkspaceDifusion, update::PathUpdate, i)
    num_segments = length(ws.WW[i])
    ll°_tot = 0.0
    for j in 1:num_segments
        success, ll° = forward_guide!(ws, update, i, j)
        !success && return false, -Inf
        ll°_tot += ll°
    end
    true, ll°_tot
end

function forward_guide!(ws::WorkspaceDifusion, update::PathUpdate, i, j)
    rand!(ws.WW°[i][j], ws.Wnr[i][j]) #TODO add Wnr
    crank_nicolson!(ws.WW°[i][j].y, ws.WW[i][j].y, ws.ρ[i][j]) #TODO add ρ
    solve_and_ll!(ws.XX°[i][j], ws.WW°[i][j], ws.xx0[i][j], ws.P[i][j]) #TODO implement in GuidedProposals.jl
end

function crank_nicolson!(y°::Vector, y, ρ) # For immutable types
    λ = sqrt(1-ρ^2)
    for i in 1:length(y)
        y°[i] = λ*y°[i] + ρ*y[i]
    end
end

function crank_nicolson!(y°::Vector{T}, y, ρ) where {T<:Mutable} #NOTE GPUs will need to be treated separately
    λ = sqrt(1-ρ^2)
    for i in 1:length(y)
        mul!(y°[i], y[i], true, ρ, λ)
    end
end

function path_loglikhd(ws::WorkspaceDifusion, ws_cached::Nothing, i)
    num_segments = length(ws.WW[i])
    ll_tot = 0.0
    for j in 1:num_segments
        ll_tot += llikelihood(ws.XX[i][j], ws.P[i][j])
    end
    ll_tot
end

function path_loglikhd(ws::WorkspaceDifusion, cached_ws::WorkspaceDifusion, i)
    cached_ws.ll[i]
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

function setup_proposal!(ws::WorkspaceDiffusion, global_ws::GlobalDiffusionWorkspace, step)
    prior_log_pdf = logpdf(updt.priors, ws.state°)
    (prior_log_pdf === -Inf) && return ...

    update_laws!(ws.P°, ws, global_ws, updt, step)

    ll°_tot = 0.0
    for i in 1:num_recordings
        success, ll° = recompute_paths!(i)
        success || return ...
        ll°_tot += ll°
    end
    ll°
end

function recompute_paths!()
    num_segments = length(WW)
    y = XX[1].y[1]
    z° = inv_start_pt(y, x0_prior, P[1])
    success = find_path_from_wiener!(XXᵒ, y, WW, P, 1:num_segments)
    ll° = (
        success ?
        (
            logpdf(x0_prior, y) +
            path_loglikhd() +
            llikelihood_obs()
        ) :
        -Inf
    )
    success, ll°
end


function updates_laws!(P_all, ws, step)
    num_recordings = length(P_all)
    for i in 1:num_recordings
        P, info = P_all[i], step.aux.rec[i]
        info.law_updates_needed && update_law!(P, ws.state°, updt.loc2glob_idx, global_ws)
        info.run_backward_filter && backward_filter(P)
    end
end

function update_law!(P)
end


function update!()
    num_recordings = length(ws.P)
    for i in 1:num_recordings
        num_intv = length(ws.P[i])
        success, ll° = forward_guide!(ws, update, i)
        ll = path_loglikhd(ws, ws_cached, i) # use cashed values or recompute
        accept_reject!(updt, ws, ws_global, i, ll, ll°)
    end
end
