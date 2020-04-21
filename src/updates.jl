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

struct StartingPointsImputation <: MCMCDiffusionImputation end

struct StartingPointsLangevinImputation <: MCMCDiffusionImputation end


#===============================================================================
                            Path imputation
===============================================================================#

function update!(
        update::PathImputation,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
    )
    ws, ws° = local_ws.sub_ws_diff, local_ws.sub_ws_diff°

    num_recordings = length(ws.P)

    # this loop should be entirely parallelizable, as recordings
    # are---by design---conditionally independent
    for i in 1:num_recordings
        _, ws°.ll[i] = forward_guide!(
            ws°.WW[i], ws°.XX[i], ws°.Wnr[i], ws.P[i], ws.xx0[i], ws.WW[i],
            update.ρs[i]
        )
        path_loglikhd!(ws, i) # use cashed values or recompute
        accept_reject!(updt, ws_global, local_ws, step, i)
    end
end

log_transition_density(updt::PathImputation, args...) = 0.0
log_prior(updt::PathImputation, args...) = 0.0

#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶
#               Decide what happens at the Accept/Reject step
#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶

function update!(
        accepted::Bool,
        updt::PathImputation,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
        i=1
    )
    update!(accepted, updt, local_ws, step, i)
    save_path(global_ws.sub_ws, step, i) #TODO
    register_accept_reject!(accepted, global_ws, step, i) # this should update the acceptance history
end

function register_accept_reject!(
        accepted::Bool,
        global_ws::DiffusionGlobalWorkspace,
        step,
        i=1,
    )
    global_ws.sub_ws.acceptance_history[step.mcmciter][step.pidx][i] = accepted
    global_ws.sub_ws.ll_history[step.mcmciter][step.pidx][i] = ws.sub_ws_diff.ll[i]
    global_ws.sub_ws.ll_proposal_history[step.mcmciter][step.pidx][i] = (
        accepted ?
        ws.sub_ws_diff.ll[i] :
        ws.sub_ws_diff°.ll[i]
    )
end

function update!(
        accepted::Bool,
        updt::PathImputation,
        ws::DiffusionLocalWorkspace,
        step,
        i=1,
    )
    accepted && swap_paths!(ws, i)
    register!(updt, accepted)
end

function swap_paths!(ws::DiffusionLocalWorkspace, i)
    swap!(ws.sub_ws_diff°.WW[i], ws.sub_ws_diff.WW[i])
    swap!(ws.sub_ws_diff°.XX[i], ws.sub_ws_diff.XX[i])
    swap!(ws.sub_ws_diff°.ll[i], ws.sub_ws_diff.ll[i])
end

function path_loglikhd!(ws::DiffusionLocalWorkspace, i)
    ws.ll_correct[i] || return

    num_segments = length(ws.WW[i])
    ll_tot = 0.0
    for j in 1:num_segments
        ll_tot += loglikelihood(ws.XX[i][j], ws.P[i][j])
    end
    ws.ll[i] = ll_tot
end

#===============================================================================
                        Starting point imputation
===============================================================================#

function update!(
        update::StartingPointsImputation,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
    )
    ws, ws° = ws.sub_ws_gp, ws.sub_ws_gp°
    num_recordings = length(ws.P)

    # this loop should be entirely parallelizable, as recordings
    # are---by design---conditionally independent
    for i in 1:num_recordings
        z°, y° = proposal_start_pt(ws, i) #TODO sample new sp

        num_interv = length(ws.XX°)
        success, ws°.ll[i] = solve_and_ll!(y°, ws.WW[i], ws°.XX[i], ws.P[i])
        # = ( success ? log_likelihood(ws°.XX, ws.P) : -Inf )
        accept_reject!(updt, ws, ws_global, step, i)
    end
end

log_transition_density(updt::StartingPointsImputation, args...) = 0.0

function log_prior(
        updt::StartingPointsImputation,
        ws::DiffusionLocalWorkspace,
        i,
        ::eMCMC.Proposal
    )
    y° = ws.sub_ws_gp°.XX[i][1].y[1]
    # a little bit of cheating with the second term (not a prior)
    logpdf(ws.x0_prior, y°)# + log_likelihood_obs(ws.sub_ws_gp.P[1], y°) #TODO integrate the latter directly into ll
end

function log_prior(
        updt::StartingPointsImputation,
        ws::DiffusionLocalWorkspace,
        i,
    )
    y = ws.sub_ws_gp.XX[i][1].y[1]
    # a little bit of cheating with the second term (not a prior)
    logpdf(ws.x0_prior, y)# + log_likelihood_obs(ws.sub_ws_gp.P[1], y) #TODO integrate the latter directly into ll
end


#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶
#               Decide what happens at the Accept/Reject step
#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶

function update!(
        accepted::Bool,
        updt::StartingPointsImputation,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
        i=1
    )
    accepted && update!(accepted, updt, local_ws, step, i)
    save_path(ws.sub_ws, step, i)
    register_accept_reject!(global_ws, accepted, step, i) # this should update the acceptance history
end


function update!(
        accepted::Bool,
        updt::StartingPointsImputation,
        ws::DiffusionLocalWorkspace,
        step,
        i=1,
    )
    accepted && swap_starting_pts_and_paths!(ws, i)
    register!(updt, accepted)
end

function swap_starting_pts_and_paths!(ws::DiffusionLocalWorkspace, i)
    swap!(ws.sub_ws_diff°.xx0[i], ws.sub_ws_diff.xx0[i])
    swap!(ws.sub_ws_diff°.XX[i], ws.sub_ws_diff.XX[i])
    swap!(ws.sub_ws_diff°.ll[i], ws.sub_ws_diff.ll[i])
end


#===============================================================================
                            Paramter update
===============================================================================#

# proposal!(updt, ws, global_ws, step)
# set_proposal!(global_ws, ws, updt.loc2glob_idx, step) #TODO figure out this
# compute_ll!(updt, global_ws, ws, step)
# accept_reject!(updt, global_ws, ws, step)
# update!(global_ws.stats, global_ws, step)

function set_proposal!(
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        loc2glob_idx,
        step,
    )
    ws, ws° = ws.sub_ws_diff, ws.sub_ws_diff°
    update_laws!(ws°.P, global_ws, local_ws, updt, step)

    for i in 1:num_recordings
        success, ws°.ll[i] = solve_and_ll!(ws°.XX[i], ws.WW[i], ws.P[i], ws.XX[i][1].y[1])
        success || return # ...
    end
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

function compute_ll!(
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
    )
    global_ws.ll[1] = sum(local_ws.sub_ws_diff.ll)
    global_ws.ll°[1] = sum(local_ws.sub_ws_diff°.ll)
end

function log_prior(
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        ws::DiffusionLocalWorkspace,
        i,
        ::eMCMC.Proposal
    )
    y° = ws.sub_ws_gp°.XX[i][1].y[1]
    # a little bit of cheating with the second term (not a prior)
    logpdf(ws.x0_prior, y°) + logpdf(updt.prior, global_ws.state°)# + log_likelihood_obs(ws.sub_ws_gp.P[1], y°) #TODO integrate the latter directly into ll
end

function log_prior(
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        ws::DiffusionLocalWorkspace,
        i,
    )
    y = ws.sub_ws_gp.XX[i][1].y[1]
    # a little bit of cheating with the second term (not a prior)
    logpdf(ws.x0_prior, y) + logpdf(updt.prior, global_ws.state)# + log_likelihood_obs(ws.sub_ws_gp.P[1], y) #TODO integrate the latter directly into ll
end


#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶
#               Decide what happens at the Accept/Reject step
#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶

function update!(
        accepted::Bool,
        updt::eMCMC.MCMCParamUpdate,
        local_ws::DiffusionLocalWorkspace,
        global_ws::DiffusionGlobalWorkspace,
        step,
        i=1
    )
    accepted && update!(accepted, updt, local_ws, step, i)
    save_path(ws.sub_ws, step, i)
    register_accept_reject!(global_ws, accepted, step, i) # this should update the acceptance history
end


function update!(
        accepted::Bool,
        updt::eMCMC.MCMCParamUpdate,
        ws::DiffusionLocalWorkspace,
        step,
        i=1,
    )
    accepted && swap_laws_and_rest!(ws, i)
    register!(updt, accepted)
end

function swap_laws_and_rest!(ws::DiffusionLocalWorkspace, i)
    swap!(ws.sub_ws_diff°.P[i], ws.sub_ws_diff.P[i])
    swap_starting_pts_and_paths!(ws, i)
end


#===============================================================================
                Starting point imputation via Langevin updates
===============================================================================#

function update!(
        update::StartingPointsLangevinImputation,
        ws::DiffusionLocalWorkspace,
        ws_global::DiffusionGlobalWorkspace,
        step
    )
    nothing
end
