#===============================================================================
                            Path imputation
===============================================================================#

function eMCMC.update!(
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
            ws°.WW[i], ws°.XX[i], ws°.Wnr[i], ws.P[i], ws.XX[i][1].x[1], ws.WW[i],
            update.ρs[i]
        )
        eMCMC.accept_reject!(
            update, global_ws, local_ws, step, ws.ll[i], ws°.ll[i], i
        )
    end
end

eMCMC.log_transition_density(::Any, updt::PathImputation, args...) = 0.0

eMCMC.log_prior(::Any, updt::PathImputation, args...) = 0.0

#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶
#               Decide what happens at the Accept/Reject step
#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶

function eMCMC.register_accept_reject_results!(
        accepted::Bool,
        updt::PathImputation,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
        i=1
    )
    accepted && swap_paths!(local_ws, i)
    save_path(global_ws.sub_ws_diff.XX, step, i) #TODO
    eMCMC.register_accept_reject_results!(accepted, updt, local_ws, step, i)
end

_TEMP_SAVED_PATHS = []

function save_path(XX, step, i)
    step.mcmciter % 100 == 0 && step.pidx == 1 && push!(_TEMP_SAVED_PATHS, deepcopy(XX))
end

function eMCMC.register_accept_reject_results!(
        _accepted::Bool,
        updt,
        ws::DiffusionLocalWorkspace,
        step,
        i=1,
    )
    register!(updt, _accepted) #TODO
    accepted(ws, step.mcmciter)[i] = _accepted

    # register proposal
    ll°(ws, step.mcmciter)[i] = ll°(ws)[i]

    # update local ll with accepted
    _accepted && ( ll(ws)[i] = ll°(ws)[i] )

    # register accepted
    ll(ws, step.mcmciter)[i] = ll(ws)[i]
end

function register!(updt, accepted::Bool)
    nothing #TODO implement
end

function swap!(A, B)
    for i in eachindex(A, B)
        A[i], B[i] = B[i], A[i]
    end
end

function swap_paths!(ws::DiffusionLocalWorkspace, i)
    s°, s = ws.sub_ws_diff°, ws.sub_ws_diff
    swap!(s°.WW[i], s.WW[i])
    swap!(s°.XX[i], s.XX[i])
end

function recompute_loglikhd!(ws::DiffusionLocalWorkspace, i) #TODO deprecate
    num_segments = length(ws.XX[i])
    ll_tot = 0.0
    for j in 1:num_segments
        ll_tot += loglikhd(ws.XX[i][j], ws.P[i][j])
    end
    ll(ws)[i] = ll_tot
end

#===============================================================================
                        Starting point imputation
===============================================================================#

function eMCMC.update!(
        update::StartingPointsImputation,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
    )
    ws, ws° = ws.sub_ws_diff, ws.sub_ws_diff°
    num_recordings = length(ws.P)

    # this loop should be entirely parallelizable, as recordings
    # are---by design---conditionally independent
    for i in 1:num_recordings
        z°, y° = proposal_start_pt(ws, i) #TODO sample new sp
        _, ws°.ll[i] = solve_and_ll!(y°, ws.WW[i], ws°.XX[i], ws.P[i])
        accept_reject!(
            update, global_ws, local_ws, step, ws.ll[i], ws°.ll[i], i
        )
    end
end

log_transition_density(::Any, updt::StartingPointsImputation, args...) = 0.0

function log_prior(
        ::eMCMC.Proposal,
        updt::StartingPointsImputation,
        ws::DiffusionLocalWorkspace,
        i,

    )
    y° = ws.sub_ws_diff°.XX[i][1].y[1]
    # a little bit of cheating with the second term (not a prior)
    logpdf(ws.x0_prior, y°)# + log_likelihood_obs(ws.sub_ws_gp.P[1], y°) #TODO integrate the latter directly into ll
end

function log_prior(
        ::eMCMC.Previous,
        updt::StartingPointsImputation,
        ws::DiffusionLocalWorkspace,
        i,
    )
    y = ws.sub_ws_diff.XX[i][1].y[1]
    # a little bit of cheating with the second term (not a prior)
    logpdf(ws.x0_prior, y)# + log_likelihood_obs(ws.sub_ws_gp.P[1], y) #TODO integrate the latter directly into ll
end

#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶
#               Decide what happens at the Accept/Reject step
#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶

function eMCMC.register_accept_reject_results!(
        accepted::Bool,
        updt::StartingPointsImputation,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
        i=1
    )
    accepted && swap_starting_pts_and_paths!(ws, i)
    eMCMC.register_accept_reject_results!(accepted, updt, local_ws, step, i)
end

function swap_starting_pts_and_paths!(ws::DiffusionLocalWorkspace, i)
    s°, s = ws.sub_ws_diff°, ws.sub_ws_diff
    swap!(s°.z0s[i], s.z0s[i])
    swap!(s°.XX[i], s.XX[i])
end

#===============================================================================
                            Parameter update
===============================================================================#

# compute_ll!(updt, global_ws, ws, step)
# accept_reject!(updt, global_ws, ws, step)
# eMCMC.update!(global_ws.stats, global_ws, step)

function set_proposal!(
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
    )
    ws, ws° = global_ws.sub_ws_diff, global_ws.sub_ws_diff°
    num_recordings = length(ws.WW)

    for i in 1:num_recordings
        change_type = which_param_changed(updt, ws°.P[i], state°(local_ws)) #TODO

        change_type == :none && continue

        set_parameters!(ws°.P[i], coords(updt), state°(local_ws))
        (
            !step.same_layout[i] || change_type == :extensive
        ) && backward_filter(ws°.P[i])

        success, ws°.ll[i] = solve_and_ll!(
            ws°.XX[i], ws.WW[i], ws.P[i], ws.XX[i][1].y[1]
        )
        success || return # ...
    end
end

function eMCMC.set_parameters!(
        ::eMCMC.Proposal,
        updt::RandomWalkUpdate,
        global_ws::DiffusionGlobalWorkspace,
        ws::DiffusionLocalWorkspace,
    )
    OBS.set_parameters!(
        global_ws.sub_ws_diff°.P,
        global_ws.data,
        eMCMC.coords(updt),
        eMCMC.invcoords(updt),
        state°(ws)
    )
end

function eMCMC.set_parameters!(
        ::eMCMC.Previous,
        updt::RandomWalkUpdate,
        global_ws::DiffusionGlobalWorkspace,
        ws::DiffusionLocalWorkspace,
    )
    OBS.set_parameters!(
        global_ws.sub_ws_diff.P,
        global_ws.data,
        eMCMC.coords(updt),
        eMCMC.invcoords(updt),
        eMCMC.subidx(state(global_ws), updt)
    )
end

#=
function eMCMC.update!()
    num_recordings = length(ws.P)
    for i in 1:num_recordings
        num_intv = length(ws.P[i])
        success, ll° = forward_guide!(ws, update, i)
        ll = path_loglikhd(ws, ws_cached, i) # use cashed values or recompute
        accept_reject!(updt, ws, ws_global, i, ll, ll°)
    end
end
=#

function eMCMC.compute_ll!(
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        local_ws::DiffusionLocalWorkspace,
        step,
    )
    nothing
end

function log_prior(
        ::eMCMC.Proposal,
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        ws::DiffusionLocalWorkspace,
        i,
    )
    y° = ws.sub_ws_gp°.XX[i][1].y[1]
    # possibly get rid of the first one
    logpdf(ws.x0_prior, y°) + logpdf(updt.prior, state°(local_ws))# + log_likelihood_obs(ws.sub_ws_gp.P[1], y°) #TODO integrate the latter directly into ll
end

function log_prior(
        ::eMCMC.Previous,
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        ws::DiffusionLocalWorkspace,
        i,
    )
    y = ws.sub_ws_gp.XX[i][1].y[1]
    # possibly get rid of the first one
    logpdf(ws.x0_prior, y) + logpdf(updt.prior, state(local_ws))# + log_likelihood_obs(ws.sub_ws_gp.P[1], y) #TODO integrate the latter directly into ll
end


#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶
#               Decide what happens at the Accept/Reject step
#̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶̶

function eMCMC.register_accept_reject_results!(
        accepted::Bool,
        updt::eMCMC.MCMCParamUpdate,
        local_ws::DiffusionLocalWorkspace,
        global_ws::DiffusionGlobalWorkspace,
        step,
        i=1
    )
    accepted && swap_laws_and_paths!(local_ws, i)
    eMCMC.register_accept_reject_results!(accepted, updt, local_ws, step, i)
    eMCMC.set_chain_param!(accepted, updt, global_ws, local_ws)
end

function swap_laws_and_rest!(ws::DiffusionLocalWorkspace, i)
    s°, s = ws.sub_ws_diff°, ws.sub_ws_diff
    swap!(s°.P[i], s.P[i])
    swap!(s°.XX[i], s.XX[i])
end


#===============================================================================
                Starting point imputation via Langevin updates
===============================================================================#

function eMCMC.update!(
        update::StartingPointsLangevinImputation,
        ws::DiffusionLocalWorkspace,
        ws_global::DiffusionGlobalWorkspace,
        step
    )
    nothing
end




#===============================================================================
                            Temporary measures
===============================================================================#
# temporary measure, make sure online stats work for Diffusions before removing
function eMCMC.update!(
        updt::eMCMC.MCMCParamUpdate,
        global_ws::DiffusionGlobalWorkspace,
        ws::DiffusionLocalWorkspace,
        step,
    )
    eMCMC.proposal!(updt, global_ws, ws, step) # specific to updt, see `updates.jl`
    eMCMC.set_proposal!(updt, global_ws, ws, step)
    eMCMC.compute_ll!(updt, global_ws, ws, step)
    eMCMC.accept_reject!(updt, global_ws, ws, step)
    #eMCMC.update_stats!(global_ws, ws, step)
end
