using DiffusionMCMC
using ExtensibleMCMC
using Test

const dMCMC = DiffusionMCMC
const eMCMC = ExtensibleMCMC

@testset "schedule.jl" begin
    schedule_params = (
        num_mcmc_iter = 10,
        num_params = 4,
        exclude_params = [(1,3:8), (2,4:2:10), (4,6:10)]
    )
    ∅ = nothing
    t,f = true, false

    schedule = eMCMC.MCMCSchedule(
        schedule_params...;
        start=(
            prev_mcmciter=∅,
            prev_pidx=∅,
            mcmciter=1,
            pidx=1,
            same_layout=(t,t,t)
        ),
        backend=DiffusionMCMCBackend(),
        extra_info=(
            layout_types=(
                (1,1,1),
                (1,2,2),
                (1,1,3),
                (1,2,3),
            ),
            no_blocking=false,
            blocking_layout=:not_needed,
        )
    )

    expected = (
        (∅,∅,1,1,(t,t,t)),  (1,1,1,2,(t,f,f)),  (1,2,1,3,(t,f,f)),  (1,3,1,4,(t,f,t)),
        (1,4,2,1,(t,f,f)),  (2,1,2,2,(t,f,f)),  (2,2,2,3,(t,f,f)),  (2,3,2,4,(t,f,t)),
                            (2,4,3,2,(t,t,f)),  (3,2,3,3,(t,f,f)),  (3,3,3,4,(t,f,t)),
                                                (3,4,4,3,(t,f,t)),  (4,3,4,4,(t,f,t)),
                            (4,4,5,2,(t,t,f)),  (5,2,5,3,(t,f,f)),  (5,3,5,4,(t,f,t)),
                                                (5,4,6,3,(t,f,t)),
                            (6,3,7,2,(t,f,f)),  (7,2,7,3,(t,f,f)),
                                                (7,3,8,3,(t,t,t)),
        (8,3,9,1,(t,t,f)),  (9,1,9,2,(t,f,f)),  (9,2,9,3,(t,f,f)),
        (9,3,10,1,(t,t,f)),                     (10,1,10,3,(t,t,f)),
    )
    for (i,s) in enumerate(schedule)
        @test expected[i] == tuple(s...)
    end
end

@testset "DiffusionMCMC.jl" begin
    # Write your own tests here.
end
