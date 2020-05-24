# [Inference and smoothing for the FitzHugh–Nagumo model](@id tutorials_start)
**********
> In this tutorial we will do smoothing and parameter inference of the partially observed hypoelliptic version of the FitzHugh–Nagumo model.

## Preparation
-----
```julia
using DiffusionMCMC, ExtensibleMCMC, GuidedProposals, DiffusionDefinition
using ObservationSchemes
const GP = GuidedProposals
const DD = DiffusionDefinition
const OBS = ObservationSchemes
const eMCMC = ExtensibleMCMC

using StaticArrays, Random, Plots
using OrderedCollections

@load_diffusion FitzHughNagumo
@load_diffusion FitzHughNagumoAux

# generate some data
θ = [0.1, -0.8, 1.5, 0.0, 0.3]
P = FitzHughNagumo(θ...)
tt, y1 = 0.0:0.0001:10.0, @SVector [-0.9, -1.0]
X = rand(P, tt, y1)
data = map(
	x->(x[1], x[2][1] + 0.1randn()),
	collect(zip(X.t, X.x))[1:1000:end]
)[2:end]

# let's prepare the data
recording = (
	P = P,
	obs = load_data(
		ObsScheme(
			LinearGsnObs(
				0.0, (@SVector [0.0]);
				L=(@SMatrix [1.0 0.0]), Σ=(@SMatrix [0.01])
			)
		),
		data
	),
	t0 = 0.0,
	x0_prior = KnownStartingPt(y1),
)
observs = AllObservations()
add_recording!(observs, recording)
init_obs, _ = initialize(observs)
```

## Smoothing
-----
```julia
mcmc_params = (
    mcmc = MCMC(
        [
            PathImputation(0.96, FitzHughNagumoAux),
        ];
        backend=DiffusionMCMCBackend(),
    ),
    num_mcmc_steps = Integer(1e3),
    data = init_obs,
    θinit = OrderedDict(
	    :none => -Inf, # no parameters needed for smoothing
	),
)

mcmc_kwargs = (
    path_buffer_size = 10,
	dt = 0.001,
)

# run the MCMC
glob_ws, loc_ws = run!(mcmc_params...; mcmc_kwargs...)
```


## Inference
----
```julia
# let's declare which parameters are not changing
DD.const_parameter_names(::Type{<:FitzHughNagumo}) = (:ϵ, :γ, :β, :σ)
DD.const_parameter_names(::Type{<:FitzHughNagumoAux}) = (:ϵ, :γ, :β, :σ, :t0, :T, :vT, :xT)

# and initialize for those constant parameters
init_obs, _ = initialize(observs)


# and do the inference
mcmc_params = (
    mcmc = MCMC(
        [
            PathImputation(0.96, FitzHughNagumoAux),
            RandomWalkUpdate(UniformRandomWalk([0.2]), [1]),
        ];
        backend=DiffusionMCMCBackend(),
    ),
    num_mcmc_steps = Integer(1e3),
    data = init_obs,
    θinit = OrderedDict(
	    :REC1_s => -0.8, # param 1
	),
    callbacks = eMCMC.Callback[],
)

mcmc_kwargs = (
    path_buffer_size = 10,
	dt = 0.001,
)

# run the MCMC
glob_ws, loc_ws = run!(mcmc_params...; mcmc_kwargs...)
```
