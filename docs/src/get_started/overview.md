# [Get started](@id get_started)
*******
## Installation
-----------
The package is not registered yet. To install it write:
```julia
] add https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl
```

!!! note
    The package depends on [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl), [ObservationSchemes.jl](https://github.com/JuliaDiffusionBayes/ObservationSchemes.jl), [GuidedProposals.jl](https://github.com/JuliaDiffusionBayes/GuidedProposals.jl) and [ExtensibleMCMC.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl) neither of which is registered yet. Install them in the same way as [DiffusionMCMC.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl).

## Preparation
-----------
To use the MCMC samplers implemented in [DiffusionMCMC.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMC.jl) you should define a target and auxiliary diffusions using [DiffusionDefinition.jl](https://github.com/JuliaDiffusionBayes/DiffusionDefinition.jl) and put your data in the format of `AllObservations` from [ObservationSchemes.jl](https://github.com/JuliaDiffusionBayes/ObservationSchemes.jl), for instance:

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

## The underlying idea
-----------
Just as in [ExtensibleMCMC.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl) you may define a variety of MCMC algorithms by deciding on a sequence of updates that constitute each of its steps. You have a verity of options to choose from:
- for updating the parameters of a diffusion you may choose any step from [ExtensibleMCMC.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl)
- for imputing the path on the other hand you may choose anything from a set of updates implemented in this package.
To define your MCMC algorithm you should simply pass a list of such updates to the `MCMC` object.

## Smoothing
-----------
Parameters are known and fixed, so simply path an imputation flag `PathImputation` to `MCMC`:

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

## Parameter inference
-----------
In addition to imputation step pass a parameter update step to define a suitable Metropolis-within-Gibbs algorithm
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

## Adaptive schemes
-----------

## Blocking
-----------

## Printing Callback
-----------

## Plotting Callback with [DiffusionMCMCPlots.jl](https://github.com/JuliaDiffusionBayes/DiffusionMCMCPlots.jl)
-----------

## [ExtensibleMCMC.jl](https://github.com/JuliaDiffusionBayes/ExtensibleMCMC.jl) functionality
-----------
