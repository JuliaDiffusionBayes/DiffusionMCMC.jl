# Workspaces
*********
Workspaces are containers on which most of the computations done by the MCMC algorithm are performed. It is not important to know how this work is dealt with internally; however, it is helpful to know what are the entries of the **global** and **local** workspace for diffusions, as the output of the call to `run!` is precisely that:
- the `DiffusionGlobalWorkspace`, as well as
- `DiffusionLocalWorkspace` for each update.

## Global workspaces
-----
Global workspace holds information about the history of the parameters (both accepted and proposed) as well as some standard, already pre-computed statistics about the chain.
```@docs
DiffusionMCMC.DiffusionGlobalSubworkspace
DiffusionMCMC.DiffusionGlobalWorkspace
```

## Local workspaces
----
Local workspaces hold information pertinent to particular updates. In particular it collects such information as acceptance history or a history of log-likelihoods.
```@docs
DiffusionMCMC.DiffusionLocalSubworkspace
DiffusionMCMC.DiffusionLocalWorkspace
```
