# [Conjugate updates](@id conjugate_updates_explanation)
****
Sometimes a subset of parameters conditioned on the realization of the path admits a representation that is given by one of the standard statistical distributions. This means that no Metropolis–Hastings step needs to be employed for the step of updating such parameters, and instead, they can be simply sampled exactly from their conditional distribution. For the case of diffusions two such cases are most frequently encountered in practice.
1. When a **prior** for the **parameters in the drift** is set to be (multivariate) Gaussian, then the distribution of those parameters conditionally on the path $X$ can **sometimes** be identified as a Gaussian distribution as well. We will refer to this case as **conjugate Gaussian updates**.
2. When a **prior** for the **parameters in the volatility coefficient** is set to be (multivariate) Gaussian, then the distribution of those parameters conditionally on the path $X$ can **sometimes** be identified as an Inverse-Wishart distribution. We will refer to this case as **conjugate Gaussian-Inverse-Wishart updates** (TODO this section needs to be written up and coded up).

!!! note
    The word **sometimes** is important. It will not be possible to always find conjugate Gaussian updates for any parameters in the drift, nor will it be possible for any parameters in the volatility coefficients we please. They must additionally satisfy certain conditions that we describe below.

!!! warning
    Conjugate updates may dramatically improve mixing of the Markov chains; however, for users inexperienced in stochastic analysis this might be a risky option that may potentially yield incorrect results. The reason is that your stochastic differential equation must obey certain principles (which for certain examples can become quite delicate) and if those principles are not satisfied your code might still run and return some answer, but the answer will not be correct. Consequently, we urge the user to exert caution when employing conjugate updates. Below, we try to present the relevant background for employing conjugate updates.

## Conjugate Gaussian updates
----

### Statement of the result

Suppose we have a stochastic differential equation of the form:
```math
\begin{align*}
    \dd X_t^{[1:m]} &= β(t,X_t)\dd t,\\
    \dd X_t^{[(m+1):d]} &= \left[φ(t,X_t)θ + ϕ(t,X_t)\right]\dd t + σ(t,X_t) \dd W_t,
\end{align*}
```
where the notation $\cdot^{[a:b]}$ is used to represent subvectors. If $m=0$, then the first row disappears. $W$ denotes a $(d-m)$-dimensional Wiener process, $θ\in\RR^p$, $φ:\RR\times\RR^d→\RR^{(d-m)}\times\RR^{p}$ and $ϕ:\RR\times\RR^d→\RR^{(d-m)}$. Suppose further that $θ$ comes equipped with a multivariate Gaussian prior:

```math
π(θ)∼N(μ,Σ)
```

Then, **formally**, the distribution of the parameters $θ$ conditionally on the realization of the path $X$ is Gaussian:

```math
π(θ|X)∼N\left(
    \left(\mathcal{W} + Σ^{-1}\right)^{-1}\left[λ + Σ^{-1}μ\right],
    \left(\mathcal{W} + Σ^{-1}\right)^{-1}
\right),
```
where

```math
λ:=\int_0^T \left[
        φ^T\left(σσ^T\right)^{-1}
    \right](t,X_t) \dd X_t^{[(m+1):d]}
    - \int_0^T \left[
        φ^T\left(σσ^T\right)^{-1}ϕ
    \right](t,X_t) \dd t
```
and
```math
W:=\int_0^T \left[
    φ^T\left(σσ^T\right)^{-1}φ
\right](t,X_t) \dd t
```

!!! note
    The result is only formal and not rigorous. For it to be rigorous one needs to make sure that the target and the dominating laws are well-defined and that the Radon–Nikodym derivative between the two exists (and is given by the implicitly assumed expression).

### Coding conjugate Gaussian updates up
The major bulk of the computations is already pre-coded in the package; however, the user needs to provide the functions $φ$ and $ϕ$ as well as indicate which coordinates are non-degenerate (in the above the first $m$ coordinates are degenerate; however, in general they can appear at any position and be interlaced with non-degenerate ones, the user needs to simply indicate in a list which ones have non-zero volatility).

To help with this process we can use a macro `@conjugate_gaussian` from the package [DiffusionDefinition.jl](https://juliadiffusionbayes.github.io/DiffusionDefinition.jl/dev/) (see [this page] for the documentation of this macro). To specify which coordinates are non-degenerate pass `:nonhypo --> collect((m+1):d)` say to `@conjugate_gaussian`. Function `phi` should be specified parameter after parameter by passing pairs of `parameter-name`—$φ^{[i,:]}(t,x)$, where `parameter-name` corresponds to the ith parameter and $φ^{[i,:]}(t,x)$ is simply an ith row of $φ(t,x)$. Function can be defined directly by overloading

```@docs
DiffusionMCMC.φᶜ
```
In here, $θ^c$ denotes a vector of parameters other than $θ$ (i.e. parameters assumed to be constant when updating $θ$), plus a single entry with 1, that represents terms independent from any parameters whatsoever.

Alternatively, if the remaining parameters $θ^c$ also appear in a linear fashion in the drift, i.e. we have:

```math
ϕ(t,x,θ^c) = φ^c(t,x)θ^c
```
for some function $φ^c$ independent from $θ^c$, then we may specify the rows of function $φ^c$ as if they were the rows of function $φ$ and as if the parameters $θ^c$ were being updated alongside $θ$ instead of specifying $ϕ$ directly. Then, we can simply indicate which parameters out of $(θ, θ^c)$ are actually being updated with conjugate updates.

Function $(σσ^T)^{-1}$ will be computed automatically based on the indicator of a subset of non-degenerate coordinates and based on the global volatility coefficient defining a diffusion process. However, we can speed up these computations by passing an expression for $(σσ^T)^{-1}$ directly by passing `:hypo_a_inv --> f(t,x,P)`.

### Example
Consider a FitzHugh—Nagumo model in a slightly reparameterized form from [here](https://juliadiffusionbayes.github.io/DiffusionDefinition.jl/dev/predefined_processes/fitzhugh_nagumo/#Conjugate-1). We can define conjugate updates for it as follows:
```julia
@conjugate_gaussian FitzHughNagumoConjug begin
    :intercept --> (-x[2],)
    :ϵ --> (x[1]-x[1]^3+(1-3*x[1]^2)*x[2],)
    :s --> (one(x[1]),)
    :γ --> (-x[1],)
    :β --> (-one(x[1]),)
    :hypo_a_inv --> 1.0/P.σ^2
    :nonhypo --> 2:2
end
```
