#===============================================================================

    Abstract types reflecting conceptual inheritence structure that come
    in addition to those defined in StochasticProcessMCMC.jl

===============================================================================#
#NOTE possibly superfluous
"""
    DiffusionSpecificCallback <: Callback

Supertype of all callbacks specific to diffusion processes.
"""
abstract type DiffusionSpecificCallback <: spMCMC.Callback end

#NOTE possibly superfluous
"""
    MCMCDiffusionImputation <: MCMCImputation

Supertype of all imputations related to diffusions.
"""
abstract type MCMCDiffusionImputation <: spMCMC.MCMCImputation end

"""
    BlockingType <: MCMCUpdateDecorator

Supertype of all blocking patterns.
"""
abstract type BlockingType <: spMCMC.MCMCUpdateDecorator end
