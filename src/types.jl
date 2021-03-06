#===============================================================================

    Abstract types reflecting conceptual inheritence structure that come
    in addition to those defined in StochasticProcessMCMC.jl

===============================================================================#
#NOTE possibly superfluous
"""
    DiffusionSpecificCallback <: Callback

Supertype of all callbacks specific to diffusion processes.
"""
abstract type DiffusionSpecificCallback <: eMCMC.Callback end

"""
    MCMCDiffusionImputation <: MCMCImputation

Supertype of all imputations related to diffusions.
"""
abstract type MCMCDiffusionImputation <: eMCMC.MCMCImputation end

"""
    BlockingType <: MCMCUpdateDecorator

Supertype of all blocking patterns.
"""
abstract type BlockingType <: eMCMC.MCMCUpdateDecorator end

"""
    GenericMCMCBackend

A flag that no specific backend is passed.
"""
struct DiffusionMCMCBackend <: eMCMC.MCMCBackend end
