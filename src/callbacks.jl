struct SavingPathCallback <: DiffusionSpecificCallback
end

#NOTE this needs to be coordinated with DiffusionVis.jl
struct PathPlottingCallback <: DiffusionSpecificCallback
end

struct AuxLawAdaptationCallback <: DiffusionSpecificCallback
end
