using HSVIforOSPOSGs
using Test

@testset "HSVIforOSPOSGs.jl" begin
    include("load.jl")
    include("checks.jl")
    include("save.jl")
    include("logs.jl")
    include("rewards.jl")
    include("solve.jl")
end
