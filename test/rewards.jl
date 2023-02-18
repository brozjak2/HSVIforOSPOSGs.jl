@testset "methods" begin
    osposg = OSPOSG("games/rewards.osposg")

    @test HSVIforOSPOSGs.LB_min(osposg) ≈ 0.0
    @test HSVIforOSPOSGs.UB_max(osposg) ≈ 20.0
    @test HSVIforOSPOSGs.lipschitz_delta(osposg) ≈ 10.0
end
