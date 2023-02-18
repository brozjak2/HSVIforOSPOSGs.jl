@testset "checks" begin
    @test_throws HSVIforOSPOSGs.MultiPartitionTransitionException OSPOSG("games/multi_partition_transition.osposg")
    @test_throws HSVIforOSPOSGs.IsNotDistributionException OSPOSG("games/is_not_distribution.osposg")
    @test_throws ArgumentError OSPOSG("games/discount.osposg")
end
