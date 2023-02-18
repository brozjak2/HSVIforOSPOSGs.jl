@testset "solve" begin
    with_logger(ConsoleLogger(stdout, Logging.Error)) do
        osposg = OSPOSG("../games/pursuit-evasion/peg03.osposg")
        hsvi = HSVI()
        epsilon = 1.0
        time_limit = 60.0

        recorder = solve(osposg, hsvi, epsilon, time_limit)

        @test HSVIforOSPOSGs.width(osposg, hsvi) < epsilon
        @test HSVIforOSPOSGs.LB_value(osposg, hsvi) < 83.443625 < HSVIforOSPOSGs.UB_value(osposg, hsvi)
    end
end
