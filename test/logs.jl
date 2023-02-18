using Logging
using Printf: @sprintf

@testset "logs" begin
    osposg = OSPOSG("../games/pursuit-evasion/peg03.osposg")
    neighborhood = 0.1
    epsilon = 1.0
    time_limit = 1.0

    hsvi_neigborhood = HSVI(neighborhood=neighborhood)
    msg = @sprintf(
        "neighborhood parameter = %.4e is outside bounds (%.4e, %.4e)",
        hsvi_neigborhood.neighborhood, 0.0, (1.0 - osposg.discount) * epsilon / (2.0 * HSVIforOSPOSGs.lipschitz_delta(osposg))
    )
    @test_logs (:warn, msg) min_level = Logging.Warn HSVIforOSPOSGs.check_neighborhood(osposg, hsvi_neigborhood, epsilon)

    hsvi = HSVI(presolve_time_limit=time_limit)
    msg = @sprintf("reached time limit of %8.3fs and did not converge, killed", time_limit)
    @test_logs (:warn, msg) min_level = Logging.Warn solve(osposg, hsvi, epsilon, time_limit)
end
