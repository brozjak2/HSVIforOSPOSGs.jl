using HSVIforOSPOSGs
using Test
using Logging
using Printf: @sprintf

osposg = OSPOSG("../games/pursuit-evasion/peg03.osposg")
hsvi = HSVI()
epsilon = 1.0
time_limit = 60.0

@testset "HSVIforOSPOSGs.jl" begin
    @testset "load" begin
        @test length(osposg.states) == 143
        @test length(osposg.state_labels) == 143
        @test length(osposg.partitions) == 21
        @test length(osposg.player1_action_labels) == 145
        @test length(osposg.player2_action_labels) == 13
        @test length(osposg.observation_labels) == 2
        @test length(osposg.transition_map) == 2671
        @test length(osposg.reward_map) == 2671
        @test osposg.discount == 0.95
        @test osposg.initial_partition == 5
        @test osposg.initial_belief == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        @test osposg.state_labels[1] == "[[0:0,_0:1],_0:0]"
        @test osposg.states[1].partition == 5

        @test osposg.player1_action_labels[1] == "[PursuerAction{source=0:1,_target=1:1},_PursuerAction{source=0:2,_target=1:2}]"
        @test osposg.player2_action_labels[1] == "e0[0:0--1:0]"
        @test osposg.observation_labels[1] == "end"

        @test osposg.states[5].player2_actions == [3, 8, 7, 9]
        @test osposg.partitions[3].player1_actions == [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        @test osposg.transition_map[10, 1, 8, 1, 143] == 1.0
        @test osposg.reward_map[10, 1, 8] == 95.0

        @test osposg.minimal_reward == 0.0
        @test osposg.maximal_reward == 100.0
    end

    @testset "checks" begin
        @test_throws HSVIforOSPOSGs.MultiPartitionTransitionException OSPOSG("games/multi_partition_transition.osposg")
        @test_throws HSVIforOSPOSGs.IsNotDistributionException OSPOSG("games/is_not_distribution.osposg")
        @test_throws ArgumentError OSPOSG("games/discount.osposg")
    end

    @testset "save" begin
        io = IOBuffer()
        save(io, osposg)
        seekstart(io)
        osposg_saved = OSPOSG(io)

        @test length(osposg_saved.states) == 143
        @test length(osposg_saved.state_labels) == 143
        @test length(osposg_saved.partitions) == 21
        @test length(osposg_saved.player1_action_labels) == 145
        @test length(osposg_saved.player2_action_labels) == 13
        @test length(osposg_saved.observation_labels) == 2
        @test length(osposg_saved.transition_map) == 2671
        @test length(osposg_saved.reward_map) == 2671
        @test osposg_saved.discount == 0.95
        @test osposg_saved.initial_partition == 5
        @test osposg_saved.initial_belief == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        @test osposg_saved.state_labels[1] == "[[0:0,_0:1],_0:0]"
        @test osposg_saved.states[1].partition == 5

        @test osposg_saved.player1_action_labels[1] == "[PursuerAction{source=0:1,_target=1:1},_PursuerAction{source=0:2,_target=1:2}]"
        @test osposg_saved.player2_action_labels[1] == "e0[0:0--1:0]"
        @test osposg_saved.observation_labels[1] == "end"

        @test osposg_saved.states[5].player2_actions == [3, 8, 7, 9]
        @test osposg_saved.partitions[3].player1_actions == [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

        @test osposg_saved.transition_map[10, 1, 8, 1, 143] == 1.0
        @test osposg_saved.reward_map[10, 1, 8] == 95.0

        @test osposg_saved.minimal_reward == 0.0
        @test osposg_saved.maximal_reward == 100.0
    end

    @testset "logs" begin
        neighborhood = 0.1
        hsvi_neigborhood = HSVI(neighborhood=neighborhood)

        upper_limit = (1.0 - osposg.discount) * epsilon / (2.0 * HSVIforOSPOSGs.lipschitz_delta(osposg))
        msg = @sprintf(
            "neighborhood parameter = %.4e is outside bounds (%.4e, %.4e)",
            hsvi_neigborhood.neighborhood, 0.0, upper_limit
        )
        @test_logs (:warn, msg) min_level = Logging.Warn HSVIforOSPOSGs.check_neighborhood(osposg, hsvi_neigborhood, epsilon)

        logs_time_limit = 1.0
        hsvi_time_limit = HSVI(presolve_time_limit=logs_time_limit)

        msg = @sprintf("reached time limit of %8.3fs and did not converge, killed", logs_time_limit)
        @test_logs (:warn, msg) min_level = Logging.Warn solve(deepcopy(osposg), hsvi_time_limit, epsilon, logs_time_limit)
    end

    @testset "OSPOSG" begin
        reward_osposg = OSPOSG("games/rewards.osposg")

        @test HSVIforOSPOSGs.LB_min(reward_osposg) ≈ 0.0
        @test HSVIforOSPOSGs.UB_max(reward_osposg) ≈ 20.0
        @test HSVIforOSPOSGs.lipschitz_delta(reward_osposg) ≈ 10.0
    end

    @testset "solve" begin
        with_logger(ConsoleLogger(stdout, Logging.Error)) do
            osposg_solve = deepcopy(osposg)
            recorder = solve(osposg_solve, hsvi, epsilon, time_limit)

            @test HSVIforOSPOSGs.width(osposg_solve, hsvi) < epsilon
            @test HSVIforOSPOSGs.LB_value(osposg_solve, hsvi) < 83.443625 < HSVIforOSPOSGs.UB_value(osposg_solve, hsvi)
        end
    end
end
