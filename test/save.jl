@testset "save" begin
    io = IOBuffer()
    save(io, OSPOSG("../games/pursuit-evasion/peg03.osposg"))
    seekstart(io)
    osposg = OSPOSG(io)

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
