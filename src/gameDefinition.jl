struct Game
    states
    actions1
    actions2
    observations
    reward
    transition
end

function transition(s, a1, a2, o, sp)
    @assert 1 <= s <= 4
    @assert 1 <= a1 <= 2
    @assert 1 <= a2 <= 2
    @assert 1 <= o <= 4
    @assert 1 <= sp <= 4

    if (s != 4)
        if (sp == s && o == s)
            return 1
        end
    elseif (a1 == 2)
        if (sp == 3 && o == 3)
            return 1
        end
    elseif (a2 == 1)
        if (sp == 1 && o == 1)
            return 1
        end
    else
        if (sp == 2 && o == 2)
            return 1
        end
    end

    return 0
end

nstates = 4
nactions1 = 2
nactions2 = 2
nobservations = 4
reward = cat([0. 0.; 0. 0.; 0. 0.; 1. 3.], [0. 0.; 0. 0.; 0. 0.; 4. 3.], dims=3)

game = Game(1:nstates, 1:nactions1, 1:nactions2, 1:nobservations,
            reward, transition)
