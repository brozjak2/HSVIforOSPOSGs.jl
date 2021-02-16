include("types/game_params.jl")
include("types/reward.jl")
include("types/transition.jl")
include("types/parsed_game_definition.jl")
include("types/state.jl")

const AOPair = Tuple{Int64,Int64}
const ActionPair = Tuple{Int64,Int64}
const SAAOSTuple = Tuple{Int64,Int64,Int64,Int64,Int64}
