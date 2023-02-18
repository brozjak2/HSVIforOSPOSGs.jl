```@meta
CurrentModule = HSVIforOSPOSGs
```

# Game format

The games must be fed into the algorithm in compatible format, which is specified in [`games/game_format.txt`](https://github.com/brozjak2/HSVIforOSPOSGs.jl/blob/master/games/game_format.txt):

```
The file consists of the following sections:
  1. Description of the game (number of states etc.)
  2. Description of the game states
  3. Names of actions of Player 1
  4. Names of actions of Player 2
  5. Names of observations
  6. Enumeration of actions playable by Player 2 in each of the states
  7. Enumeration of actions playable by Player 1 in each of the partition
  8. Enumeration of game transitions ( Pr[o,s' | s,a,a'] )
  9. Enumeration of rewards
 1.  Initial belief

1. Description of the game
==========================
The first line of the file contains 8 numbers (in this order separated by
white-space) as follows:
  a) Number of states (|S|)
  b) Number of "partitions". P1 (with imperfect information) always knows
     the partition he is currently in. Partitions typically correspond to
     the perfectly observable components of the state description (e.g.,
     the location of P1).
  c) Number of P1 actions (|A_1|)
  d) Number of P2 actions (|A_2|)
  e) Number of observations (|O|)
  f) Number of transitions (i.e., lines in the section 7 of the file)
  g) Number of rewards (i.e., lines in the section 8 of the file)
  h) Discount factor (floating-point number)

1. Description of the game states
=================================
This section contains |S| lines containing a string and an integer sepa-
rated by a white space. The string is the name of the i-th state (just
for convenience) and the integer denotes the partition the state belongs
to (numbered from zero).
  [state name] [partition ID]

1. Names of actions of Player 1
===============================
This section contains |A_1| strings on separate lines. (The names of the
actions are just for the purpose of convenience.)

1. Names of actions of Player 2
===============================
Similar to section 2, except for describing actions of Player 2.

1. Names of observations
========================
Similar to sections 2 and 3. Each line represents the name of one obser-
vation in the game.

1. Enumeration of actions playable by Player 2
==============================================
The solver assumes that there may be different set of actions that can be
played in different states. This section contains |S| lines and i-th line
enumerates actions that can be played by Player 2 in i-th state. The list
of actions is a white-space separated integers [0, |A_2|-1].

1. Enumeration of actions playable by Player 1
==============================================
Also Player 1 can play different actions in different partitions. Every
line in this section (they must match the number of partitions) denotes
the set of actions that can be played by Player 1 in the given partition
(a list of [0, |A_1|-1] partitions.

1. Enumeration of game transitions
==================================
Each line in this section describes one non-zero entry of the transition
function. The format of each line is as follows:
  [s] [a] [a'] [o] [s'] [prob]
which describes that the probability to transition from [s] to [s'] and
generating observation [o] when actions [a] (of P1) and [a'] has been
played is [prob]. [s], [a], [a'], [o] and [s'] are zero-based indices of
states, actions and observations.

1. Enumeration of rewards
=========================
Each line in this section describes one non-zero entry of the reward fun-
ction R. The format is as follows:
  [s] [a] [a'] [reward]
which means that the reward of Player 1 when taking actions [a] (of P1)
and [a'] in the state [s] is [reward]. [s], [a] and [a'] are zero-based
indices of states and actions. [reward] can be floating-point.

1.  Initial belief
==================
The initial belief is described by a single line containing white-space
separated numbers. The first number identifies the initial partition of
the game (Player 1 knows the partition). The remaining K numbers (where
K is the number of states in the initial partition) denote the probabi-
lity that the game starts in the given state of the partition.
```

Below is an example [`games/pursuit-evasion/peg04.osposg`](https://github.com/brozjak2/HSVIforOSPOSGs.jl/blob/master/games/pursuit-evasion/peg03.osposg) that shows how a game in this format may look like (`...` skips over some inner lines of long sections):

```
143 21 145 13 2 2671 2671 0.9500
[[0:0,_0:1],_0:0] 4
[[0:0,_0:1],_0:1] 4
[[0:0,_0:1],_0:2] 4
...
[[1:0,_2:2],_0:2] 14
[[1:2,_2:2],_0:2] 15
[[2:1,_2:2],_0:2] 11
end 20
[PursuerAction{source=0:1,_target=1:1},_PursuerAction{source=0:2,_target=1:2}]
[PursuerAction{source=0:1,_target=0:0},_PursuerAction{source=0:2,_target=1:2}]
[PursuerAction{source=0:1,_target=0:0},_PursuerAction{source=0:2,_target=0:1}]
...
[PursuerAction{source=1:1,_target=2:1},_PursuerAction{source=2:1,_target=2:0}]
[PursuerAction{source=1:1,_target=1:0},_PursuerAction{source=2:1,_target=2:2}]
[PursuerAction{source=2:1,_target=1:1},_PursuerAction{source=1:1,_target=0:1}]
end
e0[0:0--1:0]
e6[0:0--0:1]
e1[0:1--1:1]
...
e5[1:2--2:2]
e8[2:0--2:1]
e11[2:1--2:2]
end
end
cont
0 1
2 1 3
4 3
...
4 3
4 3
12
0 1 2 3 4 5
6 7 8 9 10 11
12 13 14 15 16 17 18 19 20 21 22 23
...
126 127 128 129 130 131
132 133 134 135 136 137 138 139 140 141 142 143
144
9 0 7 0 142 1.000000
82 64 11 1 30 1.000000
22 52 9 1 16 1.000000
...
118 78 6 1 130 1.000000
28 120 5 1 33 1.000000
46 100 5 1 32 1.000000
53 72 5 95.000000
64 7 3 0.000000
96 88 3 0.000000
...
133 70 1 0.000000
137 83 7 95.000000
57 143 9 0.000000
4 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 1.0000
```
