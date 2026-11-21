# Sokoban Solver

A push-optimal Sokoban solver for make.fun's `sokoban-solver-01` competition.

## How it works

The levels this competition draws are a one-square-wide hallway with small
rooms off it (3–4 boxes each, one door onto the hallway). A box pushed out
of a room into the hallway can never come back, so every box stays in its
room and a level's push optimum is the sum of its rooms' optima.

`main.ts` therefore:

1. **Splits each level into rooms** — the hallway row's dead squares cut the
   board, and each connected part holding boxes is a room.
2. **Solves every room on its own** with a push-based A* search: a state is
   the boxes' squares plus the player's reachable region (named by its
   lowest square), so walking is free and only pushes cost. The heuristic
   is the cheapest assignment of boxes to goals under walking distances
   from the goals, which never overestimates the pushes left, so the first
   solved state A\* pops is push-optimal for its room.
3. **Prunes dead states** with dead squares (found by pulling boxes back
   from the goals), frozen 2×2 blocks, and boxes walled in on all four
   sides.
4. **Stitches the rooms together** with walks along the hallway and
   reconstructs the full LURD move string.

Every search budget is a count of expanded states, never the clock, so two
runs write byte-identical answers. No network, no clock reads, no
randomness.

## Run

```sh
node main.ts
```

Reads `/input/<id>/level.txt` for each instance and writes
`/output/<id>/solution.txt`. Locally:

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Results

On the practice set the solver reaches the push optimum on all five
instances (13, 16, 29, 47 and 60 pushes) for a practice score of
**1,000,000**, in under half a second for the whole set. It scores
1,000,000 on 30 further freshly drawn sets as well (each drawn from its
own seed with the kit's published generator), every set in under 0.6 s,
and two runs on the same set write byte-identical answers. The house
baseline (plain-astar 1.0.0) scores 200,000 on the practice set.

## Licence

MIT — see `LICENSE`.
