# Evolution Idea 01: A* Routing Heuristic + Prompt Co-Evolution

## Core Idea
Evolve the priority and tie-break functions of A* node expansion while also co-evolving
mutation prompt strategies.

## Why This Is Good For Evolution
- The objective is non-convex and symbolic (good for mutation + selection).
- Small formula edits can create qualitatively different behavior.
- Failure modes are easy to score and feed back (unplaced jobs, imbalance, fragmentation).

## What To Evolve
In `mvp/tasks/astar_routing_target.py` evolve only:
- `priority_score(features)`
- `tie_break_priority(features)`

## Metrics Used In Current Evaluator
- `solved_ratio`
- `path_quality`
- `expansion_efficiency`
- `route_smoothness`

Aggregate idea:
`0.50*solved + 0.25*quality + 0.15*efficiency + 0.10*smoothness`

## Evolution Cadence (Practical)
- 100 to 300 generations in `gemini` mode
- keep `top-k=6`, `diversity-slots=2`
- every 25 generations, inspect best formulas and top prompt strategies (`summary.json -> top_prompts`)

## Success Signal
You should see survivors gradually converge to formulas that:
- solve more grids reliably,
- expand fewer nodes per solved route,
- keep routes smoother,
- and pair with prompt styles that repeatedly produce valid high-gain diffs.
