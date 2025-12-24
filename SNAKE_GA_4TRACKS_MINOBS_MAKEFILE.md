# Snake AI (Go, GA) — 4-Track Training System (Minimal Inputs + Simple Fitness + Fast Eval + Makefile)

This is the **Cursor-ready** design doc for a **single-language Go** implementation of Snake AI trained via **Genetic Algorithms**, decomposed into **4 separate GA tracks**:

1) **GA-WALL**: learn wall avoidance + basic survival  
2) **GA-SELF**: learn self-collision avoidance (body awareness)  
3) **GA-FRUIT**: learn fruit pursuit (navigation)  
4) **GA-MULTI**: combine all skills (full objective + robustness)

This design keeps:
- **Ultra-fast evaluation in Go** (no subprocess, no IPC, no per-tick allocations)
- **Config-driven knobs** (YAML)
- **Strong monitoring/debugging** (multi-seed top-K, robustness ranking, fixed benchmark suite, replays)
- **Minimal observations per track** (small vectors → faster learning)

Primary goal:
> Get **GA-WALL** working first (easy), then progressively unlock skills.

---

## 0) Non-Negotiables

- **One Episode → one EpisodeStats**, always fully populated.
- **Multi-seed aggregates all metrics**, not only score.
- **Rank by robustness** to avoid lucky agents:
  - `rankScore = scoreMean - λ*scoreStd` (λ default 0.25)
- **Everything is configurable** (no hard-coded magic numbers).
- **Fast loops**: float32 inference, no allocations in tick loop.

---

## 1) Project Layout (Go Only)

```
cmd/train/main.go

internal/config/config.go

internal/env/
  game.go         // snake env
  features.go     // per-track observation features
  stats.go        // EpisodeStats + aggregation
  replay.go       // optional action trace + seed

internal/nn/mlp.go
internal/ga/
  population.go
  selection.go
  crossover.go
  mutation.go

internal/eval/evaluator.go
internal/logging/metrics.go

configs/
  wall.yaml
  self.yaml
  fruit.yaml
  multi.yaml

artifacts/
runs/
```

---

## 2) Action Space (Make Learning Easier)

Use **relative actions** (recommended):
- `0 = straight`
- `1 = turn left`
- `2 = turn right`

This removes orientation brittleness and makes minimal inputs work well.

Reverse direction is impossible by design.

---

## 3) Minimal Observations per Track (Key Change)

We DO NOT feed the full grid for every task.  
Each track gets only what it needs.

### Shared feature helper concepts
- `danger_*` are binary: 1 if moving there causes wall/body collision next step else 0.
- `body_*_dist` is normalized (0..1): distance to body in that direction, 1 if none.
- All directions are **relative to heading**: front/left/right.

---

### 3.1 GA-WALL observation (3 floats)
Goal: survive, avoid walls.

**Features:**
1. `danger_front_wall` ∈ {0,1}
2. `danger_left_wall`  ∈ {0,1}
3. `danger_right_wall` ∈ {0,1}

Optional (if needed):
- replace binary with normalized distance-to-wall for each direction.

**Obs dim:** 3

---

### 3.2 GA-SELF observation (6–10 floats)
Goal: survive, avoid self collisions (wall still matters).

**Features:**
1. `danger_front` (wall OR body)
2. `danger_left`
3. `danger_right`
4. `body_front_dist` (0..1)
5. `body_left_dist`
6. `body_right_dist`

Optional:
7. `tail_dx` normalized [-1,1]
8. `tail_dy` normalized [-1,1]
9. `length_norm` (0..1)

**Obs dim:** 6 (or 9)

---

### 3.3 GA-FRUIT observation (6–9 floats)
Goal: reach fruit without suicide.

**Features:**
1. `fruit_dx` normalized [-1,1] (relative to heading frame or world; heading frame preferred)
2. `fruit_dy` normalized [-1,1]
3. `danger_front` (wall OR body)
4. `danger_left`
5. `danger_right`
6. `length_norm` (0..1, optional)

Alternative (more invariant):
- `fruit_cos`, `fruit_sin`, `fruit_dist_norm` (3 floats)

**Obs dim:** 5–6 (or 8 with extra)

---

### 3.4 GA-MULTI observation (10–14 floats)
Goal: full behavior.

**Recommended features:**
- dangers (3)
- body ray distances (3)
- fruit cos/sin + dist (3)
- length_norm (1)
- optional tail dx/dy (2)

**Obs dim:** 10–12 (or 14)

---

## 4) Neural Nets (Tiny per Track)

With small obs dims, keep networks small:

- WALL: `3 → 8 → 3`
- SELF: `6/9 → 16 → 3`
- FRUIT: `6/9 → 16 → 3`
- MULTI: `10–14 → 24 → 3`

Use float32, ReLU.

Genome = single contiguous `[]float32`.

---

## 5) Environment Setup per Track

### GA-WALL
- grid: 10x10
- start_length: 1 (or 2)
- fruit_enabled: false
- tick_cap: 200
- stall_window: very large or disabled

### GA-SELF
- grid: 10x10
- start_length: 8 (6–10)
- fruit_enabled: false (or on but not rewarded)
- tick_cap: 200
- stall_window: 60

### GA-FRUIT
- grid: 10x10
- start_length: 3
- fruit_enabled: true
- tick_cap: 150
- stall_window: 35–45

### GA-MULTI
- curriculum enabled: 10→12→16→20
- staged tick caps
- stall_window enabled

---

## 6) EpisodeStats (Single Source of Truth)

```go
type EpisodeStats struct {
  Score float64
  Fruits int
  Ticks int

  ProgressSum float64 // if used
  Death DeathReason
  Seed uint32
}
```

For GA-WALL and GA-SELF, you may omit progress from fitness, but still track it if cheap.

---

## 7) Fitness Functions (Trivial, Track-Specific)

### 7.1 GA-WALL fitness
```
score = ticksSurvived
if death == wall: score -= 500
```

Optional:
- `score -= 0.05*turns` to reduce jitter.

### 7.2 GA-SELF fitness
```
score = ticksSurvived
if death == self: score -= 600
if death == wall: score -= 200
if death == stall: score -= 100
```

### 7.3 GA-FRUIT fitness
Keep it simple:
```
score = 5000*fruits + 2*min(ticks,40) - deathPenalty
deathPenalty: wall/self=300, stall/timeout=150
```

If fruit learning is too sparse, add minimal shaping:
- `+ 10*progressSum` where progressSum is sum of positive distance improvements.

### 7.4 GA-MULTI fitness
```
score = 8000*fruits + 2*min(ticks,60) + 10*progressSum - deathPenalty
```

---

## 8) GA Settings (Config Defaults)

Use the same GA scaffold across tracks.

- population: 200
- elites: 4
- selection_pool: 80
- tournament_k: 3
- crossover_rate: 0.7
- mutation_rate: 0.10
- mutation_sigma: 0.06
- reset_mutation_p: 0.01
- reset_fraction: 0.10

---

## 9) Evaluation and Monitoring (Critical)

### 9.1 Evaluation loop per generation
1) single-seed evaluate population (fast)
2) pick candidates = topK by single-seed score
3) **union candidates with elites + best-ever** (avoid losing champions)
4) multi-seed evaluate candidates (S seeds)
5) rank by robustness:
   - `rankScore = mean - λ*std`

### 9.2 Fixed benchmark suite (progress tracking)
Every `benchmark_every` generations:
- evaluate top 5 on seeds 2000..2009
- log benchmark mean ticks/fruits

This removes confusion from changing random seeds.

### 9.3 Replays (optional but valuable)
Every `replay_every` generations:
- store champion seed + action trace
- allows deterministic playback

---

## 10) Config Files (Per Track)

We keep the **same schema** and override per track.

### Base config schema (YAML)

```yaml
seed: 1337

track:
  mode: "wall"          # wall|self|fruit|multi
  obs: "wall_min"       # wall_min|self_min|fruit_min|multi_min
  actions: "relative3"  # relative3

env:
  width: 10
  height: 10
  start_length: 1
  tick_cap: 200
  stall_window: 9999
  fruit_enabled: false

nn:
  hidden1: 8
  hidden2: 0
  activation: "relu"

ga:
  population: 200
  elites: 4
  selection_pool: 80
  tournament_k: 3
  crossover_rate: 0.7
  mutation_rate: 0.10
  mutation_sigma: 0.06
  reset_mutation_p: 0.01
  reset_fraction: 0.10

eval:
  topk_multiseed: 50
  multiseed_runs: 7
  multiseed_base_seed: 1000
  robustness_lambda: 0.25
  benchmark_every: 50
  benchmark_seeds: [2000,2001,2002,2003,2004,2005,2006,2007,2008,2009]
  workers: 0

logging:
  every_gen_summary: true
  topn_debug: 5
  save_champion_every: 250
  replay_every: 500
  csv_path: "runs/run.csv"
  json_path: "runs/run.jsonl"

fitness:
  mode: "wall"  # wall|self|fruit|multi
  wall_penalty: 500
  self_penalty: 600
  stall_penalty: 100
  fruit_reward: 5000
  survival_cap: 40
  survival_w: 2.0
  progress_w: 10.0
```

### Example: `configs/wall.yaml`
- `track.mode=wall`, `obs=wall_min`, `env.start_length=1`, `tick_cap=200`, `fruit_enabled=false`, `nn.hidden1=8`, `fitness.mode=wall`.

### Example: `configs/self.yaml`
- `track.mode=self`, `obs=self_min`, `env.start_length=8`, `stall_window=60`, `nn.hidden1=16`, `fitness.mode=self`.

### Example: `configs/fruit.yaml`
- `track.mode=fruit`, `obs=fruit_min`, `env.start_length=3`, `tick_cap=150`, `stall_window=40`, `fruit_enabled=true`, `nn.hidden1=16`, `fitness.mode=fruit`.

### Example: `configs/multi.yaml`
- `track.mode=multi`, `obs=multi_min`, curriculum enabled, `nn.hidden1=24`, `fitness.mode=multi`.

---

## 11) Makefile (Train Wall First)

We want the simplest workflow first: **wall avoidance**.

### Makefile targets

- `make build` — build trainer binary
- `make train-wall` — run GA-WALL with `configs/wall.yaml`
- `make train-self` — run GA-SELF
- `make train-fruit` — run GA-FRUIT
- `make train-multi` — run GA-MULTI

Example:

```make
BIN := bin/train

build:
\tmkdir -p bin
\tgo build -o $(BIN) ./cmd/train

train-wall: build
\t$(BIN) -config configs/wall.yaml

train-self: build
\t$(BIN) -config configs/self.yaml

train-fruit: build
\t$(BIN) -config configs/fruit.yaml

train-multi: build
\t$(BIN) -config configs/multi.yaml
```

---

## 12) Acceptance Criteria (Track-by-Track)

### WALL
- mean ticks > 120 (benchmark suite)
- wall deaths drop drastically

### SELF
- self deaths drop sharply at length 8
- mean ticks high without wall-only hacks

### FRUIT
- top10 multi-seed avg fruits ≥ 0.3 on 10x10

### MULTI
- stable fruit when scaling to 12x12 and 16x16 (no collapse)

---

## 13) Debug Playbook (If WALL still fails)

If GA-WALL cannot get mean ticks > 120, assume a bug:
- action mapping wrong
- spawn too close to wall
- collision detection wrong
- relative action turning wrong

Add a scripted baseline policy test:
- “go straight, turn right if wall ahead”
This should survive far longer than 5 ticks on 10x10 if the env is correct.

---

## Final Note

This decomposed approach is deliberately “engineering-first”.
It converts a hard RL-style discovery problem into a sequence of learnable control problems with trivial fitness.

Start with `make train-wall`. Do not proceed until WALL reliably survives.
