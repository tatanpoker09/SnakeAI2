package eval

import (
	"math"
	"runtime"
	"sync"

	"snakeai/internal/config"
	"snakeai/internal/env"
	"snakeai/internal/ga"
	"snakeai/internal/nn"
)

// Evaluator handles episode evaluation and fitness computation
type Evaluator struct {
	cfg      *config.Config
	features *env.FeatureExtractor
	mlp      *nn.MLP
	workers  int
}

// NewEvaluator creates a new evaluator
func NewEvaluator(cfg *config.Config) *Evaluator {
	workers := cfg.Eval.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	return &Evaluator{
		cfg:      cfg,
		features: env.NewFeatureExtractor(cfg.Track.Obs),
		mlp:      nn.NewMLP(cfg.ObsDim(), cfg.NN.Hidden1, cfg.NN.Hidden2, 3),
		workers:  workers,
	}
}

// EvaluateAgent runs a single episode with the given agent and seed
func (e *Evaluator) EvaluateAgent(agent *ga.Agent, seed uint32) env.EpisodeStats {
	// Create game
	game := env.NewGame(
		e.cfg.Env.Width,
		e.cfg.Env.Height,
		e.cfg.Env.StartLength,
		e.cfg.Env.TickCap,
		e.cfg.Env.StallWindow,
		e.cfg.Env.FruitEnabled,
		seed,
	)

	// Create local MLP and feature extractor (avoid race conditions)
	mlp := nn.NewMLP(e.cfg.ObsDim(), e.cfg.NN.Hidden1, e.cfg.NN.Hidden2, 3)
	mlp.SetWeights(agent.Genome)
	features := env.NewFeatureExtractor(e.cfg.Track.Obs)

	// Run episode
	for game.Alive {
		obs := features.Extract(game)
		action := mlp.Forward(obs)
		game.Step(env.Action(action))
	}

	stats := game.Stats(seed)
	stats.Score = e.ComputeFitness(stats)
	return stats
}

// EvaluatePopulationSingleSeed evaluates all agents with a single seed
func (e *Evaluator) EvaluatePopulationSingleSeed(pop *ga.Population, seed uint32) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, e.workers)

	for _, agent := range pop.Agents {
		wg.Add(1)
		sem <- struct{}{}
		go func(a *ga.Agent) {
			defer wg.Done()
			defer func() { <-sem }()
			stats := e.EvaluateAgent(a, seed)
			a.Stats = stats
			a.Fitness = stats.Score
		}(agent)
	}
	wg.Wait()
}

// EvaluateMultiSeed evaluates an agent across multiple seeds
func (e *Evaluator) EvaluateMultiSeed(agent *ga.Agent, baseSeed int, numSeeds int) env.AggregatedStats {
	episodes := make([]env.EpisodeStats, numSeeds)
	for i := 0; i < numSeeds; i++ {
		seed := uint32(baseSeed + i)
		episodes[i] = e.EvaluateAgent(agent, seed)
	}
	return env.Aggregate(episodes)
}

// EvaluateCandidatesMultiSeed evaluates top-K candidates with multiple seeds
func (e *Evaluator) EvaluateCandidatesMultiSeed(candidates []*ga.Agent) {
	var wg sync.WaitGroup
	sem := make(chan struct{}, e.workers)

	for _, agent := range candidates {
		wg.Add(1)
		sem <- struct{}{}
		go func(a *ga.Agent) {
			defer wg.Done()
			defer func() { <-sem }()
			agg := e.EvaluateMultiSeed(a, e.cfg.Eval.MultiseedBaseSeed, e.cfg.Eval.MultiseedRuns)
			a.AggStats = agg
			a.RobustScore = agg.RobustnessScore(e.cfg.Eval.RobustnessLambda)
		}(agent)
	}
	wg.Wait()
}

// RunBenchmark evaluates agents on the fixed benchmark seed suite
func (e *Evaluator) RunBenchmark(agents []*ga.Agent) []env.AggregatedStats {
	results := make([]env.AggregatedStats, len(agents))
	seeds := e.cfg.Eval.BenchmarkSeeds

	for i, agent := range agents {
		episodes := make([]env.EpisodeStats, len(seeds))
		for j, seed := range seeds {
			episodes[j] = e.EvaluateAgent(agent, uint32(seed))
		}
		results[i] = env.Aggregate(episodes)
	}
	return results
}

// ComputeFitness computes the fitness score based on track mode
func (e *Evaluator) ComputeFitness(stats env.EpisodeStats) float64 {
	switch e.cfg.Fitness.Mode {
	case "wall":
		return e.fitnessWall(stats)
	case "self":
		return e.fitnessSelf(stats)
	case "fruit":
		return e.fitnessFruit(stats)
	case "multi":
		return e.fitnessMulti(stats)
	default:
		return e.fitnessWall(stats)
	}
}

func (e *Evaluator) fitnessWall(stats env.EpisodeStats) float64 {
	score := float64(stats.Ticks)
	if stats.Death == env.DeathWall {
		score -= e.cfg.Fitness.WallPenalty
	}
	return score
}

func (e *Evaluator) fitnessSelf(stats env.EpisodeStats) float64 {
	score := float64(stats.Ticks)
	switch stats.Death {
	case env.DeathSelf:
		score -= e.cfg.Fitness.SelfPenalty
	case env.DeathWall:
		score -= e.cfg.Fitness.WallPenalty * 0.33 // lighter wall penalty for self track
	case env.DeathStall:
		score -= e.cfg.Fitness.StallPenalty
	}
	return score
}

func (e *Evaluator) fitnessFruit(stats env.EpisodeStats) float64 {
	score := e.cfg.Fitness.FruitReward * float64(stats.Fruits)
	survivalTicks := math.Min(float64(stats.Ticks), float64(e.cfg.Fitness.SurvivalCap))
	score += e.cfg.Fitness.SurvivalW * survivalTicks
	score += e.cfg.Fitness.ProgressW * stats.ProgressSum

	switch stats.Death {
	case env.DeathWall, env.DeathSelf:
		score -= 300
	case env.DeathStall, env.DeathTimeout:
		score -= 150
	}
	return score
}

func (e *Evaluator) fitnessMulti(stats env.EpisodeStats) float64 {
	score := 8000 * float64(stats.Fruits)
	survivalTicks := math.Min(float64(stats.Ticks), 60)
	score += 2 * survivalTicks
	score += e.cfg.Fitness.ProgressW * stats.ProgressSum

	switch stats.Death {
	case env.DeathWall, env.DeathSelf:
		score -= 300
	case env.DeathStall, env.DeathTimeout:
		score -= 150
	}
	return score
}

// EvaluateWithReplay runs an episode and records actions for replay
func (e *Evaluator) EvaluateWithReplay(agent *ga.Agent, seed uint32) (*env.Replay, env.EpisodeStats) {
	game := env.NewGame(
		e.cfg.Env.Width,
		e.cfg.Env.Height,
		e.cfg.Env.StartLength,
		e.cfg.Env.TickCap,
		e.cfg.Env.StallWindow,
		e.cfg.Env.FruitEnabled,
		seed,
	)

	replayCfg := env.ReplayConfig{
		Width:        e.cfg.Env.Width,
		Height:       e.cfg.Env.Height,
		StartLength:  e.cfg.Env.StartLength,
		TickCap:      e.cfg.Env.TickCap,
		StallWindow:  e.cfg.Env.StallWindow,
		FruitEnabled: e.cfg.Env.FruitEnabled,
	}
	replay := env.NewReplay(seed, replayCfg)

	mlp := nn.NewMLP(e.cfg.ObsDim(), e.cfg.NN.Hidden1, e.cfg.NN.Hidden2, 3)
	mlp.SetWeights(agent.Genome)
	features := env.NewFeatureExtractor(e.cfg.Track.Obs)

	for game.Alive {
		obs := features.Extract(game)
		action := mlp.Forward(obs)
		replay.Record(env.Action(action))
		game.Step(env.Action(action))
	}

	stats := game.Stats(seed)
	stats.Score = e.ComputeFitness(stats)
	replay.SetFinalStats(stats)

	return replay, stats
}

