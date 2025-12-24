package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"snakeai/internal/config"
	"snakeai/internal/env"
	"snakeai/internal/eval"
	"snakeai/internal/ga"
	"snakeai/internal/logging"
)

func main() {
	// Parse command line flags
	configPath := flag.String("config", "configs/wall.yaml", "path to config file")
	generations := flag.Int("generations", 1000, "number of generations to run")
	flag.Parse()

	// Load config
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Snake AI Trainer - Track: %s\n", cfg.Track.Mode)
	fmt.Printf("Config: %s\n", *configPath)
	fmt.Printf("Obs: %s (dim=%d), Hidden: %d\n", cfg.Track.Obs, cfg.ObsDim(), cfg.NN.Hidden1)
	fmt.Printf("Population: %d, Elites: %d, Tournament K: %d\n", cfg.GA.Population, cfg.GA.Elites, cfg.GA.TournamentK)
	fmt.Println("---")

	// Initialize RNG
	rng := rand.New(rand.NewSource(cfg.Seed))

	// Create MLP to get genome size
	genomeSize := calcGenomeSize(cfg.ObsDim(), cfg.NN.Hidden1, cfg.NN.Hidden2, 3)
	fmt.Printf("Genome size: %d weights\n", genomeSize)

	// Initialize population
	pop := ga.NewPopulation(cfg.GA.Population, genomeSize, rng)

	// Create evaluator
	evaluator := eval.NewEvaluator(cfg)

	// Create logger
	logger, err := logging.NewLogger(cfg.Logging.CSVPath, cfg.Logging.JSONPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating logger: %v\n", err)
		os.Exit(1)
	}
	if err := logger.Init(); err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing logger: %v\n", err)
		os.Exit(1)
	}
	defer logger.Close()

	// Track best ever for stability
	var bestEver *ga.Agent

	startTime := time.Now()

	// Main training loop
	for gen := 1; gen <= *generations; gen++ {
		genSeed := uint32(cfg.Seed + int64(gen))

		// 1. Evaluate population with single seed (fast)
		evaluator.EvaluatePopulationSingleSeed(pop, genSeed)

		// 2. Log generation summary
		if cfg.Logging.EveryGenSummary {
			logger.LogGeneration(gen, pop)
		}

		// 3. Get top-K candidates for multi-seed evaluation
		pop.SortByFitness()
		topK := cfg.Eval.TopKMultiseed
		if topK > len(pop.Agents) {
			topK = len(pop.Agents)
		}
		candidates := pop.Agents[:topK]

		// Include best ever in candidates to avoid losing champions
		if bestEver != nil {
			// Check if best ever is already in candidates
			found := false
			for _, c := range candidates {
				if &c.Genome[0] == &bestEver.Genome[0] {
					found = true
					break
				}
			}
			if !found {
				candidates = append(candidates, bestEver)
			}
		}

		// 4. Multi-seed evaluation for candidates
		evaluator.EvaluateCandidatesMultiSeed(candidates)

		// 5. Find best by robustness
		var bestRobust *ga.Agent
		for _, c := range candidates {
			if bestRobust == nil || c.RobustScore > bestRobust.RobustScore {
				bestRobust = c
			}
		}

		// Update best ever
		if bestEver == nil || (bestRobust != nil && bestRobust.RobustScore > bestEver.RobustScore) {
			bestEver = bestRobust.Clone()
		}

		// 6. Debug: log top-N
		if gen%10 == 0 && cfg.Logging.TopNDebug > 0 {
			logger.LogTopK(pop.Agents, cfg.Logging.TopNDebug)
		}

		// 7. Benchmark evaluation
		if cfg.Eval.BenchmarkEvery > 0 && gen%cfg.Eval.BenchmarkEvery == 0 {
			benchAgents := pop.TopK(5)
			results := evaluator.RunBenchmark(benchAgents)
			logger.LogBenchmark(gen, results)
		}

		// 8. Save champion
		if cfg.Logging.SaveChampionEvery > 0 && gen%cfg.Logging.SaveChampionEvery == 0 {
			championPath := filepath.Join("artifacts", fmt.Sprintf("champion_gen%d.json", gen))
			if err := logging.SaveChampion(championPath, pop.Best(), gen); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to save champion: %v\n", err)
			}
		}

		// 9. Save replay
		if cfg.Logging.ReplayEvery > 0 && gen%cfg.Logging.ReplayEvery == 0 {
			replay, _ := evaluator.EvaluateWithReplay(pop.Best(), genSeed)
			replayPath := filepath.Join("artifacts", fmt.Sprintf("replay_gen%d.json", gen))
			if err := replay.Save(replayPath); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: failed to save replay: %v\n", err)
			}
		}

		// 10. Create next generation
		nextGen := createNextGeneration(pop, cfg, rng)
		pop.Agents = nextGen
	}

	elapsed := time.Since(startTime)
	fmt.Println("---")
	fmt.Printf("Training complete! %d generations in %v\n", *generations, elapsed)
	if bestEver != nil {
		fmt.Printf("Best ever: Fitness=%.1f, RobustScore=%.1f, Ticks=%d, Fruits=%d\n",
			bestEver.Fitness, bestEver.RobustScore, bestEver.Stats.Ticks, bestEver.Stats.Fruits)

		// Save final champion
		championPath := filepath.Join("artifacts", "champion_final.json")
		if err := logging.SaveChampion(championPath, bestEver, *generations); err != nil {
			fmt.Fprintf(os.Stderr, "Warning: failed to save final champion: %v\n", err)
		}
	}
}

// createNextGeneration creates the next generation via selection, crossover, and mutation
func createNextGeneration(pop *ga.Population, cfg *config.Config, rng *rand.Rand) []*ga.Agent {
	newAgents := make([]*ga.Agent, cfg.GA.Population)

	// 1. Keep elites
	pop.SortByFitness()
	for i := 0; i < cfg.GA.Elites && i < len(pop.Agents); i++ {
		newAgents[i] = pop.Agents[i].Clone()
	}

	// 2. Create selection pool
	pool := ga.SelectionPool(pop, cfg.GA.SelectionPool)

	// 3. Fill rest with offspring
	for i := cfg.GA.Elites; i < cfg.GA.Population; i++ {
		// Select parents
		p1, p2 := ga.SelectParents(pool, cfg.GA.TournamentK, rng)

		// Crossover
		child := ga.CreateChild(p1, p2, cfg.GA.CrossoverRate, rng)

		// Mutation
		ga.MutateAgent(child, cfg.GA.MutationRate, cfg.GA.MutationSigma, cfg.GA.ResetMutationP, rng)

		newAgents[i] = child
	}

	// 4. Optionally reset worst fraction
	if cfg.GA.ResetFraction > 0 && rng.Float64() < 0.1 { // 10% chance per generation
		numReset := int(float64(cfg.GA.Population) * cfg.GA.ResetFraction)
		for i := cfg.GA.Population - numReset; i < cfg.GA.Population; i++ {
			if i >= cfg.GA.Elites { // Don't reset elites
				for j := range newAgents[i].Genome {
					newAgents[i].Genome[j] = float32(rng.NormFloat64() * 0.5)
				}
			}
		}
	}

	return newAgents
}

func calcGenomeSize(inputSize, hidden1, hidden2, outputSize int) int {
	size := 0
	// Input -> Hidden1 (weights + biases)
	size += (inputSize + 1) * hidden1

	if hidden2 > 0 {
		// Hidden1 -> Hidden2
		size += (hidden1 + 1) * hidden2
		// Hidden2 -> Output
		size += (hidden2 + 1) * outputSize
	} else {
		// Hidden1 -> Output
		size += (hidden1 + 1) * outputSize
	}
	return size
}

// runScriptedTest runs a simple scripted policy to verify the environment works
func runScriptedTest(cfg *config.Config) {
	fmt.Println("Running scripted baseline test...")

	game := env.NewGame(
		cfg.Env.Width,
		cfg.Env.Height,
		cfg.Env.StartLength,
		cfg.Env.TickCap,
		cfg.Env.StallWindow,
		cfg.Env.FruitEnabled,
		12345,
	)

	// Simple policy: go straight, turn right if wall ahead
	for game.Alive {
		action := env.ActionStraight
		if game.IsDangerWall(env.ActionStraight) {
			if !game.IsDangerWall(env.ActionRight) {
				action = env.ActionRight
			} else if !game.IsDangerWall(env.ActionLeft) {
				action = env.ActionLeft
			}
		}
		game.Step(action)
	}

	stats := game.Stats(12345)
	fmt.Printf("Scripted test: Ticks=%d, Death=%s\n", stats.Ticks, stats.Death)
	fmt.Println("If ticks < 50 on 10x10, there may be a bug in the environment.")
}

