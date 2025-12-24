package config

import (
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration structure
type Config struct {
	Seed    int64        `yaml:"seed"`
	Track   TrackConfig  `yaml:"track"`
	Env     EnvConfig    `yaml:"env"`
	NN      NNConfig     `yaml:"nn"`
	GA      GAConfig     `yaml:"ga"`
	Eval    EvalConfig   `yaml:"eval"`
	Logging LogConfig    `yaml:"logging"`
	Fitness FitnessConfig `yaml:"fitness"`
}

// TrackConfig defines the training track
type TrackConfig struct {
	Mode    string `yaml:"mode"`    // wall|self|fruit|multi
	Obs     string `yaml:"obs"`     // wall_min|self_min|fruit_min|multi_min
	Actions string `yaml:"actions"` // relative3
}

// EnvConfig defines environment parameters
type EnvConfig struct {
	Width        int  `yaml:"width"`
	Height       int  `yaml:"height"`
	StartLength  int  `yaml:"start_length"`
	TickCap      int  `yaml:"tick_cap"`
	StallWindow  int  `yaml:"stall_window"`
	FruitEnabled bool `yaml:"fruit_enabled"`
}

// NNConfig defines neural network architecture
type NNConfig struct {
	Hidden1    int    `yaml:"hidden1"`
	Hidden2    int    `yaml:"hidden2"`
	Activation string `yaml:"activation"` // relu
}

// GAConfig defines genetic algorithm parameters
type GAConfig struct {
	Population      int     `yaml:"population"`
	Elites          int     `yaml:"elites"`
	SelectionPool   int     `yaml:"selection_pool"`
	TournamentK     int     `yaml:"tournament_k"`
	CrossoverRate   float64 `yaml:"crossover_rate"`
	MutationRate    float64 `yaml:"mutation_rate"`
	MutationSigma   float64 `yaml:"mutation_sigma"`
	ResetMutationP  float64 `yaml:"reset_mutation_p"`
	ResetFraction   float64 `yaml:"reset_fraction"`
}

// EvalConfig defines evaluation parameters
type EvalConfig struct {
	TopKMultiseed     int     `yaml:"topk_multiseed"`
	MultiseedRuns     int     `yaml:"multiseed_runs"`
	MultiseedBaseSeed int     `yaml:"multiseed_base_seed"`
	RobustnessLambda  float64 `yaml:"robustness_lambda"`
	BenchmarkEvery    int     `yaml:"benchmark_every"`
	BenchmarkSeeds    []int   `yaml:"benchmark_seeds"`
	Workers           int     `yaml:"workers"`
}

// LogConfig defines logging parameters
type LogConfig struct {
	EveryGenSummary   bool   `yaml:"every_gen_summary"`
	TopNDebug         int    `yaml:"topn_debug"`
	SaveChampionEvery int    `yaml:"save_champion_every"`
	ReplayEvery       int    `yaml:"replay_every"`
	CSVPath           string `yaml:"csv_path"`
	JSONPath          string `yaml:"json_path"`
}

// FitnessConfig defines fitness function parameters
type FitnessConfig struct {
	Mode         string  `yaml:"mode"` // wall|self|fruit|multi
	WallPenalty  float64 `yaml:"wall_penalty"`
	SelfPenalty  float64 `yaml:"self_penalty"`
	StallPenalty float64 `yaml:"stall_penalty"`
	FruitReward  float64 `yaml:"fruit_reward"`
	SurvivalCap  int     `yaml:"survival_cap"`
	SurvivalW    float64 `yaml:"survival_w"`
	ProgressW    float64 `yaml:"progress_w"`
}

// Load reads a YAML config file and returns a Config
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	cfg := &Config{}
	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	// Apply defaults
	applyDefaults(cfg)
	return cfg, nil
}

func applyDefaults(cfg *Config) {
	if cfg.Seed == 0 {
		cfg.Seed = 1337
	}
	if cfg.Track.Actions == "" {
		cfg.Track.Actions = "relative3"
	}
	if cfg.Env.Width == 0 {
		cfg.Env.Width = 10
	}
	if cfg.Env.Height == 0 {
		cfg.Env.Height = 10
	}
	if cfg.Env.StartLength == 0 {
		cfg.Env.StartLength = 1
	}
	if cfg.Env.TickCap == 0 {
		cfg.Env.TickCap = 200
	}
	if cfg.Env.StallWindow == 0 {
		cfg.Env.StallWindow = 9999
	}
	if cfg.NN.Hidden1 == 0 {
		cfg.NN.Hidden1 = 8
	}
	if cfg.NN.Activation == "" {
		cfg.NN.Activation = "relu"
	}
	if cfg.GA.Population == 0 {
		cfg.GA.Population = 200
	}
	if cfg.GA.Elites == 0 {
		cfg.GA.Elites = 4
	}
	if cfg.GA.SelectionPool == 0 {
		cfg.GA.SelectionPool = 80
	}
	if cfg.GA.TournamentK == 0 {
		cfg.GA.TournamentK = 3
	}
	if cfg.GA.CrossoverRate == 0 {
		cfg.GA.CrossoverRate = 0.7
	}
	if cfg.GA.MutationRate == 0 {
		cfg.GA.MutationRate = 0.10
	}
	if cfg.GA.MutationSigma == 0 {
		cfg.GA.MutationSigma = 0.06
	}
	if cfg.GA.ResetMutationP == 0 {
		cfg.GA.ResetMutationP = 0.01
	}
	if cfg.GA.ResetFraction == 0 {
		cfg.GA.ResetFraction = 0.10
	}
	if cfg.Eval.TopKMultiseed == 0 {
		cfg.Eval.TopKMultiseed = 50
	}
	if cfg.Eval.MultiseedRuns == 0 {
		cfg.Eval.MultiseedRuns = 7
	}
	if cfg.Eval.MultiseedBaseSeed == 0 {
		cfg.Eval.MultiseedBaseSeed = 1000
	}
	if cfg.Eval.RobustnessLambda == 0 {
		cfg.Eval.RobustnessLambda = 0.25
	}
	if cfg.Eval.BenchmarkEvery == 0 {
		cfg.Eval.BenchmarkEvery = 50
	}
	if len(cfg.Eval.BenchmarkSeeds) == 0 {
		cfg.Eval.BenchmarkSeeds = []int{2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009}
	}
	if cfg.Logging.TopNDebug == 0 {
		cfg.Logging.TopNDebug = 5
	}
	if cfg.Logging.SaveChampionEvery == 0 {
		cfg.Logging.SaveChampionEvery = 250
	}
	if cfg.Logging.ReplayEvery == 0 {
		cfg.Logging.ReplayEvery = 500
	}
	if cfg.Logging.CSVPath == "" {
		cfg.Logging.CSVPath = "runs/run.csv"
	}
	if cfg.Logging.JSONPath == "" {
		cfg.Logging.JSONPath = "runs/run.jsonl"
	}
	if cfg.Fitness.WallPenalty == 0 {
		cfg.Fitness.WallPenalty = 500
	}
	if cfg.Fitness.SelfPenalty == 0 {
		cfg.Fitness.SelfPenalty = 600
	}
	if cfg.Fitness.StallPenalty == 0 {
		cfg.Fitness.StallPenalty = 100
	}
	if cfg.Fitness.FruitReward == 0 {
		cfg.Fitness.FruitReward = 5000
	}
	if cfg.Fitness.SurvivalCap == 0 {
		cfg.Fitness.SurvivalCap = 40
	}
	if cfg.Fitness.SurvivalW == 0 {
		cfg.Fitness.SurvivalW = 2.0
	}
	if cfg.Fitness.ProgressW == 0 {
		cfg.Fitness.ProgressW = 10.0
	}
}

// ObsDim returns the observation dimension for the given obs type
func (c *Config) ObsDim() int {
	switch c.Track.Obs {
	case "wall_min":
		return 3
	case "self_min":
		return 6
	case "fruit_min":
		return 6
	case "multi_min":
		return 10
	default:
		return 3
	}
}

