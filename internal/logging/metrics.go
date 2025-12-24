package logging

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"

	"snakeai/internal/env"
	"snakeai/internal/ga"
)

// Logger handles all training output and artifact saving
type Logger struct {
	csvPath     string
	jsonPath    string
	csvFile     *os.File
	csvWriter   *csv.Writer
	jsonFile    *os.File
	initialized bool
}

// NewLogger creates a new logger
func NewLogger(csvPath, jsonPath string) (*Logger, error) {
	l := &Logger{
		csvPath:  csvPath,
		jsonPath: jsonPath,
	}

	// Ensure directories exist
	if err := os.MkdirAll(filepath.Dir(csvPath), 0755); err != nil {
		return nil, err
	}
	if err := os.MkdirAll(filepath.Dir(jsonPath), 0755); err != nil {
		return nil, err
	}

	return l, nil
}

// Init initializes the log files
func (l *Logger) Init() error {
	var err error

	// Open CSV file
	l.csvFile, err = os.Create(l.csvPath)
	if err != nil {
		return err
	}
	l.csvWriter = csv.NewWriter(l.csvFile)

	// Write CSV header
	header := []string{
		"generation", "best_fitness", "mean_fitness", "best_ticks", "mean_ticks",
		"best_fruits", "mean_fruits", "deaths_wall", "deaths_self", "deaths_stall", "deaths_timeout",
	}
	if err := l.csvWriter.Write(header); err != nil {
		return err
	}

	// Open JSON file
	l.jsonFile, err = os.OpenFile(l.jsonPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}

	l.initialized = true
	return nil
}

// Close closes all log files
func (l *Logger) Close() {
	if l.csvWriter != nil {
		l.csvWriter.Flush()
	}
	if l.csvFile != nil {
		l.csvFile.Close()
	}
	if l.jsonFile != nil {
		l.jsonFile.Close()
	}
}

// GenerationSummary holds per-generation statistics
type GenerationSummary struct {
	Generation    int                    `json:"generation"`
	BestFitness   float64                `json:"best_fitness"`
	MeanFitness   float64                `json:"mean_fitness"`
	BestTicks     int                    `json:"best_ticks"`
	MeanTicks     float64                `json:"mean_ticks"`
	BestFruits    int                    `json:"best_fruits"`
	MeanFruits    float64                `json:"mean_fruits"`
	DeathCounts   map[string]int         `json:"death_counts"`
	RobustScore   float64                `json:"robust_score,omitempty"`
	BenchmarkTicks float64               `json:"benchmark_ticks,omitempty"`
}

// LogGeneration logs a generation summary
func (l *Logger) LogGeneration(gen int, pop *ga.Population) {
	if !l.initialized {
		return
	}

	// Compute statistics
	var sumFitness, sumTicks, sumFruits float64
	deathCounts := make(map[env.DeathReason]int)
	best := pop.Best()

	for _, a := range pop.Agents {
		sumFitness += a.Fitness
		sumTicks += float64(a.Stats.Ticks)
		sumFruits += float64(a.Stats.Fruits)
		deathCounts[a.Stats.Death]++
	}

	n := float64(len(pop.Agents))
	summary := GenerationSummary{
		Generation:  gen,
		BestFitness: best.Fitness,
		MeanFitness: sumFitness / n,
		BestTicks:   best.Stats.Ticks,
		MeanTicks:   sumTicks / n,
		BestFruits:  best.Stats.Fruits,
		MeanFruits:  sumFruits / n,
		DeathCounts: make(map[string]int),
	}

	for reason, count := range deathCounts {
		summary.DeathCounts[reason.String()] = count
	}

	// Write CSV row
	row := []string{
		strconv.Itoa(gen),
		fmt.Sprintf("%.2f", summary.BestFitness),
		fmt.Sprintf("%.2f", summary.MeanFitness),
		strconv.Itoa(summary.BestTicks),
		fmt.Sprintf("%.2f", summary.MeanTicks),
		strconv.Itoa(summary.BestFruits),
		fmt.Sprintf("%.2f", summary.MeanFruits),
		strconv.Itoa(deathCounts[env.DeathWall]),
		strconv.Itoa(deathCounts[env.DeathSelf]),
		strconv.Itoa(deathCounts[env.DeathStall]),
		strconv.Itoa(deathCounts[env.DeathTimeout]),
	}
	l.csvWriter.Write(row)
	l.csvWriter.Flush()

	// Write JSON line
	jsonLine, _ := json.Marshal(summary)
	l.jsonFile.WriteString(string(jsonLine) + "\n")

	// Print to console
	fmt.Printf("Gen %4d | Best: %8.1f | Mean: %8.1f | Ticks: %4d | Fruits: %d | Deaths: W=%d S=%d St=%d T=%d\n",
		gen, summary.BestFitness, summary.MeanFitness, summary.BestTicks, summary.BestFruits,
		deathCounts[env.DeathWall], deathCounts[env.DeathSelf],
		deathCounts[env.DeathStall], deathCounts[env.DeathTimeout])
}

// LogBenchmark logs benchmark results
func (l *Logger) LogBenchmark(gen int, results []env.AggregatedStats) {
	if len(results) == 0 {
		return
	}

	// Average across all benchmarked agents
	var avgTicks, avgFruits float64
	for _, r := range results {
		avgTicks += r.TicksMean
		avgFruits += r.FruitsMean
	}
	avgTicks /= float64(len(results))
	avgFruits /= float64(len(results))

	fmt.Printf("  [Benchmark] Gen %d: Avg Ticks=%.1f, Avg Fruits=%.2f\n", gen, avgTicks, avgFruits)
}

// LogTopK logs debug info for top K agents
func (l *Logger) LogTopK(agents []*ga.Agent, k int) {
	if k > len(agents) {
		k = len(agents)
	}
	fmt.Printf("  Top %d agents:\n", k)
	for i := 0; i < k; i++ {
		a := agents[i]
		fmt.Printf("    #%d: Fitness=%.1f, Ticks=%d, Fruits=%d, Death=%s\n",
			i+1, a.Fitness, a.Stats.Ticks, a.Stats.Fruits, a.Stats.Death)
	}
}

// SaveChampion saves the champion genome to a file
func SaveChampion(path string, agent *ga.Agent, gen int) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	data := struct {
		Generation int       `json:"generation"`
		Fitness    float64   `json:"fitness"`
		Ticks      int       `json:"ticks"`
		Fruits     int       `json:"fruits"`
		Genome     []float32 `json:"genome"`
	}{
		Generation: gen,
		Fitness:    agent.Fitness,
		Ticks:      agent.Stats.Ticks,
		Fruits:     agent.Stats.Fruits,
		Genome:     agent.Genome,
	}

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, jsonData, 0644)
}

// LoadChampion loads a champion genome from a file
func LoadChampion(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var saved struct {
		Genome []float32 `json:"genome"`
	}
	if err := json.Unmarshal(data, &saved); err != nil {
		return nil, err
	}

	return saved.Genome, nil
}

