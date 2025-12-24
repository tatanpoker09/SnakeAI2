package ga

import (
	"math/rand"

	"snakeai/internal/nn"
)

// UniformCrossover performs uniform crossover between two parents
// Returns two children genomes
func UniformCrossover(p1, p2 []float32, rate float64, rng *rand.Rand) ([]float32, []float32) {
	size := len(p1)
	c1 := make([]float32, size)
	c2 := make([]float32, size)

	for i := 0; i < size; i++ {
		if rng.Float64() < rate {
			// Swap genes
			c1[i] = p2[i]
			c2[i] = p1[i]
		} else {
			// Keep original
			c1[i] = p1[i]
			c2[i] = p2[i]
		}
	}

	return c1, c2
}

// SinglePointCrossover performs single-point crossover
func SinglePointCrossover(p1, p2 []float32, rng *rand.Rand) ([]float32, []float32) {
	size := len(p1)
	point := rng.Intn(size)

	c1 := make([]float32, size)
	c2 := make([]float32, size)

	copy(c1[:point], p1[:point])
	copy(c1[point:], p2[point:])
	copy(c2[:point], p2[:point])
	copy(c2[point:], p1[point:])

	return c1, c2
}

// CreateChild creates a single child from two parents using crossover
func CreateChild(p1, p2 *Agent, crossoverRate float64, rng *rand.Rand) *Agent {
	if rng.Float64() > crossoverRate {
		// No crossover, clone one parent
		if rng.Float64() < 0.5 {
			return &Agent{Genome: nn.CloneGenome(p1.Genome)}
		}
		return &Agent{Genome: nn.CloneGenome(p2.Genome)}
	}

	// Uniform crossover, take only first child
	c1, _ := UniformCrossover(p1.Genome, p2.Genome, 0.5, rng)
	return &Agent{Genome: c1}
}

