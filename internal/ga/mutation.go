package ga

import (
	"math/rand"
)

// Mutate applies Gaussian mutation to a genome in-place
func Mutate(genome []float32, rate, sigma float64, rng *rand.Rand) {
	for i := range genome {
		if rng.Float64() < rate {
			genome[i] += float32(rng.NormFloat64() * sigma)
		}
	}
}

// MutateWithReset applies mutation with occasional random reset
func MutateWithReset(genome []float32, rate, sigma, resetP float64, rng *rand.Rand) {
	for i := range genome {
		if rng.Float64() < resetP {
			// Random reset
			genome[i] = float32(rng.NormFloat64() * 0.5)
		} else if rng.Float64() < rate {
			// Gaussian perturbation
			genome[i] += float32(rng.NormFloat64() * sigma)
		}
	}
}

// MutateAgent applies mutation to an agent's genome
func MutateAgent(a *Agent, rate, sigma, resetP float64, rng *rand.Rand) {
	MutateWithReset(a.Genome, rate, sigma, resetP, rng)
}

// ResetFraction randomly reinitializes a fraction of the population
func ResetFraction(pop *Population, fraction float64, rng *rand.Rand) {
	numReset := int(float64(len(pop.Agents)) * fraction)
	
	// Reset the worst agents (assume sorted by fitness descending)
	pop.SortByFitness()
	
	for i := len(pop.Agents) - numReset; i < len(pop.Agents); i++ {
		if i >= 0 {
			for j := range pop.Agents[i].Genome {
				pop.Agents[i].Genome[j] = float32(rng.NormFloat64() * 0.5)
			}
		}
	}
}

