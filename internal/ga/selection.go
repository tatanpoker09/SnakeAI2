package ga

import (
	"math/rand"
)

// TournamentSelect selects an agent using tournament selection
func TournamentSelect(agents []*Agent, k int, rng *rand.Rand) *Agent {
	if len(agents) == 0 {
		return nil
	}
	if k > len(agents) {
		k = len(agents)
	}

	best := agents[rng.Intn(len(agents))]
	for i := 1; i < k; i++ {
		candidate := agents[rng.Intn(len(agents))]
		if candidate.Fitness > best.Fitness {
			best = candidate
		}
	}
	return best
}

// TournamentSelectFromPool selects from a pre-filtered pool
func TournamentSelectFromPool(pool []*Agent, k int, rng *rand.Rand) *Agent {
	return TournamentSelect(pool, k, rng)
}

// SelectionPool returns the top agents to form the mating pool
func SelectionPool(pop *Population, poolSize int) []*Agent {
	pop.SortByFitness()
	if poolSize > len(pop.Agents) {
		poolSize = len(pop.Agents)
	}
	return pop.Agents[:poolSize]
}

// SelectParents selects two parents from the pool using tournament selection
func SelectParents(pool []*Agent, k int, rng *rand.Rand) (*Agent, *Agent) {
	p1 := TournamentSelect(pool, k, rng)
	p2 := TournamentSelect(pool, k, rng)
	return p1, p2
}

