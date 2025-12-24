package ga

import (
	"math/rand"
	"sort"

	"snakeai/internal/env"
	"snakeai/internal/nn"
)

// Agent represents an individual in the population
type Agent struct {
	Genome  []float32
	Fitness float64
	Stats   env.EpisodeStats
	AggStats env.AggregatedStats // for multi-seed evaluation
	RobustScore float64 // mean - lambda*std
}

// Population manages the collection of agents
type Population struct {
	Agents     []*Agent
	GenomeSize int
	rng        *rand.Rand
}

// NewPopulation creates a new random population
func NewPopulation(size, genomeSize int, rng *rand.Rand) *Population {
	p := &Population{
		Agents:     make([]*Agent, size),
		GenomeSize: genomeSize,
		rng:        rng,
	}

	for i := 0; i < size; i++ {
		p.Agents[i] = &Agent{
			Genome: nn.RandomGenome(genomeSize, rng),
		}
	}

	return p
}

// Size returns the population size
func (p *Population) Size() int {
	return len(p.Agents)
}

// SortByFitness sorts agents by fitness (descending)
func (p *Population) SortByFitness() {
	sort.Slice(p.Agents, func(i, j int) bool {
		return p.Agents[i].Fitness > p.Agents[j].Fitness
	})
}

// SortByRobustScore sorts agents by robustness score (descending)
func (p *Population) SortByRobustScore() {
	sort.Slice(p.Agents, func(i, j int) bool {
		return p.Agents[i].RobustScore > p.Agents[j].RobustScore
	})
}

// TopK returns the top K agents by fitness
func (p *Population) TopK(k int) []*Agent {
	p.SortByFitness()
	if k > len(p.Agents) {
		k = len(p.Agents)
	}
	return p.Agents[:k]
}

// Best returns the agent with highest fitness
func (p *Population) Best() *Agent {
	if len(p.Agents) == 0 {
		return nil
	}
	best := p.Agents[0]
	for _, a := range p.Agents[1:] {
		if a.Fitness > best.Fitness {
			best = a
		}
	}
	return best
}

// BestByRobust returns the agent with highest robustness score
func (p *Population) BestByRobust() *Agent {
	if len(p.Agents) == 0 {
		return nil
	}
	best := p.Agents[0]
	for _, a := range p.Agents[1:] {
		if a.RobustScore > best.RobustScore {
			best = a
		}
	}
	return best
}

// Clone creates a deep copy of an agent
func (a *Agent) Clone() *Agent {
	return &Agent{
		Genome:      nn.CloneGenome(a.Genome),
		Fitness:     a.Fitness,
		Stats:       a.Stats,
		AggStats:    a.AggStats,
		RobustScore: a.RobustScore,
	}
}

// Replace replaces population with new agents, preserving elites
func (p *Population) Replace(newAgents []*Agent, elites int) {
	// Sort current population by fitness
	p.SortByFitness()

	// Keep elites
	for i := 0; i < elites && i < len(p.Agents); i++ {
		newAgents[i] = p.Agents[i].Clone()
	}

	p.Agents = newAgents
}

// GetRNG returns the population's random number generator
func (p *Population) GetRNG() *rand.Rand {
	return p.rng
}

// ResetFitness resets all agents' fitness to 0
func (p *Population) ResetFitness() {
	for _, a := range p.Agents {
		a.Fitness = 0
		a.RobustScore = 0
	}
}

