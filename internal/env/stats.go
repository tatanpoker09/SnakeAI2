package env

import "math"

// DeathReason indicates how the snake died
type DeathReason int

const (
	DeathNone    DeathReason = iota
	DeathWall                // hit a wall
	DeathSelf                // hit own body
	DeathStall               // no fruit for too long
	DeathTimeout             // tick cap reached
)

func (d DeathReason) String() string {
	switch d {
	case DeathNone:
		return "none"
	case DeathWall:
		return "wall"
	case DeathSelf:
		return "self"
	case DeathStall:
		return "stall"
	case DeathTimeout:
		return "timeout"
	default:
		return "unknown"
	}
}

// EpisodeStats captures all metrics from a single episode
type EpisodeStats struct {
	Score       float64     // computed fitness score
	Fruits      int         // number of fruits eaten
	Ticks       int         // number of ticks survived
	ProgressSum float64     // cumulative distance improvement
	Death       DeathReason // how the episode ended
	Seed        uint32      // seed used for this episode
}

// AggregatedStats holds statistics across multiple episodes
type AggregatedStats struct {
	ScoreMean    float64
	ScoreStd     float64
	FruitsMean   float64
	TicksMean    float64
	ProgressMean float64
	DeathCounts  map[DeathReason]int
	NumEpisodes  int
}

// Aggregate computes statistics from multiple episode stats
func Aggregate(episodes []EpisodeStats) AggregatedStats {
	n := len(episodes)
	if n == 0 {
		return AggregatedStats{DeathCounts: make(map[DeathReason]int)}
	}

	agg := AggregatedStats{
		DeathCounts: make(map[DeathReason]int),
		NumEpisodes: n,
	}

	var scoreSum, fruitsSum, ticksSum, progressSum float64
	for _, ep := range episodes {
		scoreSum += ep.Score
		fruitsSum += float64(ep.Fruits)
		ticksSum += float64(ep.Ticks)
		progressSum += ep.ProgressSum
		agg.DeathCounts[ep.Death]++
	}

	nf := float64(n)
	agg.ScoreMean = scoreSum / nf
	agg.FruitsMean = fruitsSum / nf
	agg.TicksMean = ticksSum / nf
	agg.ProgressMean = progressSum / nf

	// Compute standard deviation for score
	var variance float64
	for _, ep := range episodes {
		diff := ep.Score - agg.ScoreMean
		variance += diff * diff
	}
	agg.ScoreStd = math.Sqrt(variance / nf)

	return agg
}

// RobustnessScore computes the ranking score: mean - lambda * std
func (a AggregatedStats) RobustnessScore(lambda float64) float64 {
	return a.ScoreMean - lambda*a.ScoreStd
}

