package env

import (
	"encoding/json"
	"os"
)

// Replay stores a deterministic action trace for playback
type Replay struct {
	Seed        uint32   `json:"seed"`
	Actions     []Action `json:"actions"`
	FinalStats  EpisodeStats `json:"final_stats"`
	Config      ReplayConfig `json:"config"`
}

// ReplayConfig stores environment config for replay
type ReplayConfig struct {
	Width        int  `json:"width"`
	Height       int  `json:"height"`
	StartLength  int  `json:"start_length"`
	TickCap      int  `json:"tick_cap"`
	StallWindow  int  `json:"stall_window"`
	FruitEnabled bool `json:"fruit_enabled"`
}

// NewReplay creates a new replay recorder
func NewReplay(seed uint32, config ReplayConfig) *Replay {
	return &Replay{
		Seed:    seed,
		Actions: make([]Action, 0, 256),
		Config:  config,
	}
}

// Record adds an action to the replay
func (r *Replay) Record(action Action) {
	r.Actions = append(r.Actions, action)
}

// SetFinalStats sets the final episode statistics
func (r *Replay) SetFinalStats(stats EpisodeStats) {
	r.FinalStats = stats
}

// Save writes the replay to a file
func (r *Replay) Save(path string) error {
	data, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// LoadReplay loads a replay from a file
func LoadReplay(path string) (*Replay, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Replay
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// Playback recreates the game from the replay
func (r *Replay) Playback() *Game {
	g := NewGame(
		r.Config.Width,
		r.Config.Height,
		r.Config.StartLength,
		r.Config.TickCap,
		r.Config.StallWindow,
		r.Config.FruitEnabled,
		r.Seed,
	)
	return g
}

// PlaybackStep runs the replay up to step n
func (r *Replay) PlaybackStep(g *Game, step int) {
	if step > len(r.Actions) {
		step = len(r.Actions)
	}
	for i := 0; i < step && g.Alive; i++ {
		g.Step(r.Actions[i])
	}
}

