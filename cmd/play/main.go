package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"runtime"
	"time"

	"snakeai/internal/config"
	"snakeai/internal/env"
	"snakeai/internal/nn"
)

// ChampionData represents saved champion format
type ChampionData struct {
	Generation int       `json:"generation"`
	Fitness    float64   `json:"fitness"`
	Ticks      int       `json:"ticks"`
	Fruits     int       `json:"fruits"`
	Genome     []float32 `json:"genome"`
}

func main() {
	// Parse flags
	configPath := flag.String("config", "configs/wall.yaml", "path to config file")
	championPath := flag.String("champion", "artifacts/champion_final.json", "path to champion JSON")
	seed := flag.Uint("seed", 12345, "random seed for the game")
	delay := flag.Int("delay", 100, "delay between frames in milliseconds")
	noDisplay := flag.Bool("no-display", false, "run without display (just print stats)")
	noTimeout := flag.Bool("no-timeout", false, "disable tick cap (play until death)")
	noStall := flag.Bool("no-stall", false, "disable stall detection")
	flag.Parse()

	// Load config
	cfg, err := config.Load(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading config: %v\n", err)
		os.Exit(1)
	}

	// Override timeout settings if requested
	if *noTimeout {
		cfg.Env.TickCap = 999999
	}
	if *noStall {
		cfg.Env.StallWindow = 999999
	}

	// Load champion
	champion, err := loadChampion(*championPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error loading champion: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Loaded champion from gen %d (fitness=%.1f, ticks=%d, fruits=%d)\n",
		champion.Generation, champion.Fitness, champion.Ticks, champion.Fruits)
	fmt.Printf("Config: %s, Seed: %d\n", *configPath, *seed)
	fmt.Println("Press Ctrl+C to exit")
	fmt.Println()

	// Create game
	game := env.NewGame(
		cfg.Env.Width,
		cfg.Env.Height,
		cfg.Env.StartLength,
		cfg.Env.TickCap,
		cfg.Env.StallWindow,
		cfg.Env.FruitEnabled,
		uint32(*seed),
	)

	// Create neural network
	mlp := nn.NewMLP(cfg.ObsDim(), cfg.NN.Hidden1, cfg.NN.Hidden2, 3)
	mlp.SetWeights(champion.Genome)

	// Create feature extractor
	features := env.NewFeatureExtractor(cfg.Track.Obs)

	// Display helper
	display := NewDisplay(cfg.Env.Width, cfg.Env.Height)

	// Run game loop
	frameDelay := time.Duration(*delay) * time.Millisecond
	
	for game.Alive {
		// Get observation and action
		obs := features.Extract(game)
		action := mlp.Forward(obs)

		// Display current state
		if !*noDisplay {
			display.Render(game, action)
			time.Sleep(frameDelay)
		}

		// Step game
		game.Step(env.Action(action))
	}

	// Final display
	if !*noDisplay {
		display.Render(game, -1)
	}

	// Print final stats
	stats := game.Stats(uint32(*seed))
	fmt.Println()
	fmt.Println("═══════════════════════════════════")
	fmt.Printf("  Game Over! Death: %s\n", stats.Death)
	fmt.Printf("  Ticks: %d, Fruits: %d\n", stats.Ticks, stats.Fruits)
	fmt.Printf("  Progress Sum: %.2f\n", stats.ProgressSum)
	fmt.Println("═══════════════════════════════════")
}

func loadChampion(path string) (*ChampionData, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var champion ChampionData
	if err := json.Unmarshal(data, &champion); err != nil {
		return nil, err
	}
	return &champion, nil
}

// Display handles terminal rendering
type Display struct {
	width  int
	height int
}

// NewDisplay creates a new display
func NewDisplay(width, height int) *Display {
	return &Display{width: width, height: height}
}

// Render draws the game state to terminal
func (d *Display) Render(game *env.Game, action int) {
	clearScreen()

	// Build grid
	grid := make([][]rune, d.height)
	for y := 0; y < d.height; y++ {
		grid[y] = make([]rune, d.width)
		for x := 0; x < d.width; x++ {
			grid[y][x] = '·'
		}
	}

	// Place fruit
	if game.FruitEnabled {
		fruit := game.Fruit
		if fruit.X >= 0 && fruit.X < d.width && fruit.Y >= 0 && fruit.Y < d.height {
			grid[fruit.Y][fruit.X] = '🍎'
		}
	}

	// Place snake body
	for i := len(game.Snake) - 1; i >= 0; i-- {
		p := game.Snake[i]
		if p.X >= 0 && p.X < d.width && p.Y >= 0 && p.Y < d.height {
			if i == 0 {
				// Head - show direction
				grid[p.Y][p.X] = directionHead(game.Dir)
			} else {
				grid[p.Y][p.X] = '█'
			}
		}
	}

	// Draw border and grid
	fmt.Print("┌")
	for x := 0; x < d.width; x++ {
		fmt.Print("──")
	}
	fmt.Println("┐")

	for y := 0; y < d.height; y++ {
		fmt.Print("│")
		for x := 0; x < d.width; x++ {
			c := grid[y][x]
			if c == '🍎' {
				fmt.Print("🍎")
			} else if c == '·' {
				fmt.Print(" ·")
			} else {
				fmt.Printf(" %c", c)
			}
		}
		fmt.Println("│")
	}

	fmt.Print("└")
	for x := 0; x < d.width; x++ {
		fmt.Print("──")
	}
	fmt.Println("┘")

	// Status line
	actionStr := []string{"STRAIGHT", "LEFT", "RIGHT"}
	actionDisplay := "---"
	if action >= 0 && action < 3 {
		actionDisplay = actionStr[action]
	}

	fmt.Printf("  Tick: %3d | Fruits: %d | Length: %d | Action: %s\n",
		game.Tick, game.FruitsEaten, len(game.Snake), actionDisplay)

	if !game.Alive {
		fmt.Printf("  💀 DEAD: %s\n", game.DeathReason)
	}
}

func directionHead(dir env.Direction) rune {
	switch dir {
	case env.DirUp:
		return '▲'
	case env.DirRight:
		return '▶'
	case env.DirDown:
		return '▼'
	case env.DirLeft:
		return '◀'
	}
	return 'O'
}

func clearScreen() {
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.Command("cmd", "/c", "cls")
	} else {
		cmd = exec.Command("clear")
	}
	cmd.Stdout = os.Stdout
	cmd.Run()
}

