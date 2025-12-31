# Snake AI - Genetic Algorithm Training System

A Go-based Snake AI that learns to play through genetic algorithms. Features 4 specialized training tracks, minimal neural network observations, and real-time visualization.

## Features

- **4 Training Tracks**: Progressive difficulty from wall avoidance to full gameplay
- **Genetic Algorithm**: Tournament selection, uniform crossover, Gaussian mutation
- **Minimal Observations**: 3-10 input features per track (heading-relative)
- **Fast Training**: Float32 inference, parallel evaluation
- **Robustness Ranking**: Multi-seed evaluation prevents lucky agents
- **Live Visualization**: Watch trained agents play in the terminal

## Requirements

- Go 1.21 or higher
- Make (optional, for convenience commands)

## Quick Start

```bash
# Clone and enter the project
cd SnakeAI3

# Train a fruit-eating agent (1000 generations, ~1 minute)
make train-fruit

# Watch it play!
make play-fruit
```

## Installation

```bash
# Download dependencies
go mod tidy

# Build both binaries
make build
```

## Training

Train agents on different tracks with increasing complexity:

```bash
# Track 1: Wall Avoidance (simplest - 3 inputs)
make train-wall

# Track 2: Self-Collision Avoidance (6 inputs, longer snake)
make train-self

# Track 3: Fruit Collection (6 inputs, must eat fruit)
make train-fruit

# Track 4: Full Game (10 inputs, everything combined)
make train-multi
```

### Custom Training

```bash
# Run with custom generations
./bin/train -config configs/fruit.yaml -generations 500

# Use a different config
./bin/train -config configs/multi.yaml -generations 2000
```

### Training Output

- **Console**: Real-time progress with fitness, ticks, fruits, and death counts
- **CSV Log**: `runs/<track>_run.csv` - Per-generation statistics
- **JSON Log**: `runs/<track>_run.jsonl` - Detailed metrics
- **Champions**: `artifacts/champion_final.json` - Best agent genome

## Playing / Visualization

Watch a trained agent play in real-time:

```bash
# Play with the last trained champion
make play-fruit

# Or use the binary directly with options
./bin/play -config configs/fruit.yaml -champion artifacts/champion_final.json
```

### Play Options

```bash
./bin/play [options]
  -config <file>      Config file (default: configs/wall.yaml)
  -champion <file>    Champion JSON file (default: artifacts/champion_final.json)
  -seed <int>         Random seed for game (default: 12345)
  -delay <ms>         Frame delay in milliseconds (default: 100)
  -no-timeout         Disable tick limit, play until death
  -no-stall           Disable stall detection
  -no-display         Run without visualization, print stats only
```

### Example

```bash
# Watch at faster speed with a specific seed
./bin/play -config configs/fruit.yaml -no-timeout -delay 50 -seed 42
```

### Display Legend

```
┌────────────────────┐
│🍎 · · · · · · · · ·│   🍎 = Fruit
│ · · · █ █ █ █ █ █ █│   █  = Snake body
│ · · · █ ▲ · · · · █│   ▲  = Snake head (facing up)
│ · · · █ █ · · · · █│   ▶  = Head facing right
│ · · · · · · · · · ·│   ▼  = Head facing down
│ · · · · · · · · · ·│   ◀  = Head facing left
└────────────────────┘   ·  = Empty cell
  Tick: 110 | Fruits: 12 | Length: 15 | Action: STRAIGHT
```

## Training Tracks

| Track | Obs Dim | Description | Goal |
|-------|---------|-------------|------|
| `wall` | 3 | Wall danger sensors only | Survive by avoiding walls |
| `self` | 6 | Wall + body sensors | Avoid walls and own body |
| `fruit` | 6 | Fruit direction + dangers | Collect fruit efficiently |
| `multi` | 10 | Full observation set | Master all behaviors |

## Configuration

Edit YAML files in `configs/` to customize:

```yaml
# configs/fruit.yaml
env:
  width: 10           # Grid width
  height: 10          # Grid height
  start_length: 3     # Initial snake length
  tick_cap: 150       # Max ticks per episode
  stall_window: 40    # Ticks without fruit = stall death
  fruit_enabled: true # Enable fruit spawning

nn:
  hidden1: 16         # Hidden layer size
  hidden2: 0          # Second hidden layer (0 = none)

ga:
  population: 200     # Population size
  elites: 4           # Top agents preserved each generation
  mutation_rate: 0.10 # Probability of mutating each weight
  mutation_sigma: 0.06 # Mutation strength (std dev)
```

## Project Structure

```
SnakeAI3/
├── cmd/
│   ├── train/main.go      # Training entry point
│   └── play/main.go       # Visualization entry point
├── internal/
│   ├── config/            # YAML configuration
│   ├── env/               # Game environment
│   │   ├── game.go        # Snake game logic
│   │   ├── features.go    # Observation extraction
│   │   ├── stats.go       # Episode statistics
│   │   └── replay.go      # Action recording
│   ├── nn/mlp.go          # Neural network
│   ├── ga/                # Genetic algorithm
│   │   ├── population.go  # Agent management
│   │   ├── selection.go   # Tournament selection
│   │   ├── crossover.go   # Uniform crossover
│   │   └── mutation.go    # Gaussian mutation
│   ├── eval/evaluator.go  # Fitness evaluation
│   └── logging/metrics.go # CSV/JSON logging
├── configs/               # Track configurations
│   ├── wall.yaml
│   ├── self.yaml
│   ├── fruit.yaml
│   └── multi.yaml
├── artifacts/             # Saved champions
├── runs/                  # Training logs
├── Makefile
└── README.md
```

## How It Works

1. **Population**: Random neural networks are created
2. **Evaluation**: Each agent plays Snake, fitness = fruits + survival
3. **Selection**: Top performers are chosen via tournament
4. **Crossover**: Parents' weights are mixed to create children
5. **Mutation**: Small random changes add diversity
6. **Repeat**: Process continues for N generations

The neural network uses:
- **Inputs**: Danger sensors + fruit direction (heading-relative)
- **Hidden Layer**: 8-24 neurons with ReLU activation
- **Outputs**: 3 actions (straight, turn left, turn right)

## Tips

- **Start with `wall`**: Verify the system works before complex tracks
- **Watch early generations**: Use `make play-fruit` to see learning progress
- **Increase generations**: More training = better performance
- **Try different seeds**: `-seed 42` may give better/worse runs
- **Adjust delay**: `-delay 50` for faster visualization

## Makefile Targets

```bash
make build        # Build train and play binaries
make train-wall   # Train wall avoidance
make train-self   # Train self-collision avoidance
make train-fruit  # Train fruit collection
make train-multi  # Train full game
make play         # Play with default config
make play-wall    # Play wall-trained model
make play-self    # Play self-trained model
make play-fruit   # Play fruit-trained model
make play-multi   # Play multi-trained model
make clean        # Remove binaries and artifacts
```

## License

MIT

