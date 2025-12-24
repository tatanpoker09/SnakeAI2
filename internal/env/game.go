package env

import (
	"math/rand"
)

// Direction represents the snake's heading
type Direction int

const (
	DirUp Direction = iota
	DirRight
	DirDown
	DirLeft
)

// Action represents a relative action
type Action int

const (
	ActionStraight Action = iota
	ActionLeft
	ActionRight
)

// Point represents a coordinate on the grid
type Point struct {
	X, Y int
}

// Game represents the snake game environment
type Game struct {
	Width       int
	Height      int
	TickCap     int
	StallWindow int
	FruitEnabled bool

	// State
	Snake        []Point   // head is at index 0
	Dir          Direction
	Fruit        Point
	Tick         int
	TicksNoFruit int
	FruitsEaten  int
	Alive        bool
	DeathReason  DeathReason
	ProgressSum  float64
	LastFruitDist float64

	rng *rand.Rand
}

// NewGame creates a new game instance
func NewGame(width, height, startLength, tickCap, stallWindow int, fruitEnabled bool, seed uint32) *Game {
	g := &Game{
		Width:        width,
		Height:       height,
		TickCap:      tickCap,
		StallWindow:  stallWindow,
		FruitEnabled: fruitEnabled,
		rng:          rand.New(rand.NewSource(int64(seed))),
	}
	g.Reset(startLength)
	return g
}

// Reset initializes the game to starting state
func (g *Game) Reset(startLength int) {
	g.Tick = 0
	g.TicksNoFruit = 0
	g.FruitsEaten = 0
	g.Alive = true
	g.DeathReason = DeathNone
	g.ProgressSum = 0
	g.LastFruitDist = 0

	// Spawn snake in center, facing right
	centerX := g.Width / 2
	centerY := g.Height / 2
	g.Dir = DirRight

	g.Snake = make([]Point, startLength)
	for i := 0; i < startLength; i++ {
		g.Snake[i] = Point{X: centerX - i, Y: centerY}
	}

	// Spawn fruit
	if g.FruitEnabled {
		g.spawnFruit()
		g.LastFruitDist = g.distanceToFruit()
	}
}

// Step advances the game by one tick with the given action
func (g *Game) Step(action Action) {
	if !g.Alive {
		return
	}

	g.Tick++
	g.TicksNoFruit++

	// Turn based on relative action
	g.Dir = g.applyTurn(action)

	// Move head
	head := g.Snake[0]
	newHead := g.moveInDirection(head, g.Dir)

	// Check wall collision
	if newHead.X < 0 || newHead.X >= g.Width || newHead.Y < 0 || newHead.Y >= g.Height {
		g.Alive = false
		g.DeathReason = DeathWall
		return
	}

	// Check self collision (excluding tail which will move)
	for i := 0; i < len(g.Snake)-1; i++ {
		if g.Snake[i] == newHead {
			g.Alive = false
			g.DeathReason = DeathSelf
			return
		}
	}

	// Check fruit
	ateFruit := g.FruitEnabled && newHead == g.Fruit

	// Update snake body
	if ateFruit {
		// Grow: don't remove tail
		g.Snake = append([]Point{newHead}, g.Snake...)
		g.FruitsEaten++
		g.TicksNoFruit = 0
		g.spawnFruit()
		g.LastFruitDist = g.distanceToFruit()
	} else {
		// Move: shift body
		g.Snake = append([]Point{newHead}, g.Snake[:len(g.Snake)-1]...)
		
		// Track progress toward fruit
		if g.FruitEnabled {
			newDist := g.distanceToFruit()
			improvement := g.LastFruitDist - newDist
			if improvement > 0 {
				g.ProgressSum += improvement
			}
			g.LastFruitDist = newDist
		}
	}

	// Check stall
	if g.TicksNoFruit >= g.StallWindow {
		g.Alive = false
		g.DeathReason = DeathStall
		return
	}

	// Check tick cap
	if g.Tick >= g.TickCap {
		g.Alive = false
		g.DeathReason = DeathTimeout
		return
	}
}

// applyTurn returns new direction after applying relative action
func (g *Game) applyTurn(action Action) Direction {
	switch action {
	case ActionLeft:
		return Direction((g.Dir + 3) % 4) // turn left
	case ActionRight:
		return Direction((g.Dir + 1) % 4) // turn right
	default:
		return g.Dir // straight
	}
}

// moveInDirection returns the new point after moving in direction
func (g *Game) moveInDirection(p Point, dir Direction) Point {
	switch dir {
	case DirUp:
		return Point{X: p.X, Y: p.Y - 1}
	case DirRight:
		return Point{X: p.X + 1, Y: p.Y}
	case DirDown:
		return Point{X: p.X, Y: p.Y + 1}
	case DirLeft:
		return Point{X: p.X - 1, Y: p.Y}
	}
	return p
}

// spawnFruit places fruit at a random empty cell
func (g *Game) spawnFruit() {
	// Build set of occupied cells
	occupied := make(map[Point]bool)
	for _, p := range g.Snake {
		occupied[p] = true
	}

	// Find all empty cells
	var empty []Point
	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			p := Point{X: x, Y: y}
			if !occupied[p] {
				empty = append(empty, p)
			}
		}
	}

	if len(empty) > 0 {
		g.Fruit = empty[g.rng.Intn(len(empty))]
	}
}

// distanceToFruit returns Manhattan distance from head to fruit
func (g *Game) distanceToFruit() float64 {
	head := g.Snake[0]
	dx := head.X - g.Fruit.X
	dy := head.Y - g.Fruit.Y
	if dx < 0 {
		dx = -dx
	}
	if dy < 0 {
		dy = -dy
	}
	return float64(dx + dy)
}

// Head returns the snake's head position
func (g *Game) Head() Point {
	return g.Snake[0]
}

// Tail returns the snake's tail position
func (g *Game) Tail() Point {
	return g.Snake[len(g.Snake)-1]
}

// Stats returns the episode statistics
func (g *Game) Stats(seed uint32) EpisodeStats {
	return EpisodeStats{
		Fruits:      g.FruitsEaten,
		Ticks:       g.Tick,
		ProgressSum: g.ProgressSum,
		Death:       g.DeathReason,
		Seed:        seed,
	}
}

// IsDangerWall checks if moving in direction would hit wall
func (g *Game) IsDangerWall(relDir Action) bool {
	newDir := g.applyTurn(relDir)
	newPos := g.moveInDirection(g.Snake[0], newDir)
	return newPos.X < 0 || newPos.X >= g.Width || newPos.Y < 0 || newPos.Y >= g.Height
}

// IsDangerBody checks if moving in direction would hit body
func (g *Game) IsDangerBody(relDir Action) bool {
	newDir := g.applyTurn(relDir)
	newPos := g.moveInDirection(g.Snake[0], newDir)
	// Check all but tail (it will move)
	for i := 0; i < len(g.Snake)-1; i++ {
		if g.Snake[i] == newPos {
			return true
		}
	}
	return false
}

// IsDanger checks if moving in direction would cause any collision
func (g *Game) IsDanger(relDir Action) bool {
	return g.IsDangerWall(relDir) || g.IsDangerBody(relDir)
}

// BodyDistanceInDir returns normalized distance to body in relative direction (0..1, 1 if none)
func (g *Game) BodyDistanceInDir(relDir Action) float32 {
	newDir := g.applyTurn(relDir)
	head := g.Snake[0]
	maxDist := float32(g.Width + g.Height) // max possible

	// Cast ray in direction
	for dist := 1; dist < g.Width+g.Height; dist++ {
		var checkPos Point
		switch newDir {
		case DirUp:
			checkPos = Point{X: head.X, Y: head.Y - dist}
		case DirRight:
			checkPos = Point{X: head.X + dist, Y: head.Y}
		case DirDown:
			checkPos = Point{X: head.X, Y: head.Y + dist}
		case DirLeft:
			checkPos = Point{X: head.X - dist, Y: head.Y}
		}

		// Out of bounds
		if checkPos.X < 0 || checkPos.X >= g.Width || checkPos.Y < 0 || checkPos.Y >= g.Height {
			return 1.0
		}

		// Check body collision
		for _, p := range g.Snake {
			if p == checkPos {
				return float32(dist) / maxDist
			}
		}
	}
	return 1.0
}

// FruitDirection returns (dx, dy) normalized to [-1, 1] in heading-relative frame
func (g *Game) FruitDirection() (float32, float32) {
	if !g.FruitEnabled {
		return 0, 0
	}

	head := g.Snake[0]
	// World-space delta
	dx := float32(g.Fruit.X - head.X)
	dy := float32(g.Fruit.Y - head.Y)

	// Normalize by grid size
	maxD := float32(g.Width + g.Height)
	dx /= maxD
	dy /= maxD

	// Rotate to heading-relative frame
	// Heading: Up=0, Right=1, Down=2, Left=3
	// We want: positive Y = forward (in front), positive X = right
	switch g.Dir {
	case DirUp:
		return dx, -dy // up is -y in world, so negate
	case DirRight:
		return -dy, dx
	case DirDown:
		return -dx, dy
	case DirLeft:
		return dy, -dx
	}
	return dx, dy
}

// FruitDistanceNorm returns normalized distance to fruit
func (g *Game) FruitDistanceNorm() float32 {
	if !g.FruitEnabled {
		return 1.0
	}
	maxDist := float32(g.Width + g.Height)
	return float32(g.distanceToFruit()) / maxDist
}

// LengthNorm returns normalized snake length
func (g *Game) LengthNorm() float32 {
	maxLen := float32(g.Width * g.Height)
	return float32(len(g.Snake)) / maxLen
}

// TailDirection returns normalized (dx, dy) to tail
func (g *Game) TailDirection() (float32, float32) {
	head := g.Snake[0]
	tail := g.Tail()
	dx := float32(tail.X - head.X)
	dy := float32(tail.Y - head.Y)
	maxD := float32(g.Width + g.Height)
	return dx / maxD, dy / maxD
}

