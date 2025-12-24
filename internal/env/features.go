package env

// FeatureExtractor builds observation vectors for different tracks
type FeatureExtractor struct {
	obsType string
	buffer  []float32
}

// NewFeatureExtractor creates a feature extractor for the given observation type
func NewFeatureExtractor(obsType string) *FeatureExtractor {
	size := ObsDim(obsType)
	return &FeatureExtractor{
		obsType: obsType,
		buffer:  make([]float32, size),
	}
}

// ObsDim returns the observation dimension for the given type
func ObsDim(obsType string) int {
	switch obsType {
	case "wall_min":
		return 3
	case "self_min":
		return 6
	case "fruit_min":
		return 6
	case "multi_min":
		return 10
	default:
		return 3
	}
}

// Extract builds the observation vector for the current game state
// Returns a slice that should not be modified (internal buffer)
func (f *FeatureExtractor) Extract(g *Game) []float32 {
	switch f.obsType {
	case "wall_min":
		f.extractWallMin(g)
	case "self_min":
		f.extractSelfMin(g)
	case "fruit_min":
		f.extractFruitMin(g)
	case "multi_min":
		f.extractMultiMin(g)
	default:
		f.extractWallMin(g)
	}
	return f.buffer
}

// extractWallMin: 3 floats - danger_front/left/right for walls only
func (f *FeatureExtractor) extractWallMin(g *Game) {
	f.buffer[0] = boolToFloat(g.IsDangerWall(ActionStraight))
	f.buffer[1] = boolToFloat(g.IsDangerWall(ActionLeft))
	f.buffer[2] = boolToFloat(g.IsDangerWall(ActionRight))
}

// extractSelfMin: 6 floats - danger (wall OR body) + body ray distances
func (f *FeatureExtractor) extractSelfMin(g *Game) {
	// Danger signals (wall OR body)
	f.buffer[0] = boolToFloat(g.IsDanger(ActionStraight))
	f.buffer[1] = boolToFloat(g.IsDanger(ActionLeft))
	f.buffer[2] = boolToFloat(g.IsDanger(ActionRight))

	// Body distance rays (0..1, 1 if none)
	f.buffer[3] = g.BodyDistanceInDir(ActionStraight)
	f.buffer[4] = g.BodyDistanceInDir(ActionLeft)
	f.buffer[5] = g.BodyDistanceInDir(ActionRight)
}

// extractFruitMin: 6 floats - fruit direction + dangers + length
func (f *FeatureExtractor) extractFruitMin(g *Game) {
	// Fruit direction (heading-relative)
	fruitDX, fruitDY := g.FruitDirection()
	f.buffer[0] = fruitDX
	f.buffer[1] = fruitDY

	// Danger signals
	f.buffer[2] = boolToFloat(g.IsDanger(ActionStraight))
	f.buffer[3] = boolToFloat(g.IsDanger(ActionLeft))
	f.buffer[4] = boolToFloat(g.IsDanger(ActionRight))

	// Normalized length
	f.buffer[5] = g.LengthNorm()
}

// extractMultiMin: 10 floats - dangers + body rays + fruit + length
func (f *FeatureExtractor) extractMultiMin(g *Game) {
	// Danger signals (3)
	f.buffer[0] = boolToFloat(g.IsDanger(ActionStraight))
	f.buffer[1] = boolToFloat(g.IsDanger(ActionLeft))
	f.buffer[2] = boolToFloat(g.IsDanger(ActionRight))

	// Body ray distances (3)
	f.buffer[3] = g.BodyDistanceInDir(ActionStraight)
	f.buffer[4] = g.BodyDistanceInDir(ActionLeft)
	f.buffer[5] = g.BodyDistanceInDir(ActionRight)

	// Fruit direction and distance (3)
	fruitDX, fruitDY := g.FruitDirection()
	f.buffer[6] = fruitDX
	f.buffer[7] = fruitDY
	f.buffer[8] = g.FruitDistanceNorm()

	// Length (1)
	f.buffer[9] = g.LengthNorm()
}

func boolToFloat(b bool) float32 {
	if b {
		return 1.0
	}
	return 0.0
}

