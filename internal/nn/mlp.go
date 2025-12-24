package nn

import (
	"math"
	"math/rand"
)

// MLP is a simple feedforward neural network with float32 weights
type MLP struct {
	InputSize  int
	Hidden1    int
	Hidden2    int // 0 means no second hidden layer
	OutputSize int

	// Weights stored contiguously
	Weights []float32

	// Pre-allocated buffers for forward pass (no allocations in hot path)
	h1  []float32
	h2  []float32
	out []float32
}

// NewMLP creates a new MLP with the given architecture
func NewMLP(inputSize, hidden1, hidden2, outputSize int) *MLP {
	m := &MLP{
		InputSize:  inputSize,
		Hidden1:    hidden1,
		Hidden2:    hidden2,
		OutputSize: outputSize,
	}

	// Calculate total weights needed
	totalWeights := m.GenomeSize()
	m.Weights = make([]float32, totalWeights)

	// Allocate forward pass buffers
	m.h1 = make([]float32, hidden1)
	if hidden2 > 0 {
		m.h2 = make([]float32, hidden2)
	}
	m.out = make([]float32, outputSize)

	return m
}

// GenomeSize returns the total number of weights (including biases)
func (m *MLP) GenomeSize() int {
	size := 0
	// Input -> Hidden1 (weights + biases)
	size += (m.InputSize + 1) * m.Hidden1

	if m.Hidden2 > 0 {
		// Hidden1 -> Hidden2
		size += (m.Hidden1 + 1) * m.Hidden2
		// Hidden2 -> Output
		size += (m.Hidden2 + 1) * m.OutputSize
	} else {
		// Hidden1 -> Output
		size += (m.Hidden1 + 1) * m.OutputSize
	}
	return size
}

// SetWeights copies genome into the network weights
func (m *MLP) SetWeights(genome []float32) {
	copy(m.Weights, genome)
}

// Forward performs a forward pass and returns the output index with max value
func (m *MLP) Forward(input []float32) int {
	offset := 0

	// Input -> Hidden1
	for j := 0; j < m.Hidden1; j++ {
		sum := m.Weights[offset] // bias
		offset++
		for i := 0; i < m.InputSize; i++ {
			sum += input[i] * m.Weights[offset]
			offset++
		}
		m.h1[j] = relu(sum)
	}

	var lastHidden []float32

	if m.Hidden2 > 0 {
		// Hidden1 -> Hidden2
		for j := 0; j < m.Hidden2; j++ {
			sum := m.Weights[offset] // bias
			offset++
			for i := 0; i < m.Hidden1; i++ {
				sum += m.h1[i] * m.Weights[offset]
				offset++
			}
			m.h2[j] = relu(sum)
		}
		lastHidden = m.h2
	} else {
		lastHidden = m.h1
	}

	// Last hidden -> Output
	hiddenSize := len(lastHidden)
	for j := 0; j < m.OutputSize; j++ {
		sum := m.Weights[offset] // bias
		offset++
		for i := 0; i < hiddenSize; i++ {
			sum += lastHidden[i] * m.Weights[offset]
			offset++
		}
		m.out[j] = sum // no activation on output
	}

	// Return argmax
	return argmax(m.out)
}

// ForwardRaw performs forward pass and returns raw output values
func (m *MLP) ForwardRaw(input []float32) []float32 {
	m.Forward(input)
	result := make([]float32, m.OutputSize)
	copy(result, m.out)
	return result
}

func relu(x float32) float32 {
	if x > 0 {
		return x
	}
	return 0
}

func argmax(vals []float32) int {
	maxIdx := 0
	maxVal := vals[0]
	for i := 1; i < len(vals); i++ {
		if vals[i] > maxVal {
			maxVal = vals[i]
			maxIdx = i
		}
	}
	return maxIdx
}

// RandomGenome generates a random genome for the network
func RandomGenome(size int, rng *rand.Rand) []float32 {
	genome := make([]float32, size)
	// Xavier-like initialization
	scale := float32(math.Sqrt(2.0 / float64(size)))
	for i := range genome {
		genome[i] = float32(rng.NormFloat64()) * scale
	}
	return genome
}

// CloneGenome makes a copy of a genome
func CloneGenome(src []float32) []float32 {
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}

