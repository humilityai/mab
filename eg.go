// Copyright 2020 Humility AI Incorporated, All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package mab

import (
	"math/rand"
	"sync"

	"github.com/humilityai/sam"
)

// EpsilonGreedy is the simplest, easiest, and most "good-enough"
// multi-armed bandit optimizer to utilize.
type EpsilonGreedy struct {
	Epsilon float64
	C       sam.SliceInt
	R       sam.SliceFloat64

	sync.RWMutex
}

// NewEpsilonGreedy will create and return a new EpsilonGreedy mab optimizer.
// Epsilon value must be between 0 and 1.
// The number of options must be 2 or greater.
func NewEpsilonGreedy(options int, epsilon float64) (Optimizer, error) {
	if epsilon < 0 || epsilon > 1 {
		return nil, ErrEpsilon
	}

	if options < 2 {
		return nil, ErrOptions
	}

	return &EpsilonGreedy{
		Epsilon: epsilon,
		C:       make(sam.SliceInt, options),
		R:       make(sam.SliceFloat64, options),
	}, nil
}

// Select will select an option randomly.
func (g *EpsilonGreedy) Select() int {
	g.Lock()
	defer g.Unlock()

	if rand.Float64() > g.Epsilon {
		return g.R.MaxIndex()
	}

	return rand.Intn(len(g.R))
}

// Update should be used to increment the given option with
// the given reward amount.
func (g *EpsilonGreedy) Update(option int, reward float64) error {
	g.Lock()
	defer g.Unlock()

	if option < 0 || option > len(g.R) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	// update count
	g.C[option]++

	// update reward
	n := float64(g.C[option])
	g.R[option] = (g.R[option]*(n-1) + reward) / n

	return nil
}

// Counts returns a copy of the counts slice
func (g *EpsilonGreedy) Counts() sam.SliceInt {
	g.Lock()
	defer g.Unlock()

	s := make(sam.SliceInt, len(g.C))
	copy(s, g.C)
	return s
}

// Rewards returns a copy of the rewards slice
func (g *EpsilonGreedy) Rewards() sam.SliceFloat64 {
	g.Lock()
	defer g.Unlock()

	s := make(sam.SliceFloat64, len(g.R))
	copy(s, g.R)
	return s
}
