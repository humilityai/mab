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
	epsilon float64
	c       sam.SliceInt
	r       sam.SliceFloat64

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
		epsilon: epsilon,
		c:       make(sam.SliceInt, options),
		r:       make(sam.SliceFloat64, options),
	}, nil
}

// Counts returns a copy of the counts slice
func (e *EpsilonGreedy) Counts() sam.SliceInt {
	e.Lock()
	defer e.Unlock()

	s := make(sam.SliceInt, len(e.c))
	copy(s, e.c)
	return s
}

// Remove --
func (e *EpsilonGreedy) Remove(option int) {
	if option < 0 || option > len(e.c)-1 {
		return
	}

	e.c[option] = -1
	e.r[option] = -1.0
}

// Rewards returns a copy of the rewards slice
func (e *EpsilonGreedy) Rewards() sam.SliceFloat64 {
	e.Lock()
	defer e.Unlock()

	s := make(sam.SliceFloat64, len(e.r))
	copy(s, e.r)
	return s
}

// Select will select an option randomly.
func (e *EpsilonGreedy) Select() int {
	e.Lock()
	defer e.Unlock()

	if rand.Float64() > e.epsilon {
		return e.r.MaxIndex()
	}

	return e.randSelect()
}

func (e *EpsilonGreedy) randSelect() int {
	res := rand.Intn(len(e.r))
	if res < 0 {
		return e.randSelect()
	}
	return res
}

// Update should be used to increment the given option with
// the given reward amount.
func (e *EpsilonGreedy) Update(option int, reward float64) error {
	e.Lock()
	defer e.Unlock()

	if option < 0 || option > len(e.r) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	// check if removed
	if e.c[option] < 0 {
		return nil
	}

	e.c[option]++
	e.r[option] += reward

	return nil
}
