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
	"math"
	"math/rand"
	"sync"

	"github.com/humilityai/sam"
)

// AnnealingSoftmax ...
type AnnealingSoftmax struct {
	c sam.SliceInt
	r sam.SliceFloat64

	sync.RWMutex
}

// NewAnnealingSoftmax ...
func NewAnnealingSoftmax(options int) (Optimizer, error) {
	if options < 2 {
		return nil, ErrOptions
	}

	return &AnnealingSoftmax{
		c: make(sam.SliceInt, options),
		r: make(sam.SliceFloat64, options),
	}, nil
}

// Counts returns a copy of the counts slice
func (a *AnnealingSoftmax) Counts() sam.SliceInt {
	a.Lock()
	defer a.Unlock()

	s := make(sam.SliceInt, len(a.c))
	copy(s, a.c)
	return s
}

// Extend --
func (a *AnnealingSoftmax) Extend(n int) {
	a.Lock()
	defer a.Unlock()

	i := make([]int, n, n)
	f := make([]float64, n, n)

	a.c = append(a.c, i...)
	a.r = append(a.r, f...)
}

// Remove --
func (a *AnnealingSoftmax) Remove(option int) {
	if option < 0 || option > len(a.c)-1 {
		return
	}

	a.c[option] = -1
	a.r[option] = 0
}

// Rewards returns a copy of the rewards slice
func (a *AnnealingSoftmax) Rewards() sam.SliceFloat64 {
	a.Lock()
	defer a.Unlock()

	s := make(sam.SliceFloat64, len(a.r))
	copy(s, a.r)
	return s
}

// Select ...
func (a *AnnealingSoftmax) Select() int {
	a.Lock()
	defer a.Unlock()

	temp := 1.0 / math.Log(float64(a.c.Sum())+1e-7)

	var z float64
	for i, reward := range a.r {
		if a.c[i] >= 0 {
			z += math.Exp(reward / temp)
		}
	}

	probs := make(sam.SliceFloat64, len(a.r))
	for i, reward := range a.r {
		probs[i] = math.Exp(reward/temp) / z
	}

	return probs.BoundedSum(rand.Float64())
}

// Significant --
func (a *AnnealingSoftmax) Significant(pvalue float64) bool {
	rewards, counts := make(sam.SliceFloat64, 0, 2), make(sam.SliceFloat64, 0, 2)
	top2 := a.r.MaxNWithIndex(2)
	for idx, score := range top2 {
		rewards = append(rewards, score)
		counts = append(counts, float64(a.c[idx]))
	}

	return significant(pvalue, rewards, counts)
}

// Update should be used to increment the given option with
// the given reward amount.
func (a *AnnealingSoftmax) Update(option int, reward float64) error {
	a.Lock()
	defer a.Unlock()

	if option < 0 || option > len(a.r) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	a.c[option]++
	a.r[option] += reward

	return nil
}
