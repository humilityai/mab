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
	"sync"

	"github.com/humilityai/sam"
	"gonum.org/v1/gonum/stat/distuv"
)

// ThompsonSampling ...
type ThompsonSampling struct {
	c sam.SliceInt
	r sam.SliceFloat64

	sync.RWMutex
}

// NewThompsonSampling ...
func NewThompsonSampling(options int) (Optimizer, error) {
	if options < 2 {
		return nil, ErrOptions
	}

	return &ThompsonSampling{
		c: make(sam.SliceInt, options),
		r: make(sam.SliceFloat64, options),
	}, nil
}

// Counts returns a copy of the counts slice
func (t *ThompsonSampling) Counts() sam.SliceInt {
	t.Lock()
	defer t.Unlock()

	s := make(sam.SliceInt, len(t.c))
	copy(s, t.c)
	return s
}

// Extend --
func (t *ThompsonSampling) Extend(n int) {
	t.Lock()
	defer t.Unlock()

	i := make([]int, n, n)
	f := make([]float64, n, n)

	t.c = append(t.c, i...)
	t.r = append(t.r, f...)
}

// Remove --
func (t *ThompsonSampling) Remove(option int) {
	t.Lock()
	defer t.Unlock()

	if option < 0 || option > len(t.c)-1 {
		return
	}

	t.c[option] = -1
}

// Rewards returns a copy of the rewards slice
func (t *ThompsonSampling) Rewards() sam.SliceFloat64 {
	t.Lock()
	defer t.Unlock()

	s := make(sam.SliceFloat64, len(t.r))
	copy(s, t.r)
	return s
}

// Select ...
func (t *ThompsonSampling) Select() int {
	t.Lock()
	defer t.Unlock()

	scores := make(sam.SliceFloat64, len(t.c))

	if t.r.IsZeroed() {
		for i, count := range t.c {
			if count < 0 {
				scores[i] = -1.0
			}

			d := distuv.Beta{
				Alpha: float64(count + 1.0),
				Beta:  float64(t.c.Sum() + 1),
			}
			scores[i] = d.Rand()
		}
	} else {
		for i, count := range t.c {
			if count < 0 {
				scores[i] = -1.0
			}

			d := distuv.Beta{
				Alpha: t.r[i] + 1.0,
				Beta:  float64(t.r.Sum() + 1),
			}
			scores[i] = d.Rand()
		}
	}

	return scores.MaxIndex()
}

// Update should be used to increment the given option with
// the given reward amount.
func (t *ThompsonSampling) Update(option int, reward float64) error {
	t.Lock()
	defer t.Unlock()

	if option < 0 || option > len(t.r) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	// check if removed
	if t.c[option] < 0 {
		return nil
	}

	t.c[option]++
	t.r[option] += reward

	return nil
}
