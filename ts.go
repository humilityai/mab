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
	C sam.SliceInt
	R sam.SliceFloat64

	sync.RWMutex
}

// NewThompsonSampling ...
func NewThompsonSampling(options int) (Optimizer, error) {
	if options < 2 {
		return nil, ErrOptions
	}

	return &ThompsonSampling{
		C: make(sam.SliceInt, options),
		R: make(sam.SliceFloat64, options),
	}, nil
}

// Select ...
func (t *ThompsonSampling) Select() int {
	t.Lock()
	defer t.Unlock()

	scores := make(sam.SliceFloat64, len(t.C))
	for i, count := range t.C {
		d := distuv.Beta{
			Alpha: float64(count + 1),
			Beta:  float64(t.C.Sum() - count + 1),
		}
		scores[i] = d.Rand()
	}

	return scores.MaxIndex()
}

// Update should be used to increment the given option with
// the given reward amount.
func (t *ThompsonSampling) Update(option int, reward float64) error {
	t.Lock()
	defer t.Unlock()

	if option < 0 || option > len(t.R) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	// update count
	t.C[option]++

	// update reward
	n := float64(t.C[option])
	t.R[option] = (t.R[option]*(n-1) + reward) / n

	return nil
}

// Counts returns a copy of the counts slice
func (t *ThompsonSampling) Counts() sam.SliceInt {
	t.Lock()
	defer t.Unlock()

	s := make(sam.SliceInt, len(t.C))
	copy(s, t.C)
	return s
}

// Rewards returns a copy of the rewards slice
func (t *ThompsonSampling) Rewards() sam.SliceFloat64 {
	t.Lock()
	defer t.Unlock()

	s := make(sam.SliceFloat64, len(t.R))
	copy(s, t.R)
	return s
}
