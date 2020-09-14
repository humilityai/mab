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
	C sam.SliceInt
	R sam.SliceFloat64

	sync.RWMutex
}

// NewAnnealingSoftmax ...
func NewAnnealingSoftmax(options int) (Optimizer, error) {
	if options < 2 {
		return nil, ErrOptions
	}

	return &AnnealingSoftmax{
		C: make(sam.SliceInt, options),
		R: make(sam.SliceFloat64, options),
	}, nil
}

// Select ...
func (a *AnnealingSoftmax) Select() int {
	a.Lock()
	defer a.Unlock()

	temp := 1.0 / math.Log(float64(a.C.Sum())+1e-7)

	var z float64
	for _, reward := range a.R {
		z += math.Exp(reward / temp)
	}

	probs := make(sam.SliceFloat64, len(a.R))
	for i, reward := range a.R {
		probs[i] = math.Exp(reward/temp) / z
	}

	return probs.BoundedSum(rand.Float64())
}

// Update should be used to increment the given option with
// the given reward amount.
func (a *AnnealingSoftmax) Update(option int, reward float64) error {
	a.Lock()
	defer a.Unlock()

	if option < 0 || option > len(a.R) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	// update count
	a.C[option]++

	// update reward
	n := float64(a.C[option])
	a.R[option] = (a.R[option]*(n-1) + reward) / n

	return nil
}

// Counts returns a copy of the counts slice
func (a *AnnealingSoftmax) Counts() sam.SliceInt {
	a.Lock()
	defer a.Unlock()

	s := make(sam.SliceInt, len(a.C))
	copy(s, a.C)
	return s
}

// Rewards returns a copy of the rewards slice
func (a *AnnealingSoftmax) Rewards() sam.SliceFloat64 {
	a.Lock()
	defer a.Unlock()

	s := make(sam.SliceFloat64, len(a.R))
	copy(s, a.R)
	return s
}
