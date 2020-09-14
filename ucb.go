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
	"sync"

	"github.com/humilityai/sam"
)

// UpperConfidenceBound ...
type UpperConfidenceBound struct {
	C sam.SliceInt
	R sam.SliceFloat64

	sync.RWMutex
}

// NewUpperConfidenceBound ...
func NewUpperConfidenceBound(options int) (Optimizer, error) {
	if options < 2 {
		return nil, ErrOptions
	}

	return &UpperConfidenceBound{
		C: make(sam.SliceInt, options),
		R: make(sam.SliceFloat64, options),
	}, nil
}

// Select ...
func (u *UpperConfidenceBound) Select() int {
	u.Lock()
	defer u.Unlock()

	bonused := make(sam.SliceFloat64, len(u.R))

	for i, count := range u.C {
		if count == 0 {
			return i
		}

		b := math.Sqrt(2 * math.Log(float64(u.C.Sum())/float64(count)))
		bonused[i] = u.R[i] + b
	}

	return bonused.MaxIndex()
}

// Update should be used to increment the given option with
// the given reward amount.
func (u *UpperConfidenceBound) Update(option int, reward float64) error {
	u.Lock()
	defer u.Unlock()

	if option < 0 || option > len(u.R) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	// update count
	u.C[option]++

	// update reward
	n := float64(u.C[option])
	u.R[option] = (u.R[option]*(n-1) + reward) / n

	return nil
}

// Counts returns a copy of the counts slice
func (u *UpperConfidenceBound) Counts() sam.SliceInt {
	u.Lock()
	defer u.Unlock()

	s := make(sam.SliceInt, len(u.C))
	copy(s, u.C)
	return s
}

// Rewards returns a copy of the rewards slice
func (u *UpperConfidenceBound) Rewards() sam.SliceFloat64 {
	u.Lock()
	defer u.Unlock()

	s := make(sam.SliceFloat64, len(u.R))
	copy(s, u.R)
	return s
}
