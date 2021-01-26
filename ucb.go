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
	c sam.SliceInt
	r sam.SliceFloat64

	sync.RWMutex
}

// NewUpperConfidenceBound ...
func NewUpperConfidenceBound(options int) (Optimizer, error) {
	if options < 2 {
		return nil, ErrOptions
	}

	return &UpperConfidenceBound{
		c: make(sam.SliceInt, options),
		r: make(sam.SliceFloat64, options),
	}, nil
}

// Counts returns a copy of the counts slice
func (u *UpperConfidenceBound) Counts() sam.SliceInt {
	u.Lock()
	defer u.Unlock()

	s := make(sam.SliceInt, len(u.c))
	copy(s, u.c)
	return s
}

// Extend --
func (u *UpperConfidenceBound) Extend(n int) {
	u.Lock()
	defer u.Unlock()

	i := make([]int, n, n)
	f := make([]float64, n, n)

	u.c = append(u.c, i...)
	u.r = append(u.r, f...)
}

// Remove --
func (u *UpperConfidenceBound) Remove(option int) {
	if option < 0 || option > len(u.c)-1 {
		return
	}

	u.c[option] = -1
}

// Rewards returns a copy of the rewards slice
func (u *UpperConfidenceBound) Rewards() sam.SliceFloat64 {
	u.Lock()
	defer u.Unlock()

	s := make(sam.SliceFloat64, len(u.r))
	copy(s, u.r)
	return s
}

// Select ...
func (u *UpperConfidenceBound) Select() int {
	u.Lock()
	defer u.Unlock()

	bonused := make(sam.SliceFloat64, len(u.r))
	for i, count := range u.c {
		if count < 0 {
			bonused[i] = -1.0
			continue
		}

		if count == 0 {
			return i
		}

		b := math.Sqrt(2 * math.Log(float64(u.c.Sum())/float64(count)))
		bonused[i] = u.r[i] + b
	}

	return bonused.MaxIndex()
}

// Update should be used to increment the given option with
// the given reward amount.
func (u *UpperConfidenceBound) Update(option int, reward float64) error {
	u.Lock()
	defer u.Unlock()

	if option < 0 || option > len(u.r) {
		return ErrIndex
	}

	if reward < 0 {
		return ErrReward
	}

	// check if removed
	if u.c[option] < 0 {
		return nil
	}

	u.c[option]++
	u.r[option] += reward

	return nil
}
