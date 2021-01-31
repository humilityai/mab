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

	"github.com/humilityai/sam"
	"gonum.org/v1/gonum/stat/distuv"
)

// Optimizer represents a generic recommendation learning based
// bandit.
type Optimizer interface {
	Select() int
	Update(selection int, reward float64) error
	Extend(int)
	Remove(int)
	Significant(float64) bool
	Counts() sam.SliceInt
	Rewards() sam.SliceFloat64
}

func significant(pvalue float64, rewards, counts sam.SliceFloat64) bool {
	totalScore := rewards.Sum()
	var se0, se1, seDiff, zScore float64
	if counts[0] > 0 {
		se0 = math.Sqrt((rewards[0] * (totalScore - rewards[0])) / counts[0])
	}
	if counts[1] > 0 {
		se1 = math.Sqrt((rewards[1] * (totalScore - rewards[1])) / counts[1])
	}
	seDiff = math.Sqrt(se0*se0 + se1*se1)
	if seDiff != 0 {
		zScore = (rewards[1] - rewards[0]) / seDiff
	}

	pnorm := distuv.Normal{
		Mu:    0,
		Sigma: 1,
	}
	pVal := pnorm.CDF(-math.Abs(zScore))
	if pVal <= pvalue {
		return true
	}

	return false
}
