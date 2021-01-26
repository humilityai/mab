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

import "github.com/humilityai/sam"

// Optimizer represents a generic recommendation learning based
// bandit.
type Optimizer interface {
	Select() int
	Update(selection int, reward float64) error
	Remove(int)
	Counts() sam.SliceInt
	Rewards() sam.SliceFloat64
}
