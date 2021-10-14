# Copyright 2021 Sean Robertson

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration constants"""

import torch
import math

# for log count functions, we want -inf to behave the same way as if it were zero in
# the regular count function. Thus we replace any occurrence with a log value that,
# when exponentiated, is nonzero, but only just. The division by two here ensures that
# we can add two EPS_INF values without the result being zero.
EPS_INF = math.log(torch.finfo(torch.float32).tiny) / 2


# for log probabilities that we want to clamp away from 1
EPS_0 = math.log1p(-2 * torch.finfo(torch.float32).eps)
