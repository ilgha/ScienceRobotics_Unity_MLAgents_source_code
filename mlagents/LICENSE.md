ML-Agents  
Copyright © 2017 Unity Technologies
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
    http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 
---
 
### Modifications by Université libre de Bruxelles (ULB)
This software includes modified components derived from ML-Agents,  
originally developed by Unity Technologies and distributed under the Apache License 2.0.
 
Modifications © 2025 Université libre de Bruxelles (ULB),  
IRIDIA, Brussels, Belgium.  
Authors: Ilyes Gharbi under supervision of Mauro Birattari
 
Modifications include:
- trainers\poca\optimizer_torch.py: Adapt the original poca algorithm to take state instead of observation at the critic level.
- trainers\policy\torch_policy.py: Adapt the original poca algorithm to take state instead of observation at the critic level and dissociate the state to the sensor at the actor level.
 
These modifications are also distributed under the **Apache License, Version 2.0**.