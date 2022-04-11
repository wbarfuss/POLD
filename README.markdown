# Code repository for deterministic temporal-difference learning dynamics under partial observability

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6361994.svg)](https://doi.org/10.5281/zenodo.6361994)

Python implementation to create the figures used in the publication. The deterministic learning dynamics are implemented in [.\agents\deterministic.py](https://github.com/wbarfuss/POLD/blob/main/agents/deterministic.py). The folder .\environments\ contains all test environments used. The notebook files execute the agent-environment interactions and plot the results. 

## How to use

- Either download this repository and execute the '.ipynb' files with python3 on your system running.
- Alternatively, open the '.ipynb' files in Google Colab via the links below (requires a google account).



## Environments
- Simple Coordination [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbarfuss/POLD/blob/main/plot01_SimpleCoordination.ipynb) + comparison with batch algorithm [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbarfuss/POLD/blob/main/plot02_SimpleCoordinationBatch.ipynb)

- Grid World Navigation [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbarfuss/POLD/blob/main/plot03_ParrRusselGridWorld.ipynb)

- Renewable Resource [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbarfuss/POLD/blob/main/plot04_RenewableResource.ipynb)

- Social Dilemma [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbarfuss/POLD/blob/main/plot05_UncertainSocialDilemma.ipynb)

- Zerosum Competition [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wbarfuss/POLD/blob/main/plot06_ZeroSum.ipynb)


## Reference
Barfuss W & Mann RP (2022) 
*Modeling the effects of environmental and perceptual uncertainty using deterministic reinforcement learning dynamics with partial observability*
Physical Review E 105, 3, 034409.


