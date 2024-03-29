[![arXiv](https://img.shields.io/badge/arXiv-24xx.xxxxx-B31B1B.svg)](https://arxiv.org/abs/24xx.xxxxx)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

# Distinction
Codes for reproducing results from our paper arXiv:[24xx.xxxxx]
---
<!-- [<img align="right" src="plots/plots_png/NuFloorExplanation.png" height="350">](https://github.com/cajohare/NeutrinoFog/raw/master/plots/plots_png/NuFloorExplanation.png) -->

# Attention
Our work is based on the references arXiv: 2304.13665 & 2109.03116.
You can also check [`Ciaran's codes`](https://github.com/cajohare/NeutrinoFog/).

# Requirements
* [`CMasher`](https://cmasher.readthedocs.io/)
* [`Numba`](https://numba.pydata.org/)
* [`labelLine`](https://github.com/cphyc/matplotlib-label-lines)

# Running the code
Run the notebooks, and get your wants. Most computations can be done within 15 minutes.

We organize four files to content different materials:
* [`src/`] - Contains functions used in producing  results
* [`plots/`] - Contains all the plots in pdf and png formats
* [`notebooks/`] - Jupyter notebooks for obtaining and plotting results
* [`data/`] - data files, including neutrino fluxes, experimental exclusion limits, and the MC data

All classes and functions we use to derive results in our paper are maily included in [`src/`]:
* [`WIMPFuncs.py`] - Functions needed to calculate WIMP rates (based on Ciaran's code)
* [`U1Funcs.py`] - Functions needed to calculate neutrino rates w/wo new physics
* [`LabFuncs.py`] - Various utilities (from Ciaran's code)
* [`U1PlotFuncs.py`] - Plotting functions
* [`StatisticFuncs.py`] - Main classes and functions
Here, we also provide a code diagram for better understanding:
<!-- [<img align="center" src="plots/plots_png/NuFloorExplanation.png" height="350">](https://github.com/cajohare/NeutrinoFog/raw/master/plots/plots_png/NuFloorExplanation.png) -->

Bases on materials above, we offer several notebooks to obstain figures in our paper:
* [`DifferentTargets.ipynb`] - Figures 2
* [`DiscoveryLimitCurve.ipynb`] - Figure 3
* [`DLwithLSR.ipynb`] - Figure 4
* [`MCLSR.ipynb`] - Producing MC pseudo-experiments for the test statistic considering the velocity of the local standard of rest(LSR)
* [`MCNeutrino.ipynb`] - Producing MC pseudo-experiments for the test statistic only considering the neutrino fluxes
* [`MCPlot.ipynb`] - Figure 6 & 7
* [`RegeneratingNeutrinoFog.ipynb`] - Figure 1
* [`RegeneratingNeutrinoFloor.ipynb`] - Figure 5

---

If you need any further assistance or have any questions, contact me at zhangblong1036@foxmail.com. And if you do use anything here please cite the paper, [Bing-Long Zhang](https://arxiv.org/abs/2304.13665)
```
@article{Tang:2023xub,
    title={Asymptotic Analysis on Binned Likelihood and Neutrino Floor},
    author={Jian Tang and Bing-Long Zhang},
    year={2023},
    eprint={2304.13665},
    archivePrefix={arXiv},
    primaryClass={hep-ph}
}
```
