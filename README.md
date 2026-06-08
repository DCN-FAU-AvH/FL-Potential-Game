<h1 align="center">Nonlinear Equilibrium Transitions in a Potential Game Model for Federated Learning</h1>

<h4 align="center"><a href="https://sites.google.com/view/liukang/home">Kang Liu</a>, <a href="https://iziqi.github.io/">Ziqi Wang</a>, and <a href="https://dcn.nat.fau.eu/enrique-zuazua/">Enrique Zuazua</a></h4>

<p align="center">
  <a href="https://doi.org/10.1016/j.physd.2026.135288">
    <img src="https://img.shields.io/badge/DOI-Physica D: Nonlinear Phenomena-blue" alt="Read the paper"/></a>
  <a href="https://arxiv.org/abs/2411.11793">
    <img src="https://img.shields.io/badge/arXiv-2411.11793-b31b1b?logo=arxiv" alt="Read on arXiv"/></a>
</p>

## Overview

This repository contains the implementation code for the paper *"Nonlinear Equilibrium Transitions in a Potential Game Model for Federated Learning."*

## Usage

First, run `main_game.py` to compute the Nash equilibrium of the FL-Game under different reward factors.

```bash
python main_game.py --m 1000
```

The four cases involving critical reward factors will be saved in the `.\utils` directory.

Then, to evaluate the FL training performance, execute `main_train.py` with the `case` argument from 1 to 4.

```bash
python main_train.py --m 1000 --case 1
```

All results and plots will be saved in the `results` folder.

## Citation

If this code is useful for your research, please cite the paper:

```bibtex
@article{lwz2026FLGame,
title = {Nonlinear Equilibrium Transitions in a Potential Game Model for Federated Learning},
journal = {Physica D: Nonlinear Phenomena},
pages = {135288},
year = {2026},
issn = {0167-2789},
author = {Kang Liu and Ziqi Wang and Enrique Zuazua}
}
```

## Acknowledgments

Alphabetical authorship according to mathematical tradition. Funded by the European Union's Horizon Europe MSCA project [ModConFlex](https://modconflex.uni-wuppertal.de/en/) (grant number 101073558)

<img src="utils/logos/logo_ModConFlex.jpg" alt="ModConFlex" height="63"/> <img src="utils/logos/logo_EU.png" alt="Funded by the EU" height="64"/>
