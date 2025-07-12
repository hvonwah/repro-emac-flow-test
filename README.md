[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15869558.svg)](https://doi.org/10.5281/zenodo.15869558)

# Reproduction Code

This repository contains the scripts to reproduce the numerical examples presented in the paper *Benchmark stress tests for flow past a cylinder at higher Reynolds numbers using EMAC* by H. von Wahl, Leo G. Rebholz and L. Ridgway Scott.

To run the codes, appropriate versions of NGSolve and FreeFem++ are required. Parameters may be passed to the python scripts implementing the individual examples.

## Installation

### FreeFem++
See the [FreeFem++ website](https://freefem.org) for detailed installation instructions.

### NGSolve
To run the python scripts locally, a compatible combination of `Netgen/NGSolve` is required. These can be installed by building from sources or the provided pip wheels. For detailed installation instructions, we refer to the [NGSolve installation guidelines ](https://docu.ngsolve.org/latest/install/install_sources.html). Our numerical results are realized using the following version:

| Package | git commit
|-|-|
| NGSolve | `c5808be93cc4f17f38bb0cb1ac5c97ecee3829f2`

The parameters used to obtain the presented results are given in `ngsolve/parameters*.tex`.


## Content

This repository contains the following files:

| Filename | Description | 
|-|-|
| [`README.md`](README.md) | This file. |
| [`LICENCE`](LICENCE) | MIT licence file. |
| [`freefem++/CylinderFlowP2P1BDF3_500.edp`](freefem++/CylinderFlowP2P1BDF3_500.edp) | FreeFem++ implementation, mesh 2, Re=500 |
| [`freefem++/CylinderFlowP3P2BDF3_500.edp`](freefem++/CylinderFlowP3P2BDF3_500.edp) | FreeFem++ implementation, mesh 3, Re=500 |
| [`freefem++/CylinderFlowP3P2BDF3_1000.edp`](freefem++/CylinderFlowP3P2BDF3_1000.edp) | FreeFem++ implementation, mesh 2, Re=1000 |
| [`freefem++/CylinderFlowP2P1BDF3_1000.edp`](freefem++/CylinderFlowP2P1BDF3_1000.edp) | FreeFem++ implementation, mesh 3, Re=1000 |
| [`ngsolve/flow_around_cylinder_TH.py`](ngsolve/flow_around_cylinder_TH.py) | NGSolve implementation of implicit schemes |
| [`ngsolve/flow_around_cylinder_TH_IMEX.py`](ngsolve/flow_around_cylinder_TH_IMEX.py) | NGSolve implementation of IMEX schemes |
| [`ngsolve/meshes.py`](ngsolve/meshes.py) | Mesh construction |
| [`ngsolve/newton_solver.py`](ngsolve/newton_solver.py) | Quasi-Newton solver used by implicit schemes |
| [`ngsolve/parameters.txt`](ngsolve/parameters.txt) | Parameters used for implicit schemes |
| [`ngsolve/parameters_imex.txt`](ngsolve/parameters_imex.txt) | Parameters for IMEX schemes |
| [`ngsolve/postprocess.py`](ngsolve/postprocess.py) | Compute quantities of interest from result files |
| `results/*` | Raw result files |