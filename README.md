# tripMD

The TripMD package was built for the paper *TripMD: Maneuver investigation via motif detection*. In the 
[github repository](https://github.com/misilva73/tripMD), you'll find all the work presented in the paper.
 
TripMD is ...

## Github project structure

    ├── notebooks          <- Jupyter notebooks with the analysis and plots created for the paper
    ├── outputs            <- Checkpoint files and results  computed for the paper
    ├── paper              <- PDF and latex project for the paper
    ├── src                <- Source Folder - tripMD package, experiment scripts for the paper and some extra utils
    ├── README.md          <- The top-level README for this project
    ├── requirements.txt   <- The requirements file for reproducing the environment used in the paper

## Prerequisites

In order to run the tripMD package, you need the following packages:

    dtaidistance
    dtw-som==1.0.4
    numpy
    saxpy
    sklearn?
    
    
In addition to these, if you wish to run the notebooks in this repository, then you need the following packages:

    jupyterlab
    matplotlib
    pandas
    seaborn

## Installing

This packages is available on PyPI and thus can be directly installed with pip:

```bash
pip install tripmd
```

Alternatively, this package can installed from source by cloning this repository and installing it manually with the 
command:

```bash
python setup.py install
```

## Example Code