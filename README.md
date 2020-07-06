# TripMD: Driving patterns investigation via Motif Analysis

this repository contains all the code and supporting material for the article *TripMD: Driving patterns investigation via Motif Analysis*.
 
In short, TripMD is a system that extracts the most relevant driving patterns from sensor recordings (such as acceleration)
and provides a visualization that allows for an easy investigation. This repository contains the source code for TripMD
and the code to reproduce the experiments we did with the UAH-DriveSet dataset, a publicly available naturalistic driving
dataset

## Github project structure

    ├── notebooks          <- Jupyter notebooks with the analysis and plots created for the paper
    ├── outputs            <- Checkpoint files and results computed for the paper
    ├── src                <- Source Folder - tripMD package, experiment scripts for the paper and some extra utils
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    ├── requirements.txt   <- The requirements file for reproducing the environment used in the paper

## Prerequisites

Make sure you have Python 3 installed and that you install all the python packages listed in the requirements.txt file.

## Reproducing the experiments

This article contains two different experiments. In the first experiment, we pick a single driver and explore in detail
the outputs obtained from TripMD. In the second experiment, we focus on the task of identifying the driving behaviors of
an unknown driver using drivers whose behavior we do know.

Before starting to reproduce the experiments, you must clone/download this repository. Then, you have to download 
UAH-DriveSet dataset, which you can do [here](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/#download).

After you have cloned the project and installed the proper python environment setup, you can reproduce the first experiment
using the CLI tool in the script `uah_run.py`. Bellow, we are assuming that the full path for the folder where you have 
the UAH-DriveSet dataset is `/Documents/uah_data`. Don't forget to replace this for the correct path.

The steps are the following:

1. Open a terminal in the project's root.
2. Activate the python environment.
3. Run the command `python ./src/experiments/uah_run.py single_driver --data_path=/Documents/uah_data/D2`. This will run
TripMD and will save all the outputs in the folder `outputs/D2_driver`.
4. Start a jupyter lab session by running the command `jupyter lab`.
5. Run the notebook `tripmd_outputs_single_drivers.ipynb` to reproduce all the plots and analysis.

Now, the reproduce the second experiment, the steps are mostly the same:

1. Open a terminal in the project's root.
2. Activate the python environment.
3. Run the command `python ./src/experiments/uah_run.py all_drivers --data_path=/Documents/uah_data`. This will run
TripMD and will save all the outputs in the folder `outputs/all_drivers`. Recall that we are running a motif detection 
task for the entire dataset, which is quite computationally intense. Be prepared to wait some hours to get all the outputs.
4. Start a jupyter lab session by running the command `jupyter lab`.
5. Run the notebook `behavior_analysis.ipynb` to reproduce all the plots and analysis. You can additionally run the notebook
`tripmd_outputs_all_drivers.ipynb` to explore the TripMd outputs for the the entire UAH-DriveSet dataset.
