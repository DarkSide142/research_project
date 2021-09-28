This repository is the result of scientific research using unsupervised ML algorithms to prevent failure during the drilling of oil&gas wells. The main goal is to produce a system that isn't essentially required a pre-drill model because the system utilizes a self-learning and self-adjusting model and proactively identifying an anomaly in wellbore condition and mitigates a stuck pipe incident before it occurs.

## How to Run

To run and output data to a local file:

    ./run.py

To run and output data to a **matplotlib** graph:

    ./run.py --plot

> You must have **matplotlib** properly installed for this option to work.

To run and get code execution info:

    ./run.py --track

To run and finally get information & statistics about the state of the HTM:

    ./run.py --info

## Program Description

Model parameters is located in the `model_params` directory.

### The Chart Explained

The chart produced with the `--plot` option contains red highlights where anomaly loglikelihood is above 0.35. The 0.35 threshold is low enough that it may provide only significant deviation. 
