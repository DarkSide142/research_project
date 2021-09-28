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
