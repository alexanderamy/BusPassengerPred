# Bus Passenger Count Modeling
## Install package
Locally, install the `passenger_pred` package by first cloning this repository, and then running the following command:
`pip install -e .`

Within Python, the package can then be loaded using
`import passenger_predict`.

## Report directory
This section lists the different reports that we have written, and describes how they relate to each other.

# Experiment Pipeline

### Setup
1. `conda create -n bus_prediction python=3.8`
2. `conda activate bus_prediction` 
3. `pip install experiment_pipeline/requirements.txt`
4. Get `Bus` data folder from the shared drive and put it into the `data/` folder.


### Run an experiment
1. `cd experiment_pipeline`
2. `python run_experiment.py ../data B46`

Or see the `experiment_pipeline/notebooks/example_experiment.ipynb` for an example.