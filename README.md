# Gaussian Process Bayesian Optimization for TMS Parameter Selection

This repository contains the Python implementation of the Gaussian Process Bayesian Optimization (GP-BO) framework used to optimize dual-site paired-pulse transcranial magnetic stimulation (ds-ppTMS) parameters from experimentally collected motor-evoked potentials (MEPs).

The framework evaluates whether GP-BO can efficiently identify stimulation parameters associated with maximal MEP facilitation or inhibition while limiting the number of TMS queries required.

The analyses were performed offline using experimental ds-ppTMS data. At each query, one experimentally recorded trial corresponding to the selected stimulation parameters is sampled, thereby simulating sequential parameter optimization.


## Experiments

Three parameter spaces are supported:

- **ISI_only**: optimization of the interstimulus interval (ISI)
- **Cint_only**: optimization of conditioning-stimulus intensity
- **Both**: joint optimization of ISI and conditioning-stimulus intensity

The optimization objective can be set to either:

- maximize MEP amplitude (facilitation)
- minimize MEP amplitude (inhibition)

GP-BO performance is compared with exhaustive search for all experiments and with a greedy search algorithm for the two-dimensional parameter space.

## Repository structure

```text
.
├── main.py
├── config.py
├── system.py
│
├── Model/
│   └── GPy_torch.py
│
├── Tools/
│   ├── filters.py
│   ├── helper.py
│   ├── objective.py
│   ├── procedures.py
│   └── train_sequence.py
│
├── Config/
├── Data/
├── calibrated_hyperparameters/
└── Results/
```

### Main components

- **`main.py`** — Main entry point for hyperparameter calibration and offline GP-BO validation.

- **`config.py`** — Loads and manages experiment configurations.

- **`system.py`** — Defines the experimental search space, samples recorded trials, processes EMG responses, and provides observations to the optimization algorithm.

- **`Model/GPy_torch.py`** — Implements the Gaussian Process model and acquisition function using GPyTorch.

- **`Tools/train_sequence.py`** — Runs the sequential GP-BO optimization procedure.

- **`Tools/helper.py`** — Contains baseline search algorithms, model aggregation, performance normalization, plotting utilities, and result-folder management.

- **`Tools/filters.py`** — Performs EMG filtering and temporal alignment.

- **`Tools/objective.py`** — Computes MEP peak-to-peak amplitudes from processed EMG signals.

- **`Tools/procedures.py`** — Coordinates GP-BO optimization and comparison with baseline methods.

- **`Config/`** — Contains experiment-specific configuration files.

- **`Data/`** — Contains the preprocessed experimental datasets used by the offline simulations.

- **`calibrated_hyperparameters/`** — Contains participant-specific GP-BO hyperparameters obtained during calibration.

- **`Results/`** — Stores outputs generated when running the analyses.


## Requirements

The code is written in Python and relies primarily on the following packages:
- NumPy
- SciPy
- Matplotlib
- PyTorch
- GPyTorch
- tqdm


## Installation
Two installation options are provided.

### Option 1 — Conda environment (recommended)
Create the provided Conda environment:

```bash
conda env create -f Requirements/environment.yml
```
Then activate it:

```bash
conda activate TMS_GPBO
```

### Option 2 — pip
Alternatively, install the required Python packages with:

```bash
pip install -r Requirements/requirements.txt
```
The reference environment uses Python 3.13.


## Data

The offline simulations use preprocessed experimental ds-ppTMS datasets stored in the `Data/` directory.

Each participant-specific dataset is stored as a Python pickle file containing two NumPy arrays:

- **`data`** — experimentally recorded EMG signals for each stimulation condition and trial. During GP-BO optimization, the algorithm selects a stimulation condition and samples one of the available experimental trials for that condition. The corresponding EMG signal is filtered and temporally aligned, and its peak-to-peak MEP amplitude is computed before being provided to the GP-BO algorithm as an observation.

- **`Y`** — precomputed peak-to-peak MEP amplitudes corresponding to the experimental trials. These values are used to compute the participant-specific mean and standard deviation of the MEP response across trials for each stimulation condition and are also used by the baseline search procedures.

Six experimental trials are available per stimulation condition.

For the one-dimensional experiments (`ISI_only` and `Cint_only`), `data` has the following structure:

```text
(n_conditions, n_trials, n_samples)
```

For the two-dimensional experiment (`Both`), `data` has the following structure:

```text
(n_ISI, n_CSI, n_trials, n_samples)
```

where `n_trials = 6` and each EMG signal contains 3600 samples.

The `Y` array contains the corresponding precomputed MEP amplitudes, flattened across stimulation conditions and trials.

During GP-BO optimization, trials are sampled in randomized order for each stimulation condition. Once all available trials for a condition have been used, the trial order for that condition is randomized again.


## Running the code

All settings that normally need to be modified by the user are located in the **USER SETTINGS** section of `main.py`.

Run the pipeline with:

```bash

python main.py

```

### 1. Select the experiment

```python
exp_type_ = 'ISI_only'
```
Available options are:

```python
'ISI_only'
'Cint_only'
'Both'
```

`ISI_only` and `Cint_only` define one-dimensional optimization problems, whereas `Both` defines the joint two-dimensional ISI × conditioning-stimulus intensity optimization problem.

### 2. Select the optimization objective
To optimize MEP facilitation:

```python
maximize_MEP = True
```

To optimize MEP inhibition:
```python
maximize_MEP = False
```

### 3. Select participants
For example, to run all participants from a one-dimensional experiment:

```python
potential_experiments = ['P1', 'P2', 'P3', 'P4', 'P5']
```

To run a single participant:
```python
potential_experiments = ['P1']
```

The one-dimensional experiments (`ISI_only` and `Cint_only`) include five datasets, whereas the two-dimensional experiment (`Both`) includes four.

### 4. Select the query budget
For example:

```python
total_queries = 40
```

The query budgets evaluated in the study are:

ISI only    :   30, 40, 50

CSI only    :   30, 40, 50 

ISI × CSI   :   60, 80, 100 

### 5. Select the number of stochastic repetitions
```python
temp_repeats = 100
```
The analyses reported in the study use 100 stochastic repetitions for each participant.


## Running hyperparameter calibration
Hyperparameter calibration is performed sequentially. To start a calibration run, set:

```python
Calibrate = True
```

In the **CALIBRATION SETTINGS** section of `main.py`, set `calibrate = True` for the hyperparameter to be optimized and `calibrate = False` for all other hyperparameters.

For example, to calibrate `kappa`:

```python
config_1.kappa['default'] = 3
config_1.kappa['calibrate'] = True
config_1.kappa['values'] = [3, 5, 9, 13, 15, 19, 23]
```

Only one hyperparameter should be calibrated at a time.

Run:

```bash
python main.py
```

After each calibration step, the selected hyperparameter value should be used as the default value for the next calibration step.

The hyperparameters are calibrated in the following order:
1. `kappa`
2. `rho_min`
3. `rho_max`
4. `noise_min`
5. `noise_max`
6. `random_pts`


### Understanding calibration outputs

Each run creates a new numbered directory under either:

```text
Results/Hyper_opt/Maximization/tempX/
```

or:

```text
Results/Hyper_opt/Minimization/tempX/
```

where `tempX` identifies the current run.

A separate directory is created for each participant included in the run, for example:

```text
tempX/
├── dataset_P1_exp_type_ISI_only/
├── dataset_P2_exp_type_ISI_only/
├── dataset_P3_exp_type_ISI_only/
├── dataset_P4_exp_type_ISI_only/
├── dataset_P5_exp_type_ISI_only/
└── aggregate/
```

The participant directories contain the raw results, configuration logs, and calibration figures generated for that participant.

The `aggregate/` directory contains results averaged across all participants included in the run.

### Participant-specific learning curves

For each tested hyperparameter value, a learning-curve figure is generated in the corresponding participant directory:

```text
learning_curves_<hyperparameter>_<value>.png
```

For example:

```text
learning_curves_kappa_9.png
```

These figures show optimization performance as a function of the number of queries for a single tested hyperparameter value.

Depending on the experiment, the figure can contain:

- **GPBO** — Gaussian Process Bayesian Optimization
- **Exhaustive** — exhaustive search baseline
- **Greedy** — greedy-search baseline for the two-dimensional experiment

The shaded region represents the standard error of the mean across stochastic repetitions.

### Participant-specific hyperparameter calibration figure

After all candidate values of a hyperparameter have been evaluated, an additional calibration figure is generated in the participant directory:

```text
learning_curves_<hyperparameter>_<timestamp>.png
```

This figure compares all tested hyperparameter values using:

- **Mean** — mean GP-BO performance across the learning curve
- **Last** — GP-BO performance at the final query

The participant-specific hyperparameter is selected according to the final-query performance (`Last`).

For facilitation, the value with the highest final performance is selected.

For inhibition, the value producing the strongest normalized inhibitory performance is selected (`argmin` in the inhibition performance representation).

The terminal also reports the selected value, for example:

```text
Best (argmax) value of kappa is ...
```

or:

```text
Best (argmin) value of kappa is ...
```

The selected value should then be entered as the default value before calibrating the next hyperparameter.

### Aggregate calibration figures

The `aggregate/` directory contains learning curves averaged across all participants included in the run:

```text
aggregate_<model>.png
```

When multiple candidate hyperparameter values are evaluated, it also contains:

```text
summary_<model>_<hyperparameter>.png
```

For example:

```text
summary_0_kappa.png
```

These figures summarize performance across the participants included in the run. They are useful for visualizing group-level behavior but are distinct from the participant-specific calibration figures used to determine participant-specific hyperparameters.


## Running validation
The final participant-specific hyperparameter values obtained during calibration are stored in:

```text
calibrated_hyperparameters/
```

To run the validation using these values, set:

```python
Calibrate = False
```

Select the desired experiment, objective, participants, query budget, and number of repetitions in the **USER SETTINGS** section.

For example:

```python
exp_type_ = 'ISI_only'
maximize_MEP = True
Calibrate = False
potential_experiments = ['P1', 'P2', 'P3', 'P4', 'P5']
total_queries = 40
temp_repeats = 100
```

Then run:
```bash
python main.py
```
The corresponding participant-specific hyperparameters are automatically loaded from `calibrated_hyperparameters/`.
The terminal confirms which file is being used for each participant:

```text
Loaded calibrated hyperparameters from:
calibrated_hyperparameters/MEP_maximization/ISI_only/40_queries/dataset_P1.json
```

## Understanding validation outputs
As for calibration, validation results are saved under a newly created `tempX` directory.

The `aggregate/` directory contains the main aggregate learning-curve figure:

```text
aggregate_0.png
```

This figure compares GP-BO with the applicable baseline methods across the participants included in the run.

For the one-dimensional experiments, the figure compares:
- GPBO
- Exhaustive

For the two-dimensional experiment, it compares:
- GPBO
- Exhaustive
- Greedy

The x-axis represents the number of sequential TMS queries.

The y-axis represents normalized optimization performance relative to the participant-specific experimental mean responses across stimulation conditions.

The solid curves represent performance averaged across stochastic repetitions and participants. The shaded regions represent the standard error across participants when multiple participants are included.

When only one participant is included, variability is instead calculated across stochastic repetitions.

The aggregate figure therefore provides a visual representation of how efficiently each optimization strategy approaches the participant-specific optimum as additional TMS queries are performed.

## Output files

A typical run produces files such as:

```text

Results/Hyper_opt/

└── Maximization/

    └── tempX/

        ├── dataset_P1_exp_type_ISI_only/

        │   ├── raw_results.p

        │   ├── log_config_1.json

        │   ├── ground_truth.png

        │   ├── learning_curves_kappa_3.png

        │   ├── learning_curves_kappa_5.png

        │   ├── ...

        │   └── learning_curves_kappa_<timestamp>.png

        │

        ├── dataset_P2_exp_type_ISI_only/

        │   └── ...

        │

        └── aggregate/

            ├── aggregate_0.png

            └── summary_0_<hyperparameter>.png

```

The main files are:

- **`raw_results.p`** — raw simulation results used for aggregation and plotting.
- **`log_config_1.json`** — configuration used for the run, allowing the analysis settings to be recovered.
- **`ground_truth.png`** — participant-specific mean experimental response across stimulation conditions used to evaluate optimization performance.
- **`aggregate_0.png`** — aggregate learning curves across the participants included in the run.
- **`summary_0_<hyperparameter>.png`** — comparison of candidate hyperparameter values during calibration.
- **`learning_curves_<hyperparameter>_<value>.png`** — participant-specific learning curves for one tested hyperparameter value.
- **`learning_curves_<hyperparameter>_<timestamp>.png`** — participant-specific comparison of all candidate values used to select the calibrated hyperparameter.


## Reproducibility
The simulations are stochastic because:

1. initial GP-BO queries are randomly selected;
2. individual experimental trials are sampled in randomized order;
3. the baseline search algorithms contain randomized components.

For this reason, results may vary slightly between individual runs. The analyses reported in the study are based on repeated simulations and aggregate performance across repetitions and participants.

## Associated publication
This repository accompanies the paper *Automatized personalization of stimulation parameters for cortico-cortical paired associative stimulation*, describing the application of our GP-BO algorithm to individualize ds-ppTMS parameter selection.

The full citation will be added following publication.

## Authors
Julia Price, Léo Choinière, Elnaz Allahverdloo, Marco Bonizzato, Guillaume Lajoie and Numa Dancause.