# Noise Injection: Improving Out-of-Distribution Generalization for Limited Size Datasets

This repository contains the official PyTorch Lightning-based implementation of our research on the effects of noise augmentations to generalization on data from unseen sources, especially given that our training dataset is limited in both size and diversity. For a detailed write-up of our experiment results and design choices, refer to our manuscript at [to be added]

## Setup

This project uses **`pixi`** for environment management. The `pixi.lock` file guarantees a (roughly) reproducible environment.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Duongmai127/Noisy-ood.git
    cd noisy_ood
    ```

2.  **Install the Environment**
    We provide two environments: one for GPU execution and one for CPU-only execution.

    * **For GPU (Recommended):** Requires an NVIDIA GPU with CUDA Toolkit version 12.0 or higher installed on your system.
        ```bash
        pixi install -e gpu-cuda12
        ```

    * **For CPU:**
        ```bash
        pixi install -e cpu-only
        ```

3.  **Prepare the Data**
    This project requires a specific data structure. Please follow the instructions in the [`data/README.md`](data/README.md) file to set up your datasets correctly before proceeding.

## Running Experiments

Experiments are managed by Hydra and can be launched using `pixi` tasks. We provide two primary experiment configurations as a starting point.

**Important**: Our current pipeline is customized to training with GPU only!

#### Main Experiment (Training with Noise Augmentation)

This is the default experiment, corresponding to `experiment=train_one_noise_val_no_noise` in the configuration.

* **To run on GPU:**
    ```bash
    pixi run -e gpu-cuda12 train
    ```

#### Baseline Experiment (Training without Noise Augmentation)

To run the baseline experiment, you can override the configuration from the command line.

* **To run on GPU:**
    ```bash
    pixi run -e gpu-cuda12 train experiment=train_no_noise_val_no_noise
    ```

## Configuration & Customization

All experiment parameters are defined in `config/config.yaml`. You can modify this file directly or override any parameter from the command line.

For example, to run an experiment with a different noise variance:
```bash
pixi run train noise_params.variance=0.02
```

## Output
After running an experiment, the following files and directories will be created:

* `outputs/`: Contains Hydra logs for each run, including a snapshot of the configuration used.

* `training_logs/`: Contains TensorBoard logs and the best model checkpoint for each seed.

* `training_logs/summary_stats/`: A raw log of all test metrics from every run across all seeds and The final computed mean and standard deviation for each metric, aggregated across all seeds, for both in-distribution and out-of-distribution evaluations.

Using the default configuration on a modern GPU, the entire process (10 seeds) should take approximately 2-3 hours.

## Cite our work

[To be added]