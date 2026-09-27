"""
main.py

This is the main entry point for the experiment runner that reproduces the experiments
from "On Embeddings for Numerical Features in Tabular Deep Learning". It reads the configuration
from config.yaml, (optionally) performs hyperparameter tuning via Optuna, loads and preprocesses data,
builds the model with the selected embedding module and backbone, trains the model across multiple seed runs,
and evaluates both individual and ensemble performance.

The design follows the "Backbone-Embedding" pattern where per-feature embeddings are computed and then
aggregated by a chosen backbone (MLP, ResNet, or Transformer).

Author: [Your Name]
Date: [Date]
"""

import os
import sys
import yaml
import random
import logging
import copy
import numpy as np
import torch

# Import local modules
from dataset_loader import DatasetLoader
from model import Model
from trainer import Trainer
from evaluation import Evaluation

# Import optuna if available
try:
    import optuna
except ImportError:
    optuna = None

def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_hyperparameter_tuning(config: dict) -> dict:
    """
    Run hyperparameter tuning using Optuna to update the configuration.

    The objective function suggests hyperparameter values for:
      - learning_rate (training)
      - output_dim (baseline embedding's output dimension)
      - num_bins (for piecewise linear encoding if selected)
      - periodic_k and periodic_sigma (for periodic embedding if selected)

    After the study, the best hyperparameters are merged into the config.

    Returns:
        Updated configuration dictionary.
    """
    def objective(trial: optuna.trial.Trial) -> float:
        local_config = copy.deepcopy(config)
        search_spaces = local_config.get("hyperparameter_tuning", {}).get("search_spaces", {})

        # Suggest learning rate
        lr_space = search_spaces.get("learning_rate", {"low": 1e-5, "high": 3e-4})
        lr = trial.suggest_loguniform("learning_rate", lr_space["low"], lr_space["high"])
        local_config["training"]["learning_rate"] = lr

        # Suggest output dimension for baseline embedding
        output_dim_space = search_spaces.get("output_dim", {"low": 1, "high": 128})
        output_dim = trial.suggest_int("output_dim", output_dim_space["low"], output_dim_space["high"])
        local_config.setdefault("embeddings", {}).setdefault("baseline", {})["output_dim"] = output_dim

        # Determine which embedding type is selected.
        embedding_type = local_config.get("embeddings", {}).get("selected_type", "baseline").lower()
        if embedding_type in ["ple", "piecewise_linear_encoding"]:
            num_bins_space = search_spaces.get("num_bins", {"low": 2, "high": 256})
            num_bins = trial.suggest_int("num_bins", num_bins_space["low"], num_bins_space["high"])
            local_config.setdefault("embeddings", {}).setdefault("piecewise_linear_encoding", {})["num_bins"] = num_bins
        elif embedding_type == "periodic":
            periodic_k_space = search_spaces.get("periodic_k", {"low": 1, "high": 128})
            periodic_sigma_space = search_spaces.get("periodic_sigma", {"low": 0.001, "high": 0.1})
            k_val = trial.suggest_int("periodic_k", periodic_k_space["low"], periodic_k_space["high"])
            sigma_val = trial.suggest_float("periodic_sigma", periodic_sigma_space["low"], periodic_sigma_space["high"])
            local_config.setdefault("embeddings", {}).setdefault("periodic", {})["k"] = k_val
            local_config.setdefault("embeddings", {}).setdefault("periodic", {})["sigma"] = sigma_val

        # For objective evaluation, fix the seed
        local_config["seed"] = 42
        set_seed(42)

        # Load dataset
        dataset_loader = DatasetLoader(local_config)
        train_data, val_data, test_data = dataset_loader.load_data()

        # Determine number of numerical features and get bin boundaries
        num_features = len(train_data["feature_names"].get("numerical", []))
        bin_boundaries_dict = dataset_loader.get_bin_boundaries()
        numerical_feature_names = train_data["feature_names"].get("numerical", [])
        bin_boundaries_list = []
        for feature in numerical_feature_names:
            if feature in bin_boundaries_dict:
                bin_boundaries_list.append(bin_boundaries_dict[feature])
            else:
                bin_boundaries_list.append([0.0, 1.0])

        # Instantiate the Model
        model_instance = Model(local_config, num_features, bin_boundaries_list)

        # Instantiate Trainer with the Optuna trial passed.
        trainer_instance = Trainer(model_instance, (train_data, val_data, test_data), local_config, trial=trial)
        best_val_metric = trainer_instance.train()

        # For objective, regression: lower RMSE is better; for classification, maximize accuracy.
        task_type = local_config.get("task_type", "regression").lower()
        if task_type == "classification":
            # Return negative accuracy for minimization.
            return -best_val_metric
        else:
            return best_val_metric

    task_type = config.get("task_type", "regression").lower()
    study_direction = "minimize"  # For regression (RMSE) and for classification we minimize negative accuracy.
    study = optuna.create_study(direction=study_direction)
    study.optimize(objective, n_trials=20)
    best_params = study.best_trial.params

    # Update the main config with best parameters
    config["training"]["learning_rate"] = best_params.get("learning_rate", config["training"]["learning_rate"])
    config.setdefault("embeddings", {}).setdefault("baseline", {})["output_dim"] = best_params.get(
        "output_dim", config["embeddings"]["baseline"]["output_dim"]
    )
    embedding_type = config.get("embeddings", {}).get("selected_type", "baseline").lower()
    if embedding_type in ["ple", "piecewise_linear_encoding"]:
        config.setdefault("embeddings", {}).setdefault("piecewise_linear_encoding", {})["num_bins"] = best_params.get(
            "num_bins", config["embeddings"]["piecewise_linear_encoding"]["num_bins"]
        )
    elif embedding_type == "periodic":
        config.setdefault("embeddings", {}).setdefault("periodic", {})["k"] = best_params.get(
            "periodic_k", config["embeddings"]["periodic"]["k"]
        )
        config.setdefault("embeddings", {}).setdefault("periodic", {})["sigma"] = best_params.get(
            "periodic_sigma", config["embeddings"]["periodic"]["sigma"]
        )

    print("Hyperparameter tuning complete. Best parameters found:")
    print(best_params)
    return config

def main() -> None:
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment runner for numerical feature embeddings in tabular DL.")

    # Load configuration from config.yaml
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Configuration file '{config_path}' does not exist.")
        sys.exit(1)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully.")

    # Set base random seed (default 42 if not overridden in config)
    base_seed = config.get("seed", 42)

    # Optional hyperparameter tuning
    hyperparam_cfg = config.get("hyperparameter_tuning", {})
    tuning_enabled = hyperparam_cfg.get("enabled", False)
    if tuning_enabled and optuna is not None:
        logger.info("Hyperparameter tuning enabled. Starting Optuna study...")
        config = run_hyperparameter_tuning(config)
    else:
        logger.info("Hyperparameter tuning disabled or Optuna not available.")

    # Retrieve the number of seed runs (default 15)
    seed_runs = int(config.get("training", {}).get("seed_runs", 15))

    # To store test metrics and trained models for ensemble evaluation
    test_metrics_list = []
    trained_models = []

    # Loop over each seed run
    for run in range(seed_runs):
        current_seed = base_seed + run
        logger.info(f"Starting seed run {run+1}/{seed_runs} with seed {current_seed}.")
        set_seed(current_seed)
        config["seed"] = current_seed

        # Initialize dataset loader and load data
        dataset_loader = DatasetLoader(config)
        train_data, val_data, test_data = dataset_loader.load_data()

        # Determine number of numerical features and retrieve bin boundaries
        numerical_feature_names = train_data["feature_names"].get("numerical", [])
        num_features = len(numerical_feature_names)
        bin_boundaries_dict = dataset_loader.get_bin_boundaries()
        bin_boundaries_list = []
        for feature in numerical_feature_names:
            if feature in bin_boundaries_dict:
                bin_boundaries_list.append(bin_boundaries_dict[feature])
            else:
                bin_boundaries_list.append([0.0, 1.0])

        # Initialize the Model instance
        model_instance = Model(config, num_features, bin_boundaries_list)

        # Initialize and run the Trainer
        trainer_instance = Trainer(model_instance, (train_data, val_data, test_data), config)
        best_val_metric = trainer_instance.train()
        logger.info(f"Training complete for seed run {run+1}. Best validation metric: {best_val_metric:.6f}")

        # Load the best checkpoint
        if trainer_instance.best_model_state is not None:
            model_instance.load_state_dict(trainer_instance.best_model_state)
        else:
            logger.warning("No best checkpoint found; using final model state.")

        # Evaluate the trained model on the test set
        evaluation_instance = Evaluation(model_instance, test_data, config)
        test_results = evaluation_instance.evaluate()
        task_type = config.get("task_type", "regression").lower()
        if task_type == "classification":
            test_metric_value = test_results.get("accuracy", 0.0)
            logger.info(f"Seed run {run+1} Test Accuracy: {test_metric_value:.6f}")
        else:
            test_metric_value = test_results.get("RMSE", float('inf'))
            logger.info(f"Seed run {run+1} Test RMSE: {test_metric_value:.6f}")

        test_metrics_list.append(test_metric_value)
        trained_models.append(model_instance)

    # Aggregate individual seed run metrics
    test_metrics_array = np.array(test_metrics_list)
    mean_metric = float(np.mean(test_metrics_array))
    std_metric = float(np.std(test_metrics_array))
    if config.get("task_type", "regression").lower() == "classification":
        logger.info(f"Individual Seed Runs - Test Accuracy: Mean = {mean_metric:.6f}, Std = {std_metric:.6f}")
    else:
        logger.info(f"Individual Seed Runs - Test RMSE: Mean = {mean_metric:.6f}, Std = {std_metric:.6f}")

    # Ensemble evaluation: group models as specified in config
    ensemble_groups = int(config.get("evaluation", {}).get("ensemble", {}).get("groups", 1))
    logger.info(f"Performing ensemble evaluation with {ensemble_groups} group(s).")
    num_models = len(trained_models)
    group_size = num_models // ensemble_groups if ensemble_groups > 0 else num_models
    ensemble_group_metrics = []
    for group_idx in range(ensemble_groups):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size
        if group_idx == ensemble_groups - 1:
            end_idx = num_models
        group_models = trained_models[start_idx:end_idx]
        ensemble_evaluator = Evaluation(group_models, test_data, config)
        group_results = ensemble_evaluator.evaluate()
        if config.get("task_type", "regression").lower() == "classification":
            group_metric = group_results.get("ensemble_metric", 0.0)
            logger.info(f"Ensemble Group {group_idx+1} Accuracy: {group_metric:.6f}")
        else:
            group_metric = group_results.get("ensemble_metric", float('inf'))
            logger.info(f"Ensemble Group {group_idx+1} RMSE: {group_metric:.6f}")
        ensemble_group_metrics.append(group_metric)

    # Overall ensemble evaluation using all models
    overall_ensemble_evaluator = Evaluation(trained_models, test_data, config)
    overall_results = overall_ensemble_evaluator.evaluate()
    if config.get("task_type", "regression").lower() == "classification":
        overall_ensemble_metric = overall_results.get("ensemble_metric", 0.0)
        logger.info(f"Overall Ensemble Accuracy: {overall_ensemble_metric:.6f}")
    else:
        overall_ensemble_metric = overall_results.get("ensemble_metric", float('inf'))
        logger.info(f"Overall Ensemble RMSE: {overall_ensemble_metric:.6f}")

    logger.info("Experiment completed successfully.")

if __name__ == "__main__":
    main()
