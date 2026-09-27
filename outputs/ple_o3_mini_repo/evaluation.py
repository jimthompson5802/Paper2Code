"""
evaluation.py

This module defines the Evaluation class, which is responsible for
evaluating a trained model or an ensemble of models on a test dataset.
It supports both single-model evaluation and ensemble evaluation.
Depending on the task type (classification or regression), it computes
the appropriate evaluation metric (accuracy for classification or RMSE
for regression).

The Evaluation class uses a DataLoader for batch-wise processing of the test set,
and all configuration options (e.g., batch size, ensemble grouping, metric types)
are read from the configuration dictionary (typically loaded from config.yaml).

Author: [Your Name]
Date: [Date]
"""

import math
import logging
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Evaluation:
    """
    Evaluation class to run the trained model(s) on the test dataset and compute evaluation metrics.
    
    Attributes:
      model (Union[nn.Module, List[nn.Module]]): A single model or list of models for ensemble evaluation.
      dataset (Dict[str, Any]): The test dataset dictionary with keys "numerical" and "target".
      config (dict): Configuration dictionary with evaluation settings.
      device (torch.device): Device to run evaluation on.
      batch_size (int): Batch size used for evaluation.
      task_type (str): "classification" or "regression".
    """

    def __init__(self, model: Union[nn.Module, List[nn.Module]], dataset: Dict[str, Any], config: dict) -> None:
        """
        Initialize the Evaluation instance with model(s), dataset, and configuration.
        
        Args:
          model (Union[nn.Module, List[nn.Module]]): Trained model instance,
                                                     or a list of trained models (for ensemble evaluation).
          dataset (Dict[str, Any]): Test dataset dictionary containing keys "numerical" and "target".
          config (dict): Configuration settings (e.g., from config.yaml).
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.model = model
        self.dataset = dataset
        self.config = config

        # Use device: CUDA if available, else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Batch size: retrieve from training config, default to 128 if not specified.
        self.batch_size = int(config.get("training", {}).get("batch_size", 128))

        # Determine task type: use config["task_type"] if provided, else default to "regression"
        self.task_type = config.get("task_type", "regression").lower()

        # Prepare the test DataLoader using the dataset provided.
        self.test_loader = self._create_dataloader(data=self.dataset, batch_size=self.batch_size, shuffle=False)

        # Retrieve evaluation metrics from config, not used here explicitly aside from task_type check.
        evaluation_cfg = config.get("evaluation", {})
        self.metric_name: str = (evaluation_cfg.get("metrics", {}).get("classification", "accuracy")
                                  if self.task_type == "classification" else
                                  evaluation_cfg.get("metrics", {}).get("regression", "RMSE"))

        # For ensemble evaluation: number of ensemble groups.
        self.ensemble_groups: int = int(evaluation_cfg.get("ensemble", {}).get("groups", 1))
        self.logger.info(f"Evaluation initialized: task_type={self.task_type}, batch_size={self.batch_size}, "
                         f"metric={self.metric_name}, ensemble_groups={self.ensemble_groups}")

    def _create_dataloader(self, data: Dict[str, Any], batch_size: int, shuffle: bool) -> DataLoader:
        """
        Creates a PyTorch DataLoader from the provided dataset dictionary.
        Assumes that data["numerical"] and data["target"] are available.
        
        Args:
          data (Dict[str, Any]): Dataset dictionary.
          batch_size (int): Batch size for DataLoader.
          shuffle (bool): Whether to shuffle the data.
          
        Returns:
          DataLoader: A DataLoader for the test data.
        """
        # Convert numerical features to torch.Tensor
        inputs = torch.tensor(data["numerical"], dtype=torch.float32)
        # Convert targets appropriately based on task type: regression -> float, classification -> long
        if self.task_type == "regression":
            targets = torch.tensor(data["target"], dtype=torch.float32)
        else:
            targets = torch.tensor(data["target"], dtype=torch.long)
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model(s) on the test dataset and compute the evaluation metric.
        
        Supports both single-model evaluation and ensemble evaluation.
        
        Returns:
          Dict[str, Any]: A dictionary containing the computed metric(s). For example:
                For a single model: { "RMSE": value } or { "accuracy": value }.
                For ensemble evaluation: {
                      "ensemble_metric": value,
                      "group_metrics": [list of group metric values],
                      "individual_metrics": [list of metrics per model]
                }
        """
        # Check if self.model is a list (ensemble) or a single model.
        if isinstance(self.model, list):
            # Ensemble evaluation path.
            return self._evaluate_ensemble()
        else:
            # Single-model evaluation path.
            return self._evaluate_single(self.model)

    def _evaluate_single(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate a single model on the test dataset.
        
        Args:
          model (nn.Module): Trained model to evaluate.
          
        Returns:
          Dict[str, float]: Dictionary with a single key (e.g., "RMSE" or "accuracy").
        """
        model.to(self.device)
        model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_inputs, batch_targets in self.test_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                outputs = model(batch_inputs)
                
                # Process outputs based on task type.
                if self.task_type == "classification":
                    # Assume outputs are logits; use argmax to determine predicted class.
                    preds = torch.argmax(outputs, dim=1)
                else:
                    # Regression: outputs assumed to be continuous. Squeeze extra dimensions if present.
                    preds = outputs.squeeze()
                all_predictions.append(preds.cpu())
                all_targets.append(batch_targets.cpu())

        # Concatenate all predictions and targets.
        predictions_tensor = torch.cat(all_predictions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        num_samples = targets_tensor.shape[0]

        if self.task_type == "classification":
            # Compute accuracy.
            correct = (predictions_tensor == targets_tensor).sum().item()
            accuracy = correct / num_samples if num_samples > 0 else 0.0
            self.logger.info(f"Single Model Evaluation - Accuracy: {accuracy:.6f}")
            return {"accuracy": accuracy}
        else:
            # Regression: compute RMSE.
            mse = torch.mean((predictions_tensor - targets_tensor) ** 2).item()
            rmse = math.sqrt(mse)
            self.logger.info(f"Single Model Evaluation - RMSE: {rmse:.6f}")
            return {"RMSE": rmse}

    def _evaluate_ensemble(self) -> Dict[str, Any]:
        """
        Evaluate an ensemble of models on the test dataset.
        
        For each model, predictions are computed individually.
        Then, models are grouped into ensemble groups as specified by the configuration.
        Ensemble predictions are computed by averaging (elementwise) the model outputs for a group.
        Finally, the evaluation metric (accuracy for classification or RMSE for regression)
        is computed on the ensemble predictions.
        
        Returns:
          Dict[str, Any]: Dictionary containing:
                          - "ensemble_metric": Overall ensemble metric computed by averaging all model outputs.
                          - "group_metrics": List of metrics computed for each ensemble group.
                          - "individual_metrics": List of metrics computed for each individual model.
        """
        num_models = len(self.model)
        self.logger.info(f"Evaluating ensemble of {num_models} models.")
        # Collect individual predictions for each model.
        individual_predictions: List[torch.Tensor] = []
        individual_metrics: List[float] = []

        # We assume that the ground truth is the same for all models.
        all_targets = None

        for idx, single_model in enumerate(self.model):
            single_model.to(self.device)
            single_model.eval()
            model_preds = []

            with torch.no_grad():
                for batch_inputs, batch_targets in self.test_loader:
                    batch_inputs = batch_inputs.to(self.device)
                    batch_targets = batch_targets.to(self.device)
                    outputs = single_model(batch_inputs)
                    if self.task_type == "classification":
                        # For ensemble averaging, we need the raw logits.
                        # Save outputs as is.
                        model_preds.append(outputs.cpu())
                    else:
                        # Regression: outputs are continuous, squeeze if necessary.
                        model_preds.append(outputs.squeeze().cpu())
                    # Save targets once (they are the same for all models)
                    if all_targets is None:
                        all_targets = batch_targets.cpu()
            # Concatenate predictions for this model.
            preds_tensor = torch.cat(model_preds, dim=0)
            individual_predictions.append(preds_tensor)

            # Compute individual metric for this model.
            if self.task_type == "classification":
                # If predictions are logits, convert to class indices.
                preds_classes = torch.argmax(preds_tensor, dim=1)
                correct = (preds_classes == all_targets).sum().item()
                acc = correct / all_targets.shape[0]
                individual_metrics.append(acc)
                self.logger.info(f"Model {idx} individual accuracy: {acc:.6f}")
            else:
                mse = torch.mean((preds_tensor - all_targets) ** 2).item()
                rmse = math.sqrt(mse)
                individual_metrics.append(rmse)
                self.logger.info(f"Model {idx} individual RMSE: {rmse:.6f}")

        # Determine ensemble grouping:
        groups = self.ensemble_groups if self.ensemble_groups > 0 else 1
        group_size = num_models // groups if num_models >= groups else 1
        group_metrics = []
        ensemble_group_predictions = []

        for group_idx in range(groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            # Handle case where last group may include remaining models.
            if group_idx == groups - 1:
                end_idx = num_models
            group_preds_list = individual_predictions[start_idx:end_idx]
            # Average predictions elementwise
            # For classification, average the logits
            group_avg = torch.stack(group_preds_list, dim=0).mean(dim=0)
            ensemble_group_predictions.append(group_avg)
            
            # Compute metric for this group
            if self.task_type == "classification":
                group_classes = torch.argmax(group_avg, dim=1)
                correct = (group_classes == all_targets).sum().item()
                acc = correct / all_targets.shape[0]
                group_metrics.append(acc)
                self.logger.info(f"Ensemble Group {group_idx} accuracy: {acc:.6f}")
            else:
                mse = torch.mean((group_avg - all_targets) ** 2).item()
                rmse = math.sqrt(mse)
                group_metrics.append(rmse)
                self.logger.info(f"Ensemble Group {group_idx} RMSE: {rmse:.6f}")

        # Compute overall ensemble metric by averaging predictions from all models
        if self.task_type == "classification":
            overall_logits = torch.stack(individual_predictions, dim=0).mean(dim=0)
            overall_preds = torch.argmax(overall_logits, dim=1)
            correct = (overall_preds == all_targets).sum().item()
            ensemble_metric = correct / all_targets.shape[0]
            self.logger.info(f"Overall ensemble accuracy: {ensemble_metric:.6f}")
        else:
            overall_preds = torch.stack(individual_predictions, dim=0).mean(dim=0)
            mse = torch.mean((overall_preds - all_targets) ** 2).item()
            ensemble_metric = math.sqrt(mse)
            self.logger.info(f"Overall ensemble RMSE: {ensemble_metric:.6f}")

        return {
            "ensemble_metric": ensemble_metric,
            "group_metrics": group_metrics,
            "individual_metrics": individual_metrics
        }
