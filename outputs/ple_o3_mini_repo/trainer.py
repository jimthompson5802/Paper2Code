"""
trainer.py

This module defines the Trainer class which orchestrates the training 
loop, validation, checkpointing, and integration with hyperparameter tuning 
(Optuna) for tabular deep learning models. It uses the configurations specified 
in config.yaml and works with the Model and DatasetLoader modules.

Author: [Your Name]
Date: [Date]
"""

import os
import math
import logging
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Optionally import Optuna if available
try:
    import optuna
except ImportError:
    optuna = None


class Trainer:
    """
    Trainer class to handle training, validation, early stopping, checkpointing,
    and hyperparameter tuning with Optuna.
    
    Attributes:
      model (nn.Module): The model to be trained.
      train_data (dict): Dictionary containing training data.
      val_data (dict): Dictionary containing validation data.
      test_data (dict): Dictionary containing test data.
      config (dict): Configuration dictionary loaded from config.yaml.
      optimizer (torch.optim.Optimizer): Optimizer instance (AdamW).
      loss_fn (nn.Module): Loss function (CrossEntropyLoss or MSELoss).
      batch_size (int): Training batch size.
      max_epochs (int): Maximum number of epochs.
      patience (int): Early stopping patience.
      device (torch.device): Device on which training is run.
      best_val_metric (float): Best validation metric achieved.
      best_epoch (int): Epoch which achieved the best valuation.
      best_model_state (dict): The model state dict for the best model checkpoint.
      trial (Optional[optuna.trial.Trial]): Optional Optuna trial for hyperparameter tuning.
      task_type (str): "classification" or "regression".
    """

    def __init__(
        self, 
        model: nn.Module, 
        dataset: Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]], 
        config: dict, 
        trial: Optional[Any] = None
    ) -> None:
        """
        Initializes the Trainer with a model, the datasets, configuration, 
        and an optional Optuna trial.

        Args:
          model (nn.Module): The model instance.
          dataset (Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]): A tuple (train_data, val_data, test_data)
            where each data dictionary has keys "numerical" and "target".
          config (dict): Configuration dictionary.
          trial (Optional[Any]): An optional Optuna trial instance for hyperparameter tuning.
        """
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        self.model = model
        self.config = config
        self.trial = trial  # Can be None if hyperparameter tuning is not enabled

        # Unpack the dataset tuple (train, validation, test)
        self.train_data, self.val_data, self.test_data = dataset

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Retrieve training hyperparameters from config with defaults
        training_cfg = config.get("training", {})
        self.learning_rate: float = float(training_cfg.get("learning_rate", 1e-4))
        self.batch_size: int = int(training_cfg.get("batch_size", 128))
        self.max_epochs: int = int(training_cfg.get("max_epochs", 200))
        self.patience: int = int(training_cfg.get("patience", 16))
        self.seed_runs: int = int(training_cfg.get("seed_runs", 15))

        # Determine task type: default to "regression" if not specified in config.
        # It can be set in config["task_type"].
        self.task_type: str = config.get("task_type", "regression").lower()

        # Initialize the optimizer: AdamW
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Determine loss function based on task type
        if self.task_type == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            # Assume regression task
            self.loss_fn = nn.MSELoss()

        # Initialize checkpoint and early stopping parameters
        self.best_val_metric: float = math.inf if self.task_type == "regression" else -math.inf
        self.best_epoch: int = 0
        self.best_model_state: Optional[dict] = None
        self.early_stopping_counter: int = 0

        # Prepare DataLoaders for training and validation datasets
        self.train_loader = self._create_dataloader(self.train_data, self.batch_size, shuffle=True)
        self.val_loader = self._create_dataloader(self.val_data, self.batch_size, shuffle=False)

        self.logger.info("Trainer initialization complete.")

    def _create_dataloader(self, data: Dict[str, Any], batch_size: int, shuffle: bool) -> DataLoader:
        """
        Converts data dictionary into a PyTorch DataLoader. Assumes that data["numerical"]
        and data["target"] are available as NumPy arrays.

        Args:
          data (Dict[str, Any]): Data dictionary from DatasetLoader.
          batch_size (int): Batch size.
          shuffle (bool): Whether to shuffle data.

        Returns:
          DataLoader: Torch DataLoader for the provided data.
        """
        # Convert numpy arrays into torch Tensors
        # For input features, we use the 'numerical' field.
        inputs = torch.tensor(data["numerical"], dtype=torch.float32)
        targets = torch.tensor(data["target"])
        if self.task_type == "regression":
            targets = targets.float()
        else:
            # Classification: targets should be long integers.
            targets = targets.long()
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

    def train(self) -> float:
        """
        Executes the training loop over epochs, performs validation, and implements
        early stopping with checkpointing. Also integrates with Optuna if enabled.

        Returns:
          float: The best validation metric achieved during training.
        """
        self.logger.info("Starting training process.")
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0

            # Training loop over batches
            for batch_inputs, batch_targets in self.train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                # Zero out gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(batch_inputs)
                
                # Compute loss
                # For classification using CrossEntropyLoss, outputs shape should be [B, num_classes] and targets [B]
                loss = self.loss_fn(outputs, batch_targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')

            # Validate the model
            val_results = self.validate()
            val_loss = val_results.get("val_loss", float('inf'))
            val_metric = val_results.get("val_metric", None)

            # Log current epoch results
            self.logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}, Val Metric = {val_metric:.6f}")

            # Optuna integration for hyperparameter tuning if trial is not None
            if self.trial is not None:
                self.trial.report(val_metric, epoch)
                if self.trial.should_prune():
                    self.logger.info("Trial pruned by Optuna at epoch {}".format(epoch))
                    raise optuna.exceptions.TrialPruned()

            # Early stopping check: For regression, lower is better; for classification, higher is better.
            is_improved = False
            if self.task_type == "regression":
                if val_metric < self.best_val_metric:
                    is_improved = True
            else:
                if val_metric > self.best_val_metric:
                    is_improved = True

            if is_improved:
                self.logger.info(f"New best validation metric achieved at epoch {epoch}: {val_metric:.6f}")
                self.best_val_metric = val_metric
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                # Save the best model state (checkpoint)
                self.best_model_state = self.model.state_dict()
                # Optionally, save to disk if desired: using checkpoint path from config; default to "best_model.pt"
                checkpoint_path = self.config.get("checkpoint_path", "best_model.pt")
                torch.save(self.best_model_state, checkpoint_path)
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            else:
                self.early_stopping_counter += 1
                self.logger.info(f"No improvement in validation metric. Early stopping counter: {self.early_stopping_counter}")

            # Check early stopping condition
            if self.early_stopping_counter > self.patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs.")
                break

        self.logger.info(f"Training complete. Best validation metric: {self.best_val_metric:.6f} at epoch {self.best_epoch}")
        return self.best_val_metric

    def validate(self) -> Dict[str, float]:
        """
        Validates the model on the validation dataset.

        Returns:
          Dict[str, float]: Dictionary containing the average validation loss ("val_loss")
          and the validation metric ("val_metric"). For regression, the metric is RMSE.
          For classification, the metric is accuracy.
        """
        self.model.eval()
        val_loss_total = 0.0
        total_samples = 0
        metric_accum = 0.0  # For classification: count of correct predictions; for regression: sum of squared errors
        batch_count = 0

        with torch.no_grad():
            for batch_inputs, batch_targets in self.val_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                outputs = self.model(batch_inputs)
                loss = self.loss_fn(outputs, batch_targets)
                val_loss_total += loss.item()
                batch_count += 1
                # Update metric computation
                if self.task_type == "classification":
                    # For classification, assume outputs are logits and use argmax for predictions
                    predictions = torch.argmax(outputs, dim=1)
                    correct = (predictions == batch_targets).sum().item()
                    metric_accum += correct
                    total_samples += batch_targets.size(0)
                else:
                    # For regression, accumulate squared error for RMSE
                    # The loss function is MSELoss but we accumulate individual squared errors
                    # Multiply by batch size since loss is an average per batch
                    batch_size = batch_targets.size(0)
                    total_samples += batch_size
                    # Compute squared errors per sample manually
                    squared_errors = (outputs.squeeze() - batch_targets) ** 2
                    metric_accum += squared_errors.sum().item()

        avg_val_loss = val_loss_total / batch_count if batch_count > 0 else float('inf')

        if self.task_type == "classification":
            accuracy = metric_accum / total_samples if total_samples > 0 else 0.0
            val_metric = accuracy
        else:
            mse = metric_accum / total_samples if total_samples > 0 else float('inf')
            rmse = math.sqrt(mse)
            val_metric = rmse

        # Set model back to training mode
        self.model.train()

        return {"val_loss": avg_val_loss, "val_metric": val_metric}
