"""
dataset_loader.py

This module defines the DatasetLoader class that loads raw tabular data,
splits it into training, validation, and test sets, applies preprocessing (quantile transformation,
one-hot encoding, target standardization), and computes the bin boundaries for piecewise linear encoding (PLE).

Author: [Your Name]
Date: [Date]
"""

import os
import logging
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class DatasetLoader:
    """
    DatasetLoader is responsible for loading raw tabular data,
    splitting it reproducibly into training, validation, and test sets,
    performing preprocessing, and computing/caching bin boundaries for PLE.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the DatasetLoader with a configuration dictionary.

        Args:
            config (dict): Configuration settings including preprocessing flags,
                           embedding parameters for piecewise linear encoding, dataset path,
                           and optional target column name.
        """
        self.config: dict = config

        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Cache for computed bin boundaries for PLE.
        self.bin_boundaries: Dict[str, np.ndarray] = {}

        # Preprocessing configuration flags
        preprocessing_cfg = config.get("preprocessing", {})
        self.quantile_transform_flag: bool = bool(preprocessing_cfg.get("quantile_transform", True))
        self.standardize_targets_flag: bool = bool(preprocessing_cfg.get("standardize_targets", True))
        self.one_hot_encode_flag: bool = bool(preprocessing_cfg.get("one_hot_encode_categoricals", True))

        # Embedding configuration for piecewise linear encoding (PLE)
        ple_cfg = config.get("embeddings", {}).get("piecewise_linear_encoding", {})
        self.ple_method: str = ple_cfg.get("method", "quantile").lower()  # 'quantile' or 'target_aware'
        self.num_bins: int = int(ple_cfg.get("num_bins", 32))

        # Set reproducibility seed
        self.random_seed: int = config.get("seed", 42)
        np.random.seed(self.random_seed)

        # Determine target column name; default is "target"
        self.target_column: str = config.get("target_column", "target")

    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Loads the dataset from disk, applies preprocessing, splits the data into train, validation, and test sets,
        and computes bin boundaries for numerical features (for PLE).

        Returns:
            Tuple containing dictionaries for train, validation, and test data.
            Each dictionary has keys:
              - "numerical": numpy array of numerical features.
              - "categorical": numpy array of (possibly one-hot encoded) categorical features.
              - "target": numpy array of target values.
              - "feature_names": a dict with keys "numerical" and "categorical" listing the feature names.
        Raises:
            FileNotFoundError: If the data file is not found.
            ValueError: If the target column is not present in the data.
        """
        try:
            self.logger.info("Loading raw data")
            # Retrieve data path from config; default file is "data.csv"
            data_path: str = self.config.get("data_path", "data.csv")
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file '{data_path}' not found. Please provide a valid file path.")

            df: pd.DataFrame = pd.read_csv(data_path)
            self.logger.info(f"Raw data loaded with shape: {df.shape}")
        except Exception as e:
            self.logger.error("Error loading data: " + str(e))
            raise e

        # Check that target column exists
        if self.target_column not in df.columns:
            error_msg = f"Target column '{self.target_column}' not found in data columns: {list(df.columns)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Separate features from target
        features_df: pd.DataFrame = df.drop(columns=[self.target_column])
        # Identify numerical features (e.g., int64 and float64) and categorical features (object, category)
        numerical_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = features_df.select_dtypes(exclude=[np.number]).columns.tolist()
        self.logger.info(f"Identified numerical features: {numerical_features}")
        self.logger.info(f"Identified categorical features: {categorical_features}")

        # Split data reproducibly: use default ratios - train: 60%, val: 20%, test: 20%
        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=self.random_seed,
                shuffle=True
            )
            # Split train_val into train (75% of train_val -> 0.6 total) and validation (25% of train_val -> 0.2 total)
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=0.25,
                random_state=self.random_seed,
                shuffle=True
            )
            self.logger.info(f"Data split into train: {train_df.shape}, validation: {val_df.shape}, test: {test_df.shape}")
        except Exception as e:
            self.logger.error("Error splitting data: " + str(e))
            raise e

        # Preprocessing Steps
        # A. Numerical Features Preprocessing using QuantileTransformer if enabled
        if self.quantile_transform_flag and numerical_features:
            self.logger.info("Fitting QuantileTransformer on training numerical features")
            qt = QuantileTransformer(output_distribution='uniform', random_state=self.random_seed)
            train_numeric = qt.fit_transform(train_df[numerical_features])
            val_numeric = qt.transform(val_df[numerical_features])
            test_numeric = qt.transform(test_df[numerical_features])
        else:
            train_numeric = train_df[numerical_features].values
            val_numeric = val_df[numerical_features].values
            test_numeric = test_df[numerical_features].values

        # B. Categorical Features Preprocessing using one-hot encoding if enabled
        if self.one_hot_encode_flag and categorical_features:
            self.logger.info("Applying one-hot encoding on categorical features")
            # Use pandas.get_dummies on training set and align validation/test sets accordingly
            train_cat_df = pd.get_dummies(train_df[categorical_features], drop_first=False)
            val_cat_df = pd.get_dummies(val_df[categorical_features], drop_first=False)
            test_cat_df = pd.get_dummies(test_df[categorical_features], drop_first=False)
            # Align validation and test sets with training one-hot columns
            val_cat_df = val_cat_df.reindex(columns=train_cat_df.columns, fill_value=0)
            test_cat_df = test_cat_df.reindex(columns=train_cat_df.columns, fill_value=0)
            train_cat = train_cat_df.values
            val_cat = val_cat_df.values
            test_cat = test_cat_df.values
        else:
            train_cat = train_df[categorical_features].values if categorical_features else None
            val_cat = val_df[categorical_features].values if categorical_features else None
            test_cat = test_df[categorical_features].values if categorical_features else None

        # C. Target Preprocessing: standardization for regression tasks if enabled
        if self.standardize_targets_flag:
            self.logger.info("Standardizing regression targets using StandardScaler")
            target_scaler = StandardScaler()
            train_target = target_scaler.fit_transform(train_df[[self.target_column]]).flatten()
            val_target = target_scaler.transform(val_df[[self.target_column]]).flatten()
            test_target = target_scaler.transform(test_df[[self.target_column]]).flatten()
        else:
            train_target = train_df[self.target_column].values
            val_target = val_df[self.target_column].values
            test_target = test_df[self.target_column].values

        # Build data dictionaries for train, validation, and test sets
        train_data: Dict[str, Any] = {
            "numerical": train_numeric,
            "categorical": train_cat,
            "target": train_target,
            "feature_names": {
                "numerical": numerical_features,
                "categorical": categorical_features
            }
        }
        val_data: Dict[str, Any] = {
            "numerical": val_numeric,
            "categorical": val_cat,
            "target": val_target,
            "feature_names": {
                "numerical": numerical_features,
                "categorical": categorical_features
            }
        }
        test_data: Dict[str, Any] = {
            "numerical": test_numeric,
            "categorical": test_cat,
            "target": test_target,
            "feature_names": {
                "numerical": numerical_features,
                "categorical": categorical_features
            }
        }

        # Compute and cache bin boundaries for piecewise linear encoding (PLE)
        if numerical_features:
            self.logger.info("Computing bin boundaries for piecewise linear encoding (PLE)")
            if self.ple_method == "quantile":
                self.bin_boundaries = self._compute_bin_boundaries_quantile(train_df, numerical_features)
            elif self.ple_method == "target_aware":
                self.bin_boundaries = self._compute_bin_boundaries_target_aware(train_df, numerical_features)
            else:
                self.logger.warning(
                    f"Unknown PLE method '{self.ple_method}', defaulting to quantile-based method."
                )
                self.bin_boundaries = self._compute_bin_boundaries_quantile(train_df, numerical_features)

        return train_data, val_data, test_data

    def _compute_bin_boundaries_quantile(
        self, train_df: pd.DataFrame, numerical_features: list
    ) -> Dict[str, np.ndarray]:
        """
        Compute quantile-based bin boundaries for each numerical feature using training data.

        Args:
            train_df (pd.DataFrame): Training data.
            numerical_features (list): List of numerical feature names.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping each feature to its array of bin boundaries.
        """
        bin_boundaries: Dict[str, np.ndarray] = {}
        eps: float = 1e-6  # Small epsilon to filter out duplicate boundaries
        for feature in numerical_features:
            values: np.ndarray = train_df[feature].values
            # Generate quantile probabilities: num_bins+1 boundaries (from 0 to 1 inclusive)
            quantiles = np.linspace(0, 1, self.num_bins + 1)
            boundaries = np.quantile(values, quantiles)
            # Filter out duplicate boundaries due to zero-width bins
            filtered_boundaries = [boundaries[0]]
            for b in boundaries[1:]:
                if abs(b - filtered_boundaries[-1]) > eps:
                    filtered_boundaries.append(b)
            boundaries_array = np.array(filtered_boundaries)
            bin_boundaries[feature] = boundaries_array
            self.logger.info(f"Feature '{feature}': computed {len(boundaries_array) - 1} bins using quantiles")
        return bin_boundaries

    def _compute_bin_boundaries_target_aware(
        self, train_df: pd.DataFrame, numerical_features: list
    ) -> Dict[str, np.ndarray]:
        """
        Compute target-aware bin boundaries for each numerical feature using a decision tree.
        For regression tasks, uses DecisionTreeRegressor; for classification tasks, uses DecisionTreeClassifier.
        Default hyperparameters are used: max_leaf_nodes=32, min_samples_leaf=1, min_impurity_decrease=1e-9.

        Args:
            train_df (pd.DataFrame): Training data.
            numerical_features (list): List of numerical feature names.

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping each feature to its array of bin boundaries.
        """
        bin_boundaries: Dict[str, np.ndarray] = {}
        # Determine if task is regression based on the standardize_targets flag.
        is_regression: bool = self.standardize_targets_flag
        for feature in numerical_features:
            X_feature = train_df[[feature]].values
            y = train_df[self.target_column].values
            # Default hyperparameter settings for tree-based discretization
            max_leaf_nodes = 32
            min_samples_leaf = 1
            min_impurity_decrease = 1e-9
            try:
                if is_regression:
                    tree_model = DecisionTreeRegressor(
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf,
                        min_impurity_decrease=min_impurity_decrease,
                        random_state=self.random_seed
                    )
                else:
                    tree_model = DecisionTreeClassifier(
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf,
                        min_impurity_decrease=min_impurity_decrease,
                        random_state=self.random_seed
                    )
                tree_model.fit(X_feature, y)
                # Extract thresholds from the tree; threshold -2 indicates a leaf node in scikit-learn's tree
                thresholds = tree_model.tree_.threshold
                valid_thresholds = thresholds[thresholds != -2]
                unique_thresholds = np.unique(valid_thresholds)
                # Augment thresholds by adding the minimum and maximum observed values from training data
                min_val = np.min(X_feature)
                max_val = np.max(X_feature)
                boundaries = np.concatenate(([min_val], unique_thresholds, [max_val]))
                boundaries = np.sort(boundaries)
                bin_boundaries[feature] = boundaries
                self.logger.info(
                    f"Feature '{feature}': target-aware computed {len(boundaries) - 1} bins"
                )
            except Exception as e:
                self.logger.error(
                    f"Error computing target-aware bin boundaries for feature '{feature}': {e}. Using quantile method fallback."
                )
                quantile_boundaries = self._compute_bin_boundaries_quantile(train_df, [feature])
                bin_boundaries[feature] = quantile_boundaries[feature]
        return bin_boundaries

    def get_bin_boundaries(self) -> Dict[str, np.ndarray]:
        """
        Returns the cached bin boundaries computed for PLE.

        Returns:
            Dict[str, np.ndarray]: Cached bin boundaries mapping feature names to boundary arrays.
        """
        return self.bin_boundaries
