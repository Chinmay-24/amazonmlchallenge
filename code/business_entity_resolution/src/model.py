"""
ML model training and inference for entity matching.
Uses LightGBM for efficient training and high performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import pickle
import json
from pathlib import Path

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError:
    lgb = None
    print("Warning: lightgbm or sklearn not installed. Install with: pip install lightgbm scikit-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EntityMatchingModel:
    """LightGBM-based entity matching model."""
    
    def __init__(self, 
                 model_params: Optional[Dict] = None,
                 threshold: float = 0.5):
        """
        Initialize model.
        
        Args:
            model_params: LightGBM parameters
            threshold: Classification threshold for match/no-match
        """
        self.model = None
        self.threshold = threshold
        self.feature_names = None
        
        # Default parameters optimized for entity matching
        self.model_params = model_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'max_depth': 7,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
    
    def train(self, 
             X_train: pd.DataFrame, 
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             num_boost_round: int = 500,
             early_stopping_rounds: int = 50) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            num_boost_round: Number of boosting rounds
            early_stopping_rounds: Early stopping rounds
            
        Returns:
            Training history dictionary
        """
        if lgb is None:
            raise ImportError("LightGBM not installed")
        
        self.feature_names = list(X_train.columns)
        
        logger.info(f"Training model with {len(X_train)} samples and {len(self.feature_names)} features")
        logger.info(f"Positive samples: {sum(y_train)}, Negative samples: {len(y_train) - sum(y_train)}")
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
            logger.info(f"Validation set: {len(X_val)} samples")
        
        # Train
        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)
        ]
        
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        logger.info("\nTop 10 features by importance:")
        logger.info(importance.head(10).to_string(index=False))
        
        return {
            'best_iteration': self.model.best_iteration,
            'feature_importance': importance
        }
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict match probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of match probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Ensure features are in correct order
        X = X[self.feature_names]
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict match/no-match.
        
        Args:
            X: Feature dataframe
            threshold: Classification threshold (uses self.threshold if None)
            
        Returns:
            Binary predictions
        """
        threshold = threshold if threshold is not None else self.threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y_true: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Feature dataframe
            y_true: True labels
            
        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # F_0.5 score (precision-weighted)
        if precision + recall > 0:
            f_half = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            f_half = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f_0.5': f_half,
            'mean_proba': np.mean(y_proba)
        }
        
        logger.info(f"\nEvaluation metrics:")
        for name, value in metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        return metrics
    
    def optimize_threshold(self, 
                          X: pd.DataFrame, 
                          y_true: pd.Series,
                          metric: str = 'f_0.5') -> float:
        """
        Find optimal classification threshold.
        
        Args:
            X: Feature dataframe
            y_true: True labels
            metric: Metric to optimize ('f_0.5', 'f1', 'precision', 'recall')
            
        Returns:
            Optimal threshold
        """
        y_proba = self.predict_proba(X)
        
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = 0.0
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            if metric == 'f_0.5':
                if precision + recall > 0:
                    score = (1.25 * precision * recall) / (0.25 * precision + recall)
                else:
                    score = 0.0
            elif metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision
            elif metric == 'recall':
                score = recall
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.4f})")
        
        self.threshold = best_threshold
        return best_threshold
    
    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'threshold': self.threshold,
            'model_params': self.model_params
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.threshold = model_data['threshold']
        self.model_params = model_data['model_params']
        
        logger.info(f"Model loaded from {path}")
        logger.info(f"  Features: {len(self.feature_names)}")
        logger.info(f"  Threshold: {self.threshold:.3f}")


def compute_f_half_macro(predictions: Dict[str, List[str]], 
                        ground_truth: Dict[str, List[str]]) -> float:
    """
    Compute macro-averaged F_0.5 score.
    
    Args:
        predictions: Dictionary mapping s1_id -> list of predicted matched ids
        ground_truth: Dictionary mapping s1_id -> list of true matched ids
        
    Returns:
        Macro F_0.5 score
    """
    f_half_scores = []
    
    for s1_id in ground_truth.keys():
        true_matches = set(ground_truth.get(s1_id, []))
        pred_matches = set(predictions.get(s1_id, []))
        
        if len(pred_matches) == 0 and len(true_matches) == 0:
            # Correctly predicted singleton
            f_half = 1.0
        elif len(pred_matches) == 0 or len(true_matches) == 0:
            # One is empty, other is not
            f_half = 0.0
        else:
            # Both have matches
            tp = len(true_matches & pred_matches)
            fp = len(pred_matches - true_matches)
            fn = len(true_matches - pred_matches)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision + recall > 0:
                f_half = (1.25 * precision * recall) / (0.25 * precision + recall)
            else:
                f_half = 0.0
        
        f_half_scores.append(f_half)
    
    return np.mean(f_half_scores) if f_half_scores else 0.0


def main():
    """Test model training."""
    from data_loader import DataLoader
    from features import FeatureEngineer
    
    loader = DataLoader()
    
    # Load data
    s1_train, s2_train, s3_train = loader.load_source_data("train")
    gt_train = loader.load_ground_truth()
    
    # Preprocess
    s1_train = loader.preprocess_dataframe(s1_train)
    s2_train = loader.preprocess_dataframe(s2_train)
    s3_train = loader.preprocess_dataframe(s3_train)
    
    # Parse ground truth
    gt_dict = loader.parse_ground_truth(gt_train)
    
    # Combine sources
    combined = loader.combine_sources(s2_train, s3_train)
    
    # Create training pairs (small sample)
    pairs = loader.create_train_pairs(
        s1_train.head(100),  # Small sample for testing
        combined,
        gt_dict,
        negative_ratio=2.0
    )
    
    # Generate features
    engineer = FeatureEngineer()
    features_df = engineer.generate_features_for_pairs(pairs, s1_train, combined)
    
    # Split features and labels
    X = features_df.drop(['s1_id', 's2s3_id', 'label'], axis=1)
    y = features_df['label']
    
    # Train model
    model = EntityMatchingModel()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model.train(X_train, y_train, X_val, y_val, num_boost_round=100)
    
    # Evaluate
    metrics = model.evaluate(X_val, y_val)
    
    print("\nModel training completed!")


if __name__ == "__main__":
    main()
