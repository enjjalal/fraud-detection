# Multi-Model Training & Optimization Guide

## Overview

The Multi-Model Training & Optimization system provides a comprehensive framework for training, optimizing, and comparing multiple machine learning models for fraud detection. This system implements advanced techniques including ensemble methods, automated hyperparameter tuning, cross-validation pipelines, and feature importance analysis.

## Key Features

### 🚀 **Automated Model Training Pipeline**
- **XGBoost, LightGBM, and CatBoost** gradient boosting models
- **Automated hyperparameter optimization** using Optuna
- **Cross-validation evaluation** with stratified k-fold
- **Ensemble methods** (voting and stacking)
- **Feature importance analysis** across all models

### 📊 **Advanced Hyperparameter Tuning**
- **Optuna integration** with TPE sampler and median pruner
- **Cross-validation based optimization** for robust parameter selection
- **Model-specific parameter spaces** optimized for each algorithm
- **Parallel optimization** for faster tuning
- **Study persistence** for resuming interrupted optimizations

### 🤝 **Ensemble Methods**
- **Soft Voting**: Combines predicted probabilities
- **Hard Voting**: Combines predicted classes
- **Stacking**: Uses meta-learner (Logistic Regression) on base model predictions
- **Automatic ensemble evaluation** and comparison

### 📈 **Feature Importance Analysis**
- **Cross-model feature importance** comparison
- **Statistical analysis** (mean, std, coefficient of variation)
- **Visualization support** for importance rankings
- **Top feature identification** for model interpretability

## System Architecture

```
MultiModelTrainer
├── CrossValidationPipeline
│   ├── StratifiedKFold evaluation
│   └── Multiple scoring metrics
├── AdvancedHyperparameterTuner
│   ├── Optuna integration
│   ├── Model-specific parameter spaces
│   └── CV-based optimization
├── EnsembleTrainer
│   ├── Voting ensembles
│   └── Stacking ensembles
├── FeatureImportanceAnalyzer
│   ├── Importance extraction
│   └── Cross-model comparison
└── Results Management
    ├── JSON serialization
    └── Model persistence
```

## Usage Examples

### Basic Usage

```python
from src.models.multi_model_trainer import MultiModelTrainer
from src.models.train_multi_models import load_and_prepare_data

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()

# Initialize trainer
trainer = MultiModelTrainer(save_dir="models/saved")

# Run complete pipeline
results = trainer.run_complete_pipeline(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    optimize_hyperparams=True,
    n_trials=100
)
```

### Command Line Usage

```bash
# Quick demo with 10 optimization trials
python src/models/train_multi_models.py --mode demo

# Full optimization with 100 trials
python src/models/train_multi_models.py --mode full

# Training without hyperparameter optimization
python src/models/train_multi_models.py --mode no-opt

# Custom number of trials
python src/models/train_multi_models.py --mode demo --trials 50
```

## Pipeline Steps

### 1. **Data Preparation**
- Load and validate training data
- Feature engineering (if needed)
- Train/validation/test splits
- Data shape validation

### 2. **Hyperparameter Optimization** (Optional)
- **XGBoost optimization**: 12 parameters including max_depth, learning_rate, n_estimators
- **LightGBM optimization**: 11 parameters including num_leaves, min_child_samples
- **CatBoost optimization**: 8 parameters including depth, l2_leaf_reg
- **Cross-validation scoring**: F1-score optimization
- **Early stopping**: Median pruner for efficient optimization

### 3. **Base Model Training**
- Train XGBoost with optimized/default parameters
- Train LightGBM with optimized/default parameters  
- Train CatBoost with optimized/default parameters
- Validation set evaluation for each model

### 4. **Cross-Validation Evaluation**
- 5-fold stratified cross-validation
- Multiple metrics: F1, Precision, Recall, ROC-AUC
- Statistical analysis of CV scores
- Training vs validation performance comparison

### 5. **Feature Importance Analysis**
- Extract importance from each trained model
- Statistical analysis across models
- Identify most consistent important features
- Generate importance comparison DataFrame

### 6. **Ensemble Training**
- **Soft Voting**: Average predicted probabilities
- **Hard Voting**: Majority vote on predictions
- **Stacking**: Logistic regression meta-learner
- Ensemble evaluation on test set

### 7. **Final Evaluation**
- Test set evaluation for all models
- Performance comparison
- Best model identification
- Results serialization

## Configuration Options

### Hyperparameter Search Spaces

#### XGBoost Parameters
```python
{
    'max_depth': [3, 12],           # Tree depth
    'learning_rate': [0.01, 0.3],  # Learning rate (log scale)
    'n_estimators': [100, 1000],   # Number of trees
    'subsample': [0.6, 1.0],       # Row sampling
    'colsample_bytree': [0.6, 1.0], # Column sampling
    'reg_alpha': [0, 10],          # L1 regularization
    'reg_lambda': [0, 10],         # L2 regularization
    'min_child_weight': [1, 10],   # Minimum child weight
    'gamma': [0, 5],               # Minimum split loss
    'scale_pos_weight': [1, 10]    # Class imbalance handling
}
```

#### LightGBM Parameters
```python
{
    'num_leaves': [10, 300],        # Number of leaves
    'learning_rate': [0.01, 0.3],  # Learning rate (log scale)
    'n_estimators': [100, 1000],   # Number of trees
    'subsample': [0.6, 1.0],       # Row sampling
    'colsample_bytree': [0.6, 1.0], # Column sampling
    'reg_alpha': [0, 10],          # L1 regularization
    'reg_lambda': [0, 10],         # L2 regularization
    'min_child_samples': [5, 100], # Minimum samples in leaf
    'max_depth': [3, 12],          # Tree depth
    'min_split_gain': [0, 1],      # Minimum split gain
    'subsample_freq': [1, 10]      # Subsample frequency
}
```

#### CatBoost Parameters
```python
{
    'depth': [4, 10],                    # Tree depth
    'learning_rate': [0.01, 0.3],       # Learning rate (log scale)
    'iterations': [100, 1000],          # Number of trees
    'l2_leaf_reg': [1, 10],             # L2 regularization
    'border_count': [32, 255],          # Border count
    'bagging_temperature': [0, 1],      # Bagging temperature
    'random_strength': [0, 10],         # Random strength
    'subsample': [0.6, 1.0]             # Row sampling
}
```

## Output and Results

### Results Structure
```json
{
    "timestamp": "2024-01-01T12:00:00",
    "data_shapes": {
        "train": [8000, 50],
        "val": [1000, 50],
        "test": [1000, 50]
    },
    "best_hyperparameters": {
        "xgboost": {...},
        "lightgbm": {...},
        "catboost": {...}
    },
    "cv_results": {
        "xgboost": {
            "f1_test_mean": 0.85,
            "f1_test_std": 0.02,
            ...
        }
    },
    "test_results": {
        "xgboost": {
            "f1_score": 0.87,
            "precision": 0.89,
            "recall": 0.85,
            "roc_auc": 0.94
        }
    },
    "ensemble_results": {
        "voting_soft": {
            "f1_score": 0.89,
            "precision": 0.91,
            "recall": 0.87,
            "roc_auc": 0.96
        }
    },
    "feature_importance": {...}
}
```

### File Outputs
- `pipeline_results_YYYYMMDD_HHMMSS.json`: Complete results
- `best_hyperparameters.json`: Optimized parameters
- `{model_type}_study.pkl`: Optuna study objects
- Individual model files in `models/saved/`

## Performance Metrics

The system evaluates models using multiple metrics:

- **F1-Score**: Harmonic mean of precision and recall (primary metric)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **ROC-AUC**: Area under the ROC curve

## Best Practices

### 1. **Data Preparation**
- Ensure proper train/validation/test splits
- Handle class imbalance appropriately
- Validate feature engineering pipeline

### 2. **Hyperparameter Optimization**
- Start with fewer trials (10-50) for quick iteration
- Use more trials (100-500) for final optimization
- Monitor optimization progress with Optuna dashboard

### 3. **Model Selection**
- Consider both individual and ensemble performance
- Evaluate on multiple metrics, not just F1-score
- Validate on out-of-time data if available

### 4. **Feature Importance**
- Use feature importance for model interpretability
- Consider feature stability across models
- Validate important features with domain knowledge

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce `n_trials` for optimization
   - Use smaller datasets for initial testing
   - Monitor memory usage with `psutil`

2. **Long Training Times**
   - Start with demo mode (`--mode demo`)
   - Reduce cross-validation folds
   - Use parallel processing (`n_jobs=-1`)

3. **Poor Performance**
   - Check data quality and preprocessing
   - Validate feature engineering
   - Consider class imbalance handling

### Performance Tips

- Use SSD storage for faster I/O
- Ensure sufficient RAM (8GB+ recommended)
- Use multi-core CPU for parallel processing
- Monitor GPU usage for compatible models

## Integration

### With Existing Pipeline
```python
# Integration with existing feature engineering
from src.data.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer()
df_engineered = feature_engineer.create_all_features(raw_data)

# Use with multi-model trainer
trainer = MultiModelTrainer()
results = trainer.run_complete_pipeline(...)
```

### With API Deployment
```python
# Load best model for API deployment
import joblib

best_model = joblib.load("models/saved/best_model.pkl")
predictions = best_model.predict(new_data)
```

## Future Enhancements

- **Neural Network Integration**: Add deep learning models
- **AutoML Features**: Automated feature selection
- **Model Monitoring**: Performance drift detection
- **Distributed Training**: Multi-node optimization
- **Advanced Ensembles**: Bayesian model averaging

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Scikit-learn Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)
