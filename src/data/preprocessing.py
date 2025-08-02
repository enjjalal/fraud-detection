"""
Data preprocessing module for credit card fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from typing import Tuple, Dict, Any
import logging
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Handle all data preprocessing tasks."""
    
    def __init__(self, save_dir: str = "data/processed"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.scalers = {}
        
    def basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic data cleaning and preprocessing.
        
        Args:
            df: Raw dataset
            
        Returns:
            Cleaned dataset
        """
        logger.info("Starting basic preprocessing...")
        
        # Create a copy
        df_clean = df.copy()
        
        # Handle missing values
        logger.info(f"Missing values before cleaning: {df_clean.isnull().sum().sum()}")
        df_clean = df_clean.dropna()
        
        # Remove duplicates
        initial_shape = df_clean.shape[0]
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Removed {initial_shape - df_clean.shape[0]} duplicate rows")
        
        # Handle outliers in Amount column
        if 'Amount' in df_clean.columns:
            Q1 = df_clean['Amount'].quantile(0.25)
            Q3 = df_clean['Amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((df_clean['Amount'] < lower_bound) | 
                             (df_clean['Amount'] > upper_bound)).sum()
            
            # Cap outliers instead of removing them
            df_clean['Amount'] = df_clean['Amount'].clip(lower=lower_bound, upper=upper_bound)
            logger.info(f"Capped {outliers_before} outliers in Amount column")
        
        logger.info(f"Basic preprocessing completed. Final shape: {df_clean.shape}")
        return df_clean
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Preprocessed dataset
            
        Returns:
            Dataset with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df_features = df.copy()
        
        # Time-based features
        if 'Time' in df_features.columns:
            # Convert time to hours
            df_features['Hour'] = (df_features['Time'] / 3600) % 24
            df_features['Day'] = (df_features['Time'] / (3600 * 24)) % 7
            
            # Create time-based bins
            df_features['Time_bin'] = pd.cut(df_features['Hour'], 
                                           bins=[0, 6, 12, 18, 24], 
                                           labels=['Night', 'Morning', 'Afternoon', 'Evening'])
            df_features['Time_bin'] = df_features['Time_bin'].cat.codes
        
        # Amount-based features
        if 'Amount' in df_features.columns:
            # Log transformation for Amount
            df_features['Amount_log'] = np.log1p(df_features['Amount'])
            
            # Amount bins
            df_features['Amount_bin'] = pd.qcut(df_features['Amount'], 
                                              q=5, labels=False, duplicates='drop')
            
            # Amount z-score
            df_features['Amount_zscore'] = (df_features['Amount'] - df_features['Amount'].mean()) / df_features['Amount'].std()
        
        # V feature interactions (if V features exist)
        v_features = [col for col in df_features.columns if col.startswith('V')]
        if len(v_features) >= 2:
            # Create some interaction features
            df_features['V1_V2_interaction'] = df_features.get('V1', 0) * df_features.get('V2', 0)
            df_features['V3_V4_interaction'] = df_features.get('V3', 0) * df_features.get('V4', 0)
            
            # Sum of absolute values of V features
            df_features['V_sum_abs'] = df_features[v_features].abs().sum(axis=1)
            df_features['V_mean_abs'] = df_features[v_features].abs().mean(axis=1)
            df_features['V_std'] = df_features[v_features].std(axis=1)
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        logger.info(f"New features added: {df_features.shape[1] - df.shape[1]}")
        
        return df_features
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                      method: str = 'standard') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Scale features using specified method.
        
        Args:
            X_train: Training features
            X_test: Test features
            method: Scaling method ('standard', 'robust', 'none')
            
        Returns:
            Scaled training and test features
        """
        if method == 'none':
            return X_train, X_test
        
        logger.info(f"Scaling features using {method} scaler...")
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on training data and transform both
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Save scaler
        self.scalers[method] = scaler
        scaler_path = self.save_dir / f"{method}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        return X_train_scaled, X_test_scaled
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series, 
                        method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle class imbalance using specified method.
        
        Args:
            X: Features
            y: Target
            method: Resampling method ('smote', 'undersample', 'smoteenn', 'none')
            
        Returns:
            Resampled features and target
        """
        if method == 'none':
            return X, y
        
        logger.info(f"Handling class imbalance using {method}...")
        logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=42, k_neighbors=5)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=42)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=42)
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Convert back to DataFrame/Series
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)
        
        logger.info(f"New class distribution: {y_resampled.value_counts().to_dict()}")
        logger.info(f"New dataset shape: {X_resampled.shape}")
        
        return X_resampled, y_resampled
    
    def prepare_data(self, df: pd.DataFrame, 
                    test_size: float = 0.2,
                    val_size: float = 0.1,
                    scaling_method: str = 'standard',
                    resampling_method: str = 'smote',
                    random_state: int = 42) -> Dict[str, Any]:
        """
        Complete data preparation pipeline.
        
        Args:
            df: Raw dataset
            test_size: Test set proportion
            val_size: Validation set proportion
            scaling_method: Feature scaling method
            resampling_method: Class imbalance handling method
            random_state: Random seed
            
        Returns:
            Dictionary containing all prepared datasets
        """
        logger.info("Starting complete data preparation pipeline...")
        
        # Basic preprocessing
        df_clean = self.basic_preprocessing(df)
        
        # Feature engineering
        df_features = self.feature_engineering(df_clean)
        
        # Separate features and target
        target_col = 'Class'
        X = df_features.drop(columns=[target_col])
        y = df_features[target_col]
        
        # Train-test split
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Train-validation split
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Handle class imbalance (only on training data)
        X_train_balanced, y_train_balanced = self.handle_imbalance(
            X_train, y_train, method=resampling_method
        )
        
        # Scale features
        X_train_scaled, X_val_scaled = self.scale_features(
            X_train_balanced, X_val, method=scaling_method
        )
        X_train_scaled, X_test_scaled = self.scale_features(
            X_train_balanced, X_test, method=scaling_method
        )
        
        # Prepare return dictionary
        data_dict = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_balanced,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'original_shape': df.shape,
            'final_train_shape': X_train_scaled.shape
        }
        
        # Save processed data
        self._save_processed_data(data_dict)
        
        logger.info("Data preparation pipeline completed successfully!")
        return data_dict
    
    def _save_processed_data(self, data_dict: Dict[str, Any]) -> None:
        """Save processed data to disk."""
        logger.info("Saving processed data...")
        
        for key, value in data_dict.items():
            if isinstance(value, (pd.DataFrame, pd.Series)):
                file_path = self.save_dir / f"{key}.pkl"
                value.to_pickle(file_path)
                logger.info(f"Saved {key} to {file_path}")

def main():
    """Main function for testing preprocessing."""
    from download_data import DataDownloader
    
    # Load data
    downloader = DataDownloader()
    df, success = downloader.load_data()
    
    if not success:
        logger.error("Failed to load data")
        return
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    data_dict = preprocessor.prepare_data(df)
    
    print("✅ Data preprocessing completed!")
    print(f"📊 Training set shape: {data_dict['X_train'].shape}")
    print(f"📊 Validation set shape: {data_dict['X_val'].shape}")
    print(f"📊 Test set shape: {data_dict['X_test'].shape}")

if __name__ == "__main__":
    main()
