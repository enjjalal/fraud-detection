"""
Data download module for Credit Card Fraud Detection dataset from Kaggle.
"""

import os
import pandas as pd
import zipfile
from pathlib import Path
import logging
from typing import Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataDownloader:
    """Handle dataset downloading and initial loading."""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def download_credit_card_fraud_data(self) -> bool:
        """
        Download Credit Card Fraud Detection dataset from Kaggle.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if Kaggle API is configured
            kaggle_path = Path.home() / ".kaggle" / "kaggle.json"
            if not kaggle_path.exists():
                logger.warning("Kaggle API not configured. Please set up kaggle.json")
                return self._create_synthetic_data()
            
            import kaggle
            
            # Download dataset
            dataset_name = "mlg-ulb/creditcardfraud"
            logger.info(f"Downloading {dataset_name}...")
            
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(self.data_dir), 
                unzip=True
            )
            
            logger.info("Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            logger.info("Creating synthetic dataset instead...")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self) -> bool:
        """
        Create synthetic credit card fraud data for demonstration.
        
        Returns:
            bool: True if successful
        """
        try:
            import numpy as np
            from sklearn.datasets import make_classification
            
            logger.info("Creating synthetic fraud detection dataset...")
            
            # Generate synthetic data
            X, y = make_classification(
                n_samples=100000,
                n_features=30,
                n_informative=20,
                n_redundant=5,
                n_clusters_per_class=1,
                weights=[0.999, 0.001],  # Imbalanced dataset
                flip_y=0.01,
                random_state=42
            )
            
            # Create feature names similar to credit card dataset
            feature_names = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            
            # Create DataFrame
            df = pd.DataFrame(X, columns=feature_names)
            df['Class'] = y
            
            # Add realistic Time and Amount features
            df['Time'] = np.random.exponential(scale=3600, size=len(df))  # Time in seconds
            df['Amount'] = np.random.lognormal(mean=3, sigma=1.5, size=len(df))  # Transaction amounts
            
            # Save to CSV
            output_path = self.data_dir / "creditcard.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"Synthetic dataset created: {output_path}")
            logger.info(f"Dataset shape: {df.shape}")
            logger.info(f"Fraud rate: {df['Class'].mean():.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating synthetic data: {e}")
            return False
    
    def load_data(self) -> Tuple[pd.DataFrame, bool]:
        """
        Load the credit card fraud dataset.
        
        Returns:
            Tuple[pd.DataFrame, bool]: Dataset and success flag
        """
        try:
            data_path = self.data_dir / "creditcard.csv"
            
            if not data_path.exists():
                logger.info("Dataset not found. Downloading...")
                if not self.download_credit_card_fraud_data():
                    raise FileNotFoundError("Could not download or create dataset")
            
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            logger.info(f"Data loaded successfully!")
            logger.info(f"Shape: {df.shape}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Fraud cases: {df['Class'].sum()}")
            logger.info(f"Normal cases: {(df['Class'] == 0).sum()}")
            
            return df, True
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return pd.DataFrame(), False

def main():
    """Main function to download and verify data."""
    downloader = DataDownloader()
    df, success = downloader.load_data()
    
    if success:
        print(f"✅ Data loaded successfully!")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🎯 Target distribution:")
        print(df['Class'].value_counts())
    else:
        print("❌ Failed to load data")

if __name__ == "__main__":
    main()
