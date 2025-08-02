"""
Advanced Feature Engineering for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import logging
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for fraud detection."""
    
    def __init__(self):
        self.feature_names = []
        self.interaction_features = []
        self.polynomial_features = []
        self.domain_features = []
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive advanced features."""
        logger.info("🔧 Starting Advanced Feature Engineering...")
        
        df_enhanced = df.copy()
        
        # 1. Temporal features
        df_enhanced = self._create_temporal_features(df_enhanced)
        
        # 2. Amount-based features
        df_enhanced = self._create_amount_features(df_enhanced)
        
        # 3. PCA interaction features
        df_enhanced = self._create_pca_interactions(df_enhanced)
        
        # 4. Statistical features
        df_enhanced = self._create_statistical_features(df_enhanced)
        
        # 5. Domain-specific features
        df_enhanced = self._create_domain_features(df_enhanced)
        
        # 6. Polynomial features (selected)
        df_enhanced = self._create_polynomial_features(df_enhanced)
        
        # 7. Risk scoring features
        df_enhanced = self._create_risk_features(df_enhanced)
        
        logger.info(f"✅ Feature engineering completed!")
        logger.info(f"Original features: {df.shape[1]}")
        logger.info(f"Enhanced features: {df_enhanced.shape[1]}")
        logger.info(f"New features added: {df_enhanced.shape[1] - df.shape[1]}")
        
        return df_enhanced
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced temporal features."""
        logger.info("Creating temporal features...")
        
        if 'Time' not in df.columns:
            return df
        
        # Basic time features
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / (3600 * 24)) % 7
        df['Week'] = (df['Time'] / (3600 * 24 * 7)) % 52
        
        # Time-based bins
        df['Time_of_day'] = pd.cut(df['Hour'], 
                                  bins=[0, 6, 12, 18, 24], 
                                  labels=[0, 1, 2, 3],
                                  include_lowest=True).astype(int)
        
        # Weekend indicator
        df['Is_weekend'] = (df['Day'] >= 5).astype(int)
        
        # Business hours indicator
        df['Is_business_hours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)
        
        # Late night indicator (high risk period)
        df['Is_late_night'] = ((df['Hour'] >= 23) | (df['Hour'] <= 5)).astype(int)
        
        # Time since start (normalized)
        df['Time_normalized'] = (df['Time'] - df['Time'].min()) / (df['Time'].max() - df['Time'].min())
        
        # Cyclical encoding for hour
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        # Cyclical encoding for day
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 7)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 7)
        
        self.feature_names.extend([
            'Hour', 'Day', 'Week', 'Time_of_day', 'Is_weekend', 
            'Is_business_hours', 'Is_late_night', 'Time_normalized',
            'Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos'
        ])
        
        return df
    
    def _create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced amount-based features."""
        logger.info("Creating amount features...")
        
        if 'Amount' not in df.columns:
            return df
        
        # Log transformations
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_log10'] = np.log10(df['Amount'] + 1)
        df['Amount_sqrt'] = np.sqrt(df['Amount'])
        
        # Statistical transformations
        df['Amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
        df['Amount_robust_zscore'] = (df['Amount'] - df['Amount'].median()) / df['Amount'].mad()
        
        # Percentile-based features
        df['Amount_percentile'] = df['Amount'].rank(pct=True)
        
        # Amount bins
        df['Amount_bin'] = pd.qcut(df['Amount'], q=10, labels=False, duplicates='drop')
        df['Amount_bin_high'] = (df['Amount_bin'] >= 8).astype(int)
        df['Amount_bin_low'] = (df['Amount_bin'] <= 1).astype(int)
        
        # Round number indicators
        df['Amount_is_round'] = (df['Amount'] % 1 == 0).astype(int)
        df['Amount_is_round_10'] = (df['Amount'] % 10 == 0).astype(int)
        df['Amount_is_round_100'] = (df['Amount'] % 100 == 0).astype(int)
        
        # Decimal places
        df['Amount_decimal_places'] = df['Amount'].apply(
            lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0
        )
        
        # Amount categories
        amount_thresholds = [0, 1, 10, 50, 100, 500, 1000, np.inf]
        df['Amount_category'] = pd.cut(df['Amount'], bins=amount_thresholds, labels=False)
        
        self.feature_names.extend([
            'Amount_log', 'Amount_log10', 'Amount_sqrt', 'Amount_zscore',
            'Amount_robust_zscore', 'Amount_percentile', 'Amount_bin',
            'Amount_bin_high', 'Amount_bin_low', 'Amount_is_round',
            'Amount_is_round_10', 'Amount_is_round_100', 'Amount_decimal_places',
            'Amount_category'
        ])
        
        return df
    
    def _create_pca_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create PCA feature interactions."""
        logger.info("Creating PCA interaction features...")
        
        v_features = [col for col in df.columns if col.startswith('V')]
        
        if len(v_features) < 2:
            return df
        
        # Select top V features by correlation with target (if available)
        if 'Class' in df.columns:
            v_correlations = {}
            for feature in v_features:
                corr = abs(df[feature].corr(df['Class']))
                v_correlations[feature] = corr
            
            sorted_v = sorted(v_correlations.items(), key=lambda x: x[1], reverse=True)
            top_v_features = [item[0] for item in sorted_v[:10]]
        else:
            top_v_features = v_features[:10]
        
        # Create interaction features between top V features
        interaction_count = 0
        for i in range(len(top_v_features)):
            for j in range(i+1, min(i+4, len(top_v_features))):  # Limit interactions
                feature1, feature2 = top_v_features[i], top_v_features[j]
                
                # Multiplicative interaction
                df[f'{feature1}_{feature2}_mult'] = df[feature1] * df[feature2]
                
                # Additive interaction
                df[f'{feature1}_{feature2}_add'] = df[feature1] + df[feature2]
                
                # Ratio (with small epsilon to avoid division by zero)
                df[f'{feature1}_{feature2}_ratio'] = df[feature1] / (df[feature2] + 1e-8)
                
                self.interaction_features.extend([
                    f'{feature1}_{feature2}_mult',
                    f'{feature1}_{feature2}_add',
                    f'{feature1}_{feature2}_ratio'
                ])
                
                interaction_count += 3
                if interaction_count >= 30:  # Limit total interactions
                    break
            if interaction_count >= 30:
                break
        
        # V feature aggregations
        df['V_sum'] = df[v_features].sum(axis=1)
        df['V_mean'] = df[v_features].mean(axis=1)
        df['V_std'] = df[v_features].std(axis=1)
        df['V_min'] = df[v_features].min(axis=1)
        df['V_max'] = df[v_features].max(axis=1)
        df['V_range'] = df['V_max'] - df['V_min']
        df['V_median'] = df[v_features].median(axis=1)
        df['V_skew'] = df[v_features].skew(axis=1)
        df['V_kurt'] = df[v_features].kurtosis(axis=1)
        
        # Absolute value aggregations
        v_abs = df[v_features].abs()
        df['V_sum_abs'] = v_abs.sum(axis=1)
        df['V_mean_abs'] = v_abs.mean(axis=1)
        df['V_std_abs'] = v_abs.std(axis=1)
        
        # Count of positive/negative values
        df['V_positive_count'] = (df[v_features] > 0).sum(axis=1)
        df['V_negative_count'] = (df[v_features] < 0).sum(axis=1)
        df['V_zero_count'] = (df[v_features] == 0).sum(axis=1)
        
        aggregation_features = [
            'V_sum', 'V_mean', 'V_std', 'V_min', 'V_max', 'V_range',
            'V_median', 'V_skew', 'V_kurt', 'V_sum_abs', 'V_mean_abs',
            'V_std_abs', 'V_positive_count', 'V_negative_count', 'V_zero_count'
        ]
        
        self.feature_names.extend(self.interaction_features + aggregation_features)
        
        return df
    
    def _create_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        logger.info("Creating statistical features...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if 'Class' in numerical_cols:
            numerical_cols = numerical_cols.drop('Class')
        
        # Rolling statistics (if we have enough data)
        if len(df) > 1000:
            window_size = min(100, len(df) // 10)
            
            for col in ['Amount'] + [c for c in numerical_cols if c.startswith('V')][:5]:
                if col in df.columns:
                    # Rolling mean
                    df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
                    
                    # Rolling std
                    df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
                    
                    # Deviation from rolling mean
                    df[f'{col}_dev_from_rolling'] = df[col] - df[f'{col}_rolling_mean']
                    
                    self.feature_names.extend([
                        f'{col}_rolling_mean', f'{col}_rolling_std', f'{col}_dev_from_rolling'
                    ])
        
        return df
    
    def _create_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific fraud detection features."""
        logger.info("Creating domain-specific features...")
        
        # Risk indicators based on domain knowledge
        risk_features = []
        
        # Amount-based risk indicators
        if 'Amount' in df.columns:
            # Very high amounts (potential money laundering)
            df['Amount_very_high'] = (df['Amount'] > df['Amount'].quantile(0.99)).astype(int)
            
            # Very low amounts (potential testing)
            df['Amount_very_low'] = (df['Amount'] < 1).astype(int)
            
            # Suspicious round amounts
            df['Amount_suspicious_round'] = (
                (df['Amount'] % 100 == 0) & (df['Amount'] > 100)
            ).astype(int)
            
            risk_features.extend(['Amount_very_high', 'Amount_very_low', 'Amount_suspicious_round'])
        
        # Time-based risk indicators
        if 'Hour' in df.columns:
            # Unusual hours (3-6 AM)
            df['Time_unusual_hours'] = ((df['Hour'] >= 3) & (df['Hour'] <= 6)).astype(int)
            risk_features.append('Time_unusual_hours')
        
        # PCA-based risk indicators
        v_features = [col for col in df.columns if col.startswith('V')]
        if len(v_features) >= 5:
            # Extreme PCA values (potential anomalies)
            for feature in v_features[:5]:
                threshold = df[feature].std() * 3
                df[f'{feature}_extreme'] = (abs(df[feature]) > threshold).astype(int)
                risk_features.append(f'{feature}_extreme')
        
        # Composite risk score
        if risk_features:
            df['Risk_score'] = df[risk_features].sum(axis=1)
            df['Risk_score_normalized'] = df['Risk_score'] / len(risk_features)
            risk_features.extend(['Risk_score', 'Risk_score_normalized'])
        
        self.domain_features = risk_features
        self.feature_names.extend(risk_features)
        
        return df
    
    def _create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create selected polynomial features."""
        logger.info("Creating polynomial features...")
        
        # Select key features for polynomial expansion
        key_features = []
        
        if 'Amount' in df.columns:
            key_features.append('Amount')
        
        # Add top V features
        v_features = [col for col in df.columns if col.startswith('V')]
        if 'Class' in df.columns and len(v_features) > 0:
            v_correlations = {}
            for feature in v_features:
                corr = abs(df[feature].corr(df['Class']))
                v_correlations[feature] = corr
            
            sorted_v = sorted(v_correlations.items(), key=lambda x: x[1], reverse=True)
            key_features.extend([item[0] for item in sorted_v[:3]])
        else:
            key_features.extend(v_features[:3])
        
        # Create polynomial features for selected features only
        if len(key_features) > 0:
            poly_data = df[key_features].fillna(0)
            
            # Degree 2 polynomials
            poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
            poly_features = poly.fit_transform(poly_data)
            poly_feature_names = poly.get_feature_names_out(key_features)
            
            # Add only new polynomial features (not the original ones)
            new_poly_features = []
            for i, name in enumerate(poly_feature_names):
                if name not in key_features:  # Skip original features
                    df[f'poly_{name}'] = poly_features[:, i]
                    new_poly_features.append(f'poly_{name}')
            
            # Limit polynomial features to avoid overfitting
            if len(new_poly_features) > 20:
                new_poly_features = new_poly_features[:20]
                # Keep only the first 20 polynomial features
                cols_to_drop = [col for col in df.columns if col.startswith('poly_') and col not in new_poly_features]
                df = df.drop(columns=cols_to_drop)
            
            self.polynomial_features = new_poly_features
            self.feature_names.extend(new_poly_features)
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced risk scoring features."""
        logger.info("Creating risk scoring features...")
        
        # Anomaly scores based on different methods
        risk_scores = []
        
        # Z-score based anomaly detection
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if 'Class' in numerical_cols:
            numerical_cols = numerical_cols.drop('Class')
        
        # Calculate z-scores for key features
        key_features = []
        if 'Amount' in df.columns:
            key_features.append('Amount')
        
        v_features = [col for col in df.columns if col.startswith('V')]
        key_features.extend(v_features[:5])
        
        z_scores = []
        for feature in key_features:
            if feature in df.columns:
                z_score = abs((df[feature] - df[feature].mean()) / df[feature].std())
                z_scores.append(z_score)
        
        if z_scores:
            df['Anomaly_zscore'] = np.mean(z_scores, axis=0)
            df['Anomaly_zscore_max'] = np.max(z_scores, axis=0)
            risk_scores.extend(['Anomaly_zscore', 'Anomaly_zscore_max'])
        
        # Isolation-based features (simplified)
        if len(key_features) >= 2:
            # Distance from median point
            median_values = df[key_features].median()
            distances = []
            for feature in key_features:
                if feature in df.columns:
                    dist = abs(df[feature] - median_values[feature])
                    distances.append(dist)
            
            if distances:
                df['Distance_from_median'] = np.mean(distances, axis=0)
                df['Max_distance_from_median'] = np.max(distances, axis=0)
                risk_scores.extend(['Distance_from_median', 'Max_distance_from_median'])
        
        # Composite risk indicators
        if risk_scores:
            # Normalize risk scores
            for score in risk_scores:
                df[f'{score}_normalized'] = (df[score] - df[score].min()) / (df[score].max() - df[score].min() + 1e-8)
            
            # Overall risk score
            normalized_scores = [f'{score}_normalized' for score in risk_scores]
            df['Overall_risk_score'] = df[normalized_scores].mean(axis=1)
            
            risk_scores.extend(normalized_scores + ['Overall_risk_score'])
        
        self.feature_names.extend(risk_scores)
        
        return df
    
    def select_best_features(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """Select best features using statistical tests."""
        logger.info(f"Selecting top {k} features...")
        
        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1].tolist()
        if constant_features:
            logger.info(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
        
        # Feature selection using mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)} features out of {X.shape[1]}")
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def get_feature_summary(self) -> Dict[str, List[str]]:
        """Get summary of created features."""
        return {
            'all_features': self.feature_names,
            'interaction_features': self.interaction_features,
            'polynomial_features': self.polynomial_features,
            'domain_features': self.domain_features,
            'total_new_features': len(self.feature_names)
        }

def main():
    """Main function for feature engineering."""
    from download_data import DataDownloader
    
    # Load data
    downloader = DataDownloader()
    df, success = downloader.load_data()
    
    if not success:
        print("Failed to load data")
        return
    
    # Feature engineering
    engineer = AdvancedFeatureEngineer()
    df_enhanced = engineer.create_advanced_features(df)
    
    # Feature selection
    if 'Class' in df_enhanced.columns:
        X = df_enhanced.drop('Class', axis=1)
        y = df_enhanced['Class']
        
        X_selected, selected_features = engineer.select_best_features(X, y, k=100)
        
        print(f"\n🎯 Feature Engineering Summary:")
        print(f"Original features: {df.shape[1]}")
        print(f"Enhanced features: {df_enhanced.shape[1]}")
        print(f"Selected features: {len(selected_features)}")
        
        summary = engineer.get_feature_summary()
        print(f"New features created: {summary['total_new_features']}")
        print(f"Interaction features: {len(summary['interaction_features'])}")
        print(f"Polynomial features: {len(summary['polynomial_features'])}")
        print(f"Domain features: {len(summary['domain_features'])}")
    
    print("\n✅ Advanced feature engineering completed!")

if __name__ == "__main__":
    main()
