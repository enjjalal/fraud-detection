"""
Advanced Exploratory Data Analysis for Credit Card Fraud Detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedEDA:
    """Advanced EDA for fraud detection."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.fraud_df = df[df['Class'] == 1].copy()
        self.normal_df = df[df['Class'] == 0].copy()
        self.insights = {}
        
    def generate_comprehensive_report(self) -> dict:
        """Generate comprehensive EDA report."""
        print("Starting Advanced Exploratory Data Analysis...")
        
        self._dataset_overview()
        self._statistical_analysis()
        self._class_imbalance_analysis()
        self._correlation_analysis()
        self._temporal_analysis()
        self._amount_analysis()
        self._feature_importance_analysis()
        
        print("Advanced EDA completed!")
        return self.insights
    
    def _dataset_overview(self):
        """Dataset overview."""
        print("\nDataset Overview")
        
        overview = {
            'total_transactions': len(self.df),
            'fraud_transactions': len(self.fraud_df),
            'normal_transactions': len(self.normal_df),
            'fraud_rate': len(self.fraud_df) / len(self.df) * 100,
            'features': self.df.shape[1] - 1
        }
        
        for key, value in overview.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}" if isinstance(value, float) else f"{key.replace('_', ' ').title()}: {value:,}")
        
        self.insights['overview'] = overview
    
    def _statistical_analysis(self):
        """Statistical analysis."""
        print("\nStatistical Analysis")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.drop('Class')
        stats_summary = {}
        
        for col in numerical_cols:
            fraud_mean = self.fraud_df[col].mean()
            normal_mean = self.normal_df[col].mean()
            
            # Mann-Whitney U test
            try:
                _, p_value = stats.mannwhitneyu(
                    self.fraud_df[col].dropna(), 
                    self.normal_df[col].dropna(),
                    alternative='two-sided'
                )
                significant = p_value < 0.05
            except:
                p_value = 1.0
                significant = False
            
            stats_summary[col] = {
                'fraud_mean': fraud_mean,
                'normal_mean': normal_mean,
                'p_value': p_value,
                'significant': significant
            }
        
        significant_features = [k for k, v in stats_summary.items() if v['significant']]
        print(f"Features with significant difference: {len(significant_features)}")
        
        self.insights['statistics'] = stats_summary
    
    def _class_imbalance_analysis(self):
        """Class imbalance analysis."""
        print("\nClass Imbalance Analysis")
        
        class_counts = self.df['Class'].value_counts()
        imbalance_ratio = class_counts[0] / class_counts[1]
        
        imbalance_info = {
            'normal_count': class_counts[0],
            'fraud_count': class_counts[1],
            'imbalance_ratio': imbalance_ratio,
            'fraud_percentage': (class_counts[1] / len(self.df)) * 100
        }
        
        print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"Fraud percentage: {imbalance_info['fraud_percentage']:.4f}%")
        
        self.insights['imbalance'] = imbalance_info
    
    def _correlation_analysis(self):
        """Correlation analysis."""
        print("\nCorrelation Analysis")
        
        corr_matrix = self.df.corr()
        target_correlations = corr_matrix['Class'].abs().sort_values(ascending=False).drop('Class')
        
        correlation_info = {
            'top_correlations': target_correlations.head(10).to_dict(),
            'avg_correlation': corr_matrix.abs().mean().mean()
        }
        
        print("Top 5 features correlated with fraud:")
        for feature, corr in target_correlations.head(5).items():
            print(f"  {feature}: {corr:.4f}")
        
        self.insights['correlation'] = correlation_info
    
    def _temporal_analysis(self):
        """Temporal analysis."""
        print("\nTemporal Analysis")
        
        if 'Time' in self.df.columns:
            self.df['Hour'] = (self.df['Time'] / 3600) % 24
            hourly_fraud = self.df.groupby('Hour')['Class'].mean()
            
            temporal_info = {
                'peak_fraud_hour': hourly_fraud.idxmax(),
                'peak_fraud_rate': hourly_fraud.max(),
                'low_fraud_hour': hourly_fraud.idxmin(),
                'low_fraud_rate': hourly_fraud.min()
            }
            
            print(f"Peak fraud hour: {temporal_info['peak_fraud_hour']:.0f}:00")
            print(f"Peak fraud rate: {temporal_info['peak_fraud_rate']:.4f}")
            
            self.insights['temporal'] = temporal_info
    
    def _amount_analysis(self):
        """Amount analysis."""
        print("\nAmount Analysis")
        
        if 'Amount' in self.df.columns:
            fraud_amounts = self.fraud_df['Amount']
            normal_amounts = self.normal_df['Amount']
            
            amount_info = {
                'fraud_mean': fraud_amounts.mean(),
                'normal_mean': normal_amounts.mean(),
                'fraud_median': fraud_amounts.median(),
                'normal_median': normal_amounts.median()
            }
            
            print(f"Fraud mean amount: ${amount_info['fraud_mean']:.2f}")
            print(f"Normal mean amount: ${amount_info['normal_mean']:.2f}")
            
            self.insights['amount'] = amount_info
    
    def _feature_importance_analysis(self):
        """Feature importance analysis."""
        print("\nFeature Importance Analysis")
        
        from sklearn.ensemble import RandomForestClassifier
        
        X = self.df.drop('Class', axis=1)
        y = self.df['Class']
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_dict = dict(zip(X.columns, rf.feature_importances_))
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("Top 5 most important features:")
        for feature, importance in sorted_importance[:5]:
            print(f"  {feature}: {importance:.4f}")
        
        self.insights['importance'] = dict(sorted_importance[:10])
    
    def create_visualizations(self):
        """Create key visualizations."""
        print("\nCreating Visualizations...")
        
        # Create plots directory
        import os
        os.makedirs('notebooks/plots', exist_ok=True)
        
        # 1. Class distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        class_counts = self.df['Class'].value_counts()
        plt.bar(['Normal', 'Fraud'], class_counts.values, color=['skyblue', 'salmon'])
        plt.title('Class Distribution')
        plt.ylabel('Count')
        
        plt.subplot(1, 2, 2)
        plt.pie(class_counts.values, labels=['Normal', 'Fraud'], autopct='%1.4f%%', 
               colors=['skyblue', 'salmon'])
        plt.title('Class Distribution %')
        
        plt.tight_layout()
        plt.savefig('notebooks/plots/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(16, 12))
        corr_matrix = self.df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('notebooks/plots/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Feature distributions
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.drop('Class')
        target_corr = self.df[numerical_cols].corrwith(self.df['Class']).abs().sort_values(ascending=False)
        top_features = target_corr.head(6).index
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(top_features):
            self.normal_df[feature].hist(bins=50, alpha=0.7, label='Normal', 
                                       ax=axes[i], color='skyblue', density=True)
            self.fraud_df[feature].hist(bins=50, alpha=0.7, label='Fraud', 
                                      ax=axes[i], color='salmon', density=True)
            axes[i].set_title(f'{feature} Distribution')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('notebooks/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved to notebooks/plots/")

def main():
    """Main function for EDA."""
    from download_data import DataDownloader
    
    # Load data
    downloader = DataDownloader()
    df, success = downloader.load_data()
    
    if not success:
        print("Failed to load data")
        return
    
    # Run EDA
    eda = AdvancedEDA(df)
    insights = eda.generate_comprehensive_report()
    eda.create_visualizations()
    
    print("\nAdvanced EDA completed successfully!")
    return insights

if __name__ == "__main__":
    main()
