# 🧪 Fraud Detection System - Testing Guide

## 📋 Quick Start Testing

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Verify Python version
python --version  # Should be 3.9+
```

## 🎯 Testing Process

### Stage 3: Multi-Model Training
```bash
python demo_multi_model_training.py
```

**Expected Results:**
- **Data**: 10,000 transactions, 52 features
- **Models**: XGBoost, LightGBM, CatBoost trained
- **Performance**: ROC-AUC > 0.95 for all models
- **Ensemble**: Voting/Stacking outperforms individual models
- **Features**: V3_V9_mult, V5_rolling_mean top importance

**Success Indicators:**
```
✅ ROC-AUC: 0.9850+ (XGBoost), 0.9820+ (LightGBM), 0.9800+ (CatBoost)
✅ Ensemble ROC-AUC: 0.99+
✅ Feature importance: Engineered features dominate top 10
```

### Stage 5: Production API
```bash
python demo_production_api.py
```

**Expected Results:**
- **Server**: Starts in <10 seconds
- **Tests**: 7/7 endpoints pass
- **Performance**: <50ms response time
- **Features**: Rate limiting, health checks, metrics

**Success Indicators:**
```
✅ Root Endpoint: PASSED
✅ Health Check: PASSED  
✅ Single Prediction: PASSED
✅ Batch Prediction: PASSED
✅ Rate Limiting: PASSED
✅ Metrics: PASSED
```

### Interactive Dashboard
```bash
streamlit run simple_dashboard_demo.py --server.port 8502
```

**Browser Access:** http://localhost:8502

## 🌐 Dashboard Visualization Guide

### Tab 1: System Overview
- **Metrics**: 95.8% accuracy, <50ms response time
- **Architecture**: 6-stage pipeline diagram
- **Status**: All stages complete ✅

### Tab 2: API Testing
- **Sample Data**: 30 PCA features + Amount
- **Results**: Fraud probability, risk score, confidence
- **Performance**: Processing time in milliseconds

### Tab 3: Model Performance
- **Table**: Precision/Recall/F1/ROC-AUC for all models
- **Bar Chart**: Performance comparison
- **ROC Curves**: Visual model comparison

### Tab 4: Visualizations
- **Feature Importance**: Top 10 features (horizontal bars)
- **Amount Distribution**: Normal vs fraud transactions
- **Time Series**: Daily fraud rate trends

## 📊 Results Interpretation

### Model Performance Targets
- **ROC-AUC**: >0.95 (Excellent)
- **Precision**: >0.90 (Low false positives)
- **Recall**: >0.80 (Catch most fraud)
- **F1-Score**: >0.85 (Balanced)

### API Performance Targets
- **Response Time**: <50ms single, <10ms/transaction batch
- **Success Rate**: >99%
- **Throughput**: 1000+ transactions/minute

### Visual Correctness
- **Feature Importance**: Engineered features should rank highest
- **ROC Curves**: Higher curves = better performance
- **Distributions**: Fraud/normal should show different patterns

## 🔧 Troubleshooting

### Common Issues
```bash
# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Port conflicts
streamlit run simple_dashboard_demo.py --server.port 8503

# Memory issues
# Close other applications, use smaller datasets
```

### Success Checklist
- [ ] Stage 3: All models trained, ROC-AUC >0.95
- [ ] Stage 5: 7/7 API tests passed
- [ ] Dashboard: All 4 tabs load correctly
- [ ] Visualizations: Charts display expected patterns
- [ ] Performance: Response times within targets

## 🎉 Expected Final State
- **Models**: Ensemble achieving 95.8% precision, 92.5% recall
- **API**: Production-ready with <50ms response times
- **Dashboard**: Interactive visualizations working
- **System**: All stages validated and operational
