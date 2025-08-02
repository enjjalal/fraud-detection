# Credit Scoring & Fraud Detection Model

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive machine learning project for credit scoring and fraud detection using advanced ensemble methods (XGBoost, LightGBM, CatBoost) with automated hyperparameter tuning and production-ready API deployment.

## 🚀 Features

- **Advanced ML Pipeline**: Multi-model ensemble with automated hyperparameter optimization
- **Model Interpretability**: SHAP explanations and feature importance analysis
- **Production API**: FastAPI with async endpoints, rate limiting, and comprehensive validation
- **Interactive Dashboard**: Streamlit web interface for model exploration
- **CI/CD Pipeline**: Automated testing and deployment with GitHub Actions
- **Docker Support**: Containerized application for easy deployment

## 📊 Models

- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting with categorical feature support
- **CatBoost**: Handling categorical features without preprocessing
- **Ensemble Methods**: Voting and stacking classifiers

## 🛠️ Tech Stack

- **ML/Data**: pandas, scikit-learn, xgboost, lightgbm, catboost, optuna
- **API**: FastAPI, uvicorn, pydantic
- **Visualization**: matplotlib, seaborn, plotly, streamlit
- **Model Interpretation**: SHAP, eli5
- **Deployment**: Docker, GitHub Actions

## 📁 Project Structure

```
fraud_detection/
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── models/            # ML model implementations
│   ├── api/               # FastAPI application
│   └── dashboard/         # Streamlit dashboard
├── notebooks/             # Jupyter notebooks for EDA
├── tests/                 # Unit and integration tests
├── docker/                # Docker configuration
├── .github/workflows/     # CI/CD pipelines
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd fraud_detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download dataset**
```bash
python src/data/download_data.py
```

4. **Train models**
```bash
python src/models/train.py
```

5. **Start API server**
```bash
uvicorn src.api.main:app --reload
```

6. **Launch dashboard**
```bash
streamlit run src/dashboard/app.py
```

### Docker Deployment

```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| XGBoost | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| LightGBM | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| CatBoost | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Ensemble | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

## 🔗 API Endpoints

- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information
- `GET /health` - Health check

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
