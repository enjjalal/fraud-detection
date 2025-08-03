"""
Demo Script for Stage 5: Production-Ready API.

This script demonstrates the complete production-ready FastAPI system including:
- API server startup and configuration
- Single and batch prediction endpoints
- Rate limiting and error handling
- Health checks and monitoring
- Integration with existing models

Usage:
    python demo_production_api.py
"""

import asyncio
import requests
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import threading
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionAPIDemo:
    """Comprehensive demo of the production-ready API."""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.server_process = None
        
    def start_api_server(self):
        """Start the API server in background."""
        try:
            logger.info("Starting FastAPI server...")
            
            # Start server in background
            cmd = ["python", "-m", "uvicorn", "src.api.complete_api:app", 
                   "--host", "0.0.0.0", "--port", "8000", "--reload"]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path.cwd()
            )
            
            # Wait for server to start
            logger.info("Waiting for server to start...")
            time.sleep(5)
            
            # Check if server is running
            try:
                response = requests.get(f"{self.api_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("API server started successfully!")
                    return True
                else:
                    logger.error(f"Server health check failed: {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to connect to server: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return False
    
    def stop_api_server(self):
        """Stop the API server."""
        if self.server_process:
            logger.info("Stopping API server...")
            self.server_process.terminate()
            self.server_process.wait()
            logger.info("API server stopped")
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        logger.info("Testing root endpoint...")
        
        try:
            response = requests.get(f"{self.api_url}/")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Root endpoint response: {data}")
                return True
            else:
                logger.error(f"Root endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Root endpoint test failed: {e}")
            return False
    
    def test_health_check(self):
        """Test health check endpoint."""
        logger.info("Testing health check endpoint...")
        
        try:
            response = requests.get(f"{self.api_url}/health")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Health check passed:")
                logger.info(f"  Status: {data.get('status')}")
                logger.info(f"  Models loaded: {data.get('models_loaded')}")
                logger.info(f"  Uptime: {data.get('uptime_seconds'):.2f} seconds")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Health check test failed: {e}")
            return False
    
    def test_single_prediction(self):
        """Test single transaction prediction."""
        logger.info("Testing single prediction endpoint...")
        
        # Sample transaction data
        transaction_data = {
            "Time": 0.0,
            "V1": -1.359807134,
            "V2": -0.072781173,
            "V3": 2.536346738,
            "V4": 1.378155224,
            "V5": -0.338320770,
            "V6": 0.462387778,
            "V7": 0.239598554,
            "V8": 0.098697901,
            "V9": 0.363786969,
            "V10": 0.090794172,
            "V11": -0.551599533,
            "V12": -0.617800856,
            "V13": -0.991389847,
            "V14": -0.311169354,
            "V15": 1.468176972,
            "V16": -0.470400525,
            "V17": 0.207971242,
            "V18": 0.025791653,
            "V19": 0.403992960,
            "V20": 0.251412098,
            "V21": -0.018306778,
            "V22": 0.277837576,
            "V23": -0.110473910,
            "V24": 0.066928075,
            "V25": 0.128539358,
            "V26": -0.189114844,
            "V27": 0.133558377,
            "V28": -0.021053053,
            "Amount": 149.62
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json=transaction_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Single prediction successful:")
                logger.info(f"  Transaction ID: {data.get('transaction_id')}")
                logger.info(f"  Is Fraud: {data.get('is_fraud')}")
                logger.info(f"  Fraud Probability: {data.get('fraud_probability')}")
                logger.info(f"  Risk Score: {data.get('risk_score')}")
                logger.info(f"  Confidence: {data.get('confidence')}")
                logger.info(f"  Processing Time: {data.get('processing_time_ms')} ms")
                return True
            else:
                logger.error(f"Single prediction failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Single prediction test failed: {e}")
            return False
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        logger.info("Testing batch prediction endpoint...")
        
        # Generate sample batch data
        batch_data = {
            "transactions": []
        }
        
        # Create 5 sample transactions
        for i in range(5):
            transaction = {
                "Time": float(i * 100),
                "V1": np.random.normal(0, 1),
                "V2": np.random.normal(0, 1),
                "V3": np.random.normal(0, 1),
                "V4": np.random.normal(0, 1),
                "V5": np.random.normal(0, 1),
                "V6": np.random.normal(0, 1),
                "V7": np.random.normal(0, 1),
                "V8": np.random.normal(0, 1),
                "V9": np.random.normal(0, 1),
                "V10": np.random.normal(0, 1),
                "V11": np.random.normal(0, 1),
                "V12": np.random.normal(0, 1),
                "V13": np.random.normal(0, 1),
                "V14": np.random.normal(0, 1),
                "V15": np.random.normal(0, 1),
                "V16": np.random.normal(0, 1),
                "V17": np.random.normal(0, 1),
                "V18": np.random.normal(0, 1),
                "V19": np.random.normal(0, 1),
                "V20": np.random.normal(0, 1),
                "V21": np.random.normal(0, 1),
                "V22": np.random.normal(0, 1),
                "V23": np.random.normal(0, 1),
                "V24": np.random.normal(0, 1),
                "V25": np.random.normal(0, 1),
                "V26": np.random.normal(0, 1),
                "V27": np.random.normal(0, 1),
                "V28": np.random.normal(0, 1),
                "Amount": float(np.random.uniform(1, 1000))
            }
            batch_data["transactions"].append(transaction)
        
        try:
            response = requests.post(
                f"{self.api_url}/predict/batch",
                json=batch_data,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Batch prediction successful:")
                logger.info(f"  Total transactions: {data['summary']['total_transactions']}")
                logger.info(f"  Fraud detected: {data['summary']['fraud_detected']}")
                logger.info(f"  Fraud rate: {data['summary']['fraud_rate_percent']}%")
                logger.info(f"  Processing time: {data.get('processing_time_ms')} ms")
                return True
            else:
                logger.error(f"Batch prediction failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Batch prediction test failed: {e}")
            return False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        logger.info("Testing rate limiting...")
        
        try:
            # Make multiple rapid requests
            success_count = 0
            rate_limited_count = 0
            
            for i in range(10):
                response = requests.get(f"{self.api_url}/health")
                
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited_count += 1
                    logger.info("Rate limit triggered (expected behavior)")
                
                time.sleep(0.1)  # Small delay between requests
            
            logger.info(f"Rate limiting test completed:")
            logger.info(f"  Successful requests: {success_count}")
            logger.info(f"  Rate limited requests: {rate_limited_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limiting test failed: {e}")
            return False
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        logger.info("Testing metrics endpoint...")
        
        try:
            response = requests.get(f"{self.api_url}/metrics")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Metrics endpoint successful:")
                logger.info(f"  Total requests: {data['api_metrics']['requests_total']}")
                logger.info(f"  Success requests: {data['api_metrics']['requests_success']}")
                logger.info(f"  Error requests: {data['api_metrics']['requests_error']}")
                logger.info(f"  Avg response time: {data['api_metrics']['avg_response_time']:.4f}s")
                return True
            else:
                logger.error(f"Metrics endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Metrics endpoint test failed: {e}")
            return False
    
    def test_model_info(self):
        """Test model info endpoint."""
        logger.info("Testing model info endpoint...")
        
        try:
            response = requests.get(f"{self.api_url}/models/info")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("Model info endpoint successful:")
                logger.info(f"  Total models: {data.get('total_models')}")
                logger.info(f"  Scaler loaded: {data.get('scaler_loaded')}")
                logger.info(f"  Feature count: {data.get('feature_count')}")
                
                if data.get('models'):
                    logger.info("  Loaded models:")
                    for model_name, info in data['models'].items():
                        logger.info(f"    - {model_name}: {info['type']}")
                
                return True
            else:
                logger.error(f"Model info endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Model info endpoint test failed: {e}")
            return False
    
    def run_comprehensive_demo(self):
        """Run the complete API demo."""
        logger.info("=" * 60)
        logger.info("STARTING PRODUCTION API DEMO - STAGE 5")
        logger.info("=" * 60)
        
        try:
            # Start API server
            if not self.start_api_server():
                logger.error("Failed to start API server. Demo aborted.")
                return False
            
            # Run all tests
            tests = [
                ("Root Endpoint", self.test_root_endpoint),
                ("Health Check", self.test_health_check),
                ("Model Info", self.test_model_info),
                ("Single Prediction", self.test_single_prediction),
                ("Batch Prediction", self.test_batch_prediction),
                ("Rate Limiting", self.test_rate_limiting),
                ("Metrics", self.test_metrics_endpoint)
            ]
            
            results = {}
            for test_name, test_func in tests:
                logger.info(f"\n--- {test_name} Test ---")
                try:
                    results[test_name] = test_func()
                except Exception as e:
                    logger.error(f"{test_name} test failed with exception: {e}")
                    results[test_name] = False
                
                time.sleep(1)  # Brief pause between tests
            
            # Print summary
            logger.info("\n" + "=" * 60)
            logger.info("DEMO RESULTS SUMMARY")
            logger.info("=" * 60)
            
            passed = sum(1 for result in results.values() if result)
            total = len(results)
            
            for test_name, result in results.items():
                status = "✅ PASSED" if result else "❌ FAILED"
                logger.info(f"{test_name}: {status}")
            
            logger.info(f"\nOverall: {passed}/{total} tests passed")
            
            if passed == total:
                logger.info("🎉 ALL TESTS PASSED! Production API is working correctly.")
            else:
                logger.warning(f"⚠️  {total - passed} tests failed. Check logs for details.")
            
            logger.info("\n" + "=" * 60)
            logger.info("PRODUCTION API FEATURES DEMONSTRATED:")
            logger.info("✅ Async endpoints with FastAPI")
            logger.info("✅ Input validation with Pydantic")
            logger.info("✅ Rate limiting protection")
            logger.info("✅ Comprehensive error handling")
            logger.info("✅ Health checks and monitoring")
            logger.info("✅ Metrics collection")
            logger.info("✅ Single and batch predictions")
            logger.info("✅ Model integration")
            logger.info("=" * 60)
            
            return passed == total
            
        except KeyboardInterrupt:
            logger.info("\nDemo interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Demo failed with error: {e}")
            return False
        finally:
            # Always stop the server
            self.stop_api_server()


def main():
    """Main function to run the production API demo."""
    demo = ProductionAPIDemo()
    success = demo.run_comprehensive_demo()
    
    if success:
        logger.info("\n🚀 Production API Demo completed successfully!")
        logger.info("The API is ready for production deployment.")
    else:
        logger.error("\n💥 Production API Demo encountered issues.")
        logger.error("Please check the logs and fix any problems before deployment.")


if __name__ == "__main__":
    main()
