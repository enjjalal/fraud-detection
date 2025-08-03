#!/usr/bin/env python3
"""
Simple launcher for the Streamlit dashboard that fixes import issues
by adding the src directory to the Python path.
"""

import sys
import os
import subprocess

# Add the src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Set environment variable for Streamlit
os.environ['PYTHONPATH'] = src_dir + os.pathsep + os.environ.get('PYTHONPATH', '')

if __name__ == "__main__":
    # Run Streamlit with the dashboard
    dashboard_path = os.path.join(src_dir, 'dashboard', 'interpretability_dashboard.py')
    
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 
        dashboard_path,
        '--server.port', '8501',
        '--server.address', 'localhost'
    ]
    
    print("🚀 Starting Fraud Detection Dashboard...")
    print(f"📊 Dashboard will be available at: http://localhost:8501")
    print("=" * 60)
    
    subprocess.run(cmd)
