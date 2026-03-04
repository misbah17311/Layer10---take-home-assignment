#!/usr/bin/env python3
# launches the streamlit graph explorer
import subprocess, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
subprocess.run([sys.executable, "-m", "streamlit", "run", "viz/viz_app.py"])
