#!/usr/bin/env python3
import os
import subprocess
import sys
import time
import importlib.util

def check_dependency(package_name):
    """Check if a Python package is installed"""
    return importlib.util.find_spec(package_name) is not None

def check_dependencies():
    """Check for required dependencies and install if missing"""
    required_packages = ["pandas", "plotly", "dash", "folium", "numpy"]
    missing_packages = [pkg for pkg in required_packages if not check_dependency(pkg)]
    
    if missing_packages:
        print(f"Missing required dependencies: {', '.join(missing_packages)}")
        install_cmd = f"conda install -y -c conda-forge {' '.join(missing_packages)}"
        print(f"Installing with command: {install_cmd}")
        
        try:
            subprocess.run(install_cmd, shell=True, check=True)
            print("Dependency installation completed. Please run this script again.")
            sys.exit(0)
        except subprocess.CalledProcessError:
            print("Error installing dependencies with conda.")
            print("Please install manually: conda install -c conda-forge pandas plotly dash folium numpy")
            sys.exit(1)
    
    return True

def check_prerequisites():
    """Check if the required directories and files exist"""
    # First check dependencies
    check_dependencies()
    
    # Check for chunks directory
    if not os.path.exists("chunks"):
        print("Creating 'chunks' directory...")
        os.makedirs("chunks", exist_ok=True)
        print("Please place your Wikipedia XML dump files (*.xml.bz2) in the 'chunks' directory.")
        return False
    
    # Check if there are any dump files
    dump_files = [f for f in os.listdir("chunks") if f.endswith(".xml.bz2")]
    if not dump_files:
        print("No XML dump files found in 'chunks' directory.")
        print("Please place your Wikipedia XML dump files (*.xml.bz2) in the 'chunks' directory.")
        return False
    
    return True

def run_main_processing():
    """Run the main.py script to process data"""
    print("Starting data processing with main.py...")
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
        print("Data processing completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Data processing failed: {e}")
        return False

def run_title_classification():
    """Run the cluster_titles.py script to classify titles"""
    if not os.path.exists("titles.txt"):
        print("Error: titles.txt not found. Cannot run classification.")
        return False
    
    print("Starting title classification with cluster_titles.py...")
    try:
        subprocess.run([sys.executable, "cluster_titles.py"], check=True)
        print("Title classification completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: Title classification failed: {e}")
        return False

def launch_dashboard():
    """Launch the dashboard.py application"""
    print("Launching dashboard...")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    
    try:
        # Check and create required directories
        os.makedirs("assets", exist_ok=True)
        os.makedirs("cache", exist_ok=True)
        
        # Launch dashboard as a separate process
        dashboard_process = subprocess.Popen([sys.executable, "dashboard.py"])
        
        print("Dashboard is running. Press Ctrl+C to stop.")
        # Keep the script running while dashboard is active
        while dashboard_process.poll() is None:
            time.sleep(1)
        
        print("Dashboard stopped.")
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        dashboard_process.terminate()
    except Exception as e:
        print(f"Error launching dashboard: {e}")

def main():
    print("=== Wikipedia Analysis Dashboard Setup ===")
    
    # Create temp directory for Spark
    os.makedirs("spark-temp", exist_ok=True)
    
    # Check dependencies first
    check_dependencies()
    
    # Check prerequisites
    have_dumps = check_prerequisites()
    if not have_dumps:
        user_input = input("Do you want to continue with the dashboard without processing new data? (y/n): ")
        if user_input.lower() not in ['y', 'yes']:
            print("Exiting...")
            return
    else:
        # Run data processing if prerequisites are met
        user_input = input("Do you want to process Wikipedia dump files? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            success = run_main_processing()
            if not success:
                print("Warning: Proceeding with dashboard, but data may be incomplete.")
        
        # Run title classification if titles.txt exists
        if os.path.exists("titles.txt"):
            user_input = input("Do you want to classify article titles? (y/n): ")
            if user_input.lower() in ['y', 'yes']:
                run_title_classification()
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main() 