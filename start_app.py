import subprocess
import sys
import os

def run_streamlit():
    """Start Streamlit app with currently activated environment"""
    
    # Fix OpenMP conflict on macOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Streamlit app filename
    app_file = "user_interface.py"  # Change to your app filename
    
    # Check if app file exists
    if not os.path.exists(app_file):
        print(f"App file '{app_file}' not found!")
        print("Please update the filename in run.py")
        return
    
    # Build streamlit command
    cmd = [sys.executable, "-m", "streamlit", "run", app_file]
    
    print(f"Starting Streamlit app: {app_file}")
    print(f"Python path: {sys.executable}")
    print(f"Environment: {os.environ.get('VIRTUAL_ENV', 'System Python')}")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting Streamlit: {e}")
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")

if __name__ == "__main__":
    run_streamlit()