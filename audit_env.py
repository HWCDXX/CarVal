import sys
import platform

def audit_environment():
    print("==================================================")
    print("        CARVAL ENVIRONMENT INFRASTRUCTURE AUDIT   ")
    print("==================================================")
    
    # Core Language & OS Setup
    print(f"🐍 Python Engine Version : {platform.python_version()}")
    print(f"💻 Host Operating System : {platform.system()} ({platform.release()})")
    print("--------------------------------------------------")

    # Critical Toolchain Packages
    packages = [
        "streamlit",
        "joblib",
        "pandas",
        "numpy",
        "scikit-learn"
    ]
    
    for package in packages:
        try:
            if package == "scikit-learn":
                import sklearn
                print(f"📦 scikit-learn Version  : {sklearn.__version__}")
            else:
                mod = __import__(package)
                print(f"📦 {package:<20} Version  : {mod.__version__}")
        except ImportError:
            print(f"❌ {package:<20} : NOT INSTALLED IN CURRENT ENVIRONMENT")
            
    print("==================================================")

if __name__ == "__main__":
    audit_environment()
