#!/usr/bin/env python3
"""
Setup script for Diabetic Retinopathy Detection Backend
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Diabetic Retinopathy Detection Backend")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Determine activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
        pip_command = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_script = "source venv/bin/activate"
        pip_command = "venv/bin/pip"
    
    # Install requirements
    if not run_command(f"{pip_command} install -r requirements.txt", "Installing dependencies"):
        sys.exit(1)
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        env_content = """# Diabetic Retinopathy Detection API Configuration
FLASK_ENV=development
PORT=5000
SECRET_KEY=your-secret-key-change-in-production
DATABASE_URL=sqlite:///retinopathy.db
"""
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env file with default configuration")
    else:
        print("✅ .env file already exists")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run the application:")
    print("   python app.py")
    print("3. The API will be available at http://localhost:5000")

if __name__ == "__main__":
    main()
