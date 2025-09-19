@echo off
echo 🚀 Setting up Diabetic Retinopathy Detection Backend
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✅ Python detected

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 🔄 Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment and install requirements
echo 🔄 Installing dependencies...
call venv\Scripts\activate.bat
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo 🔄 Creating .env file...
    (
        echo # Diabetic Retinopathy Detection API Configuration
        echo FLASK_ENV=development
        echo PORT=5000
        echo SECRET_KEY=your-secret-key-change-in-production
        echo DATABASE_URL=sqlite:///retinopathy.db
    ) > .env
    echo ✅ .env file created
) else (
    echo ✅ .env file already exists
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo Next steps:
echo 1. Run the application: python app.py
echo 2. The API will be available at http://localhost:5000
echo 3. Test the API: python test_api.py
echo.
pause
