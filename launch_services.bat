@echo off
REM Quick launcher for all services (Windows)

echo ================================================================================
echo CREDIT SCORING MODEL - SERVICE LAUNCHER
echo ================================================================================
echo.
echo This will start all services:
echo   - MLflow UI (http://localhost:5000)
echo   - Dashboard (http://localhost:8501)
echo   - API Server (http://localhost:8000/docs)
echo.
echo Press Ctrl+C to stop all services
echo ================================================================================
echo.

cd /d "%~dp0"
poetry run python scripts/deployment/start_all.py

pause
