@echo off
title Trading Bot - Starting...
cd /d "%~dp0"

set DOCKER="%LOCALAPPDATA%\Programs\DockerDesktop\resources\bin\docker.exe"
set OLLAMA_APP="%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
set TWS=C:\Jts\tws.exe
set PYTHON=%~dp0venv\Scripts\python.exe

echo.
echo  ============================================
echo    Trading Bot - Long-Duration Compounding
echo  ============================================
echo.

:: ── 1. Docker ────────────────────────────────────────────────────────────────
echo  [1/5] Checking Docker...
%DOCKER% info >nul 2>&1
if not errorlevel 1 goto docker_ready

echo        Docker not running - starting Docker Desktop...
start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
echo        Waiting up to 90s for Docker to be ready...
set tries=0

:wait_docker
timeout /t 5 /nobreak >nul
%DOCKER% info >nul 2>&1
if not errorlevel 1 goto docker_ready
set /a tries+=1
if %tries% lss 18 goto wait_docker
echo        ERROR: Docker did not start. Open Docker Desktop and try again.
pause
exit /b 1

:docker_ready
echo        Docker is ready.

:: ── 2. Redis ─────────────────────────────────────────────────────────────────
echo  [2/5] Starting Redis...
%DOCKER% start redis >nul 2>&1
if not errorlevel 1 goto redis_ready
%DOCKER% run -d --name redis -p 6379:6379 redis:alpine >nul 2>&1

:redis_ready
echo        Redis running.

:: ── 3. Ollama ─────────────────────────────────────────────────────────────────
echo  [3/5] Checking Ollama...
curl -s --max-time 2 http://localhost:11434/api/tags >nul 2>&1
if not errorlevel 1 goto ollama_ready
echo        Starting Ollama...
start "" %OLLAMA_APP%
timeout /t 6 /nobreak >nul

:ollama_ready
echo        Ollama running.

:: ── 4. IBKR TWS ──────────────────────────────────────────────────────────────
echo  [4/5] Launching IBKR TWS...
start "" "%TWS%"
echo        TWS launched - log in with paper credentials.

:: ── 5. Trading Bot ───────────────────────────────────────────────────────────
echo  [5/5] Starting Trading Bot and all services...
start /min "Trading Bot" "%PYTHON%" "%~dp0main.py"

:: ── Open dashboard ────────────────────────────────────────────────────────────
echo.
echo  Waiting 14s for dashboard to initialize...
timeout /t 14 /nobreak >nul
start "" "http://localhost:8501"

echo.
echo  ============================================
echo    All done! Dashboard opening now.
echo    This window will close in 5 seconds.
echo  ============================================
timeout /t 5 /nobreak >nul
exit
