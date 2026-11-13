@echo off
REM ForgetLab local launcher (Windows)
REM Run from the repo root: scripts\run_local.bat

setlocal

REM 1. Check Python
where python >nul 2>&1 || (echo [error] python not found & exit /b 1)

REM 2. Activate venv if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo [launcher] activated .venv
)

REM 3. Check for HF token
if not defined HF_TOKEN (
    echo [warn] HF_TOKEN is not set. Set it before loading Llama-3.2-1B.
    echo   set HF_TOKEN=hf_...
)

REM 4. Parse arguments
set SHARE=
set PORT=7860
:parse
if "%1"=="--share" (set SHARE=--share & shift & goto parse)
if "%1"=="--port"  (set PORT=%2 & shift & shift & goto parse)

REM 5. Launch dashboard
echo [launcher] Starting ForgetLab dashboard on port %PORT%...
python -m forgetlab.app --port %PORT% %SHARE%

