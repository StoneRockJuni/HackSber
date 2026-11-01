@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Not found: .venv\Scripts\python.exe
    echo Create the environment:
    echo     python3.13 -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

rem --- find a free port, starting from 8501 -------------------------------
set PORT=8501
:findport
set BUSY=0
for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:"LISTENING" ^| findstr /c:":%PORT%"') do set BUSY=1
if "!BUSY!"=="1" (
    echo [WARN] Port %PORT% is busy, trying %PORT%+1 ...
    set /a PORT+=1
    if !PORT! gtr 8520 goto toomany
    goto findport
)
goto portfound

:toomany
echo [ERROR] No free port in range 8501-8520.
pause
exit /b 1

:portfound
echo ============================================================
echo  StroyKontur: http://localhost:%PORT%
echo  The browser tab opens automatically in ~5 seconds.
echo  If it does not open, copy this address into the browser:
echo      http://localhost:%PORT%
echo  To stop the server: press Ctrl+C in this window.
echo ============================================================

rem Open the browser after a short delay, while Streamlit boots.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 5; Start-Process 'http://localhost:%PORT%'"

".venv\Scripts\python.exe" -m streamlit run app.py --server.port %PORT% --server.address 127.0.0.1 --server.headless true

echo.
echo The server has stopped.
pause
