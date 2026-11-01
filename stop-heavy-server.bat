@echo off
rem Stop the Local Heavy server: kill whatever LISTENS on 8020.
setlocal
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8020" ^| findstr LISTENING') do (
  echo Stopping PID %%p
  taskkill /PID %%p /F >nul 2>&1
)
echo Done.
