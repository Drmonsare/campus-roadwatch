@echo off
setlocal
cd /d "%~dp0"
where node.exe >nul 2>nul
if errorlevel 1 (
  echo Node.js 18 or newer is required to run Campus RoadWatch.
  pause
  exit /b 1
)
if not exist "dist\index.html" (
  node scripts\build.mjs
  if errorlevel 1 (
    pause
    exit /b 1
  )
)
node scripts\serve.mjs --open
if errorlevel 1 pause
