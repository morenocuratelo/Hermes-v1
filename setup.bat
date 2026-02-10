@echo off
setlocal
cd /d "%~dp0"
call "%~dp0SETUP_LAB.bat"
exit /b %errorlevel%
