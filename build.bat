@echo off
REM Clean previous build
powershell Remove-Item -Recurse -Force dist\align -ErrorAction SilentlyContinue

REM Use conda environment's pyinstaller
.conda\Scripts\pyinstaller.exe align.spec --noconfirm