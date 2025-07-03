REM  Builds the frontend of the GUI application using pnpm.
@echo off
cd /d "%~dp0\..\frontend"
pnpm install
pnpm build
