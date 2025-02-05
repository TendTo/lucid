@echo off
setlocal enabledelayedexpansion

REM Read name of the software from command line argument or use default 'unknown'
set "NAME=%~1"
if "%NAME%"=="" set "NAME=unknown"

REM Read version from command line argument or use default 'unknown.unknown.unknown'
set "VERSION=%~2"
if "%VERSION%"=="" set "VERSION=unknown.unknown.unknown"

REM Read the description from command line argument or use default 'description'
set "DESCRIPTION=%~3"
if "%DESCRIPTION%"=="" set "DESCRIPTION=description"

REM Split string into an array on '.' delimiter
for /f "tokens=1,2,3 delims=." %%a in ("%VERSION%") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
    set "REVISION=%%c"
)

REM Get repository status from bazel-out/stable-status.txt
for /f "tokens=2*" %%a in ('findstr /b /c:"STABLE_REPOSITORY_STATUS " bazel-out\stable-status.txt') do set "REPOSITORY_STATUS=%%a %%b"

REM Print version header
(
echo #pragma once
echo #define LUCID_PROGRAM_NAME    "%NAME%"
echo #define LUCID_VERSION_STRING  "%VERSION%"
echo #define LUCID_VERSION_FULL     %VERSION%
echo #define LUCID_VERSION_MAJOR    %MAJOR%
echo #define LUCID_VERSION_MINOR    %MINOR%
echo #define LUCID_VERSION_REVISION %REVISION%
echo #define LUCID_VERSION_REPOSTAT "%REPOSITORY_STATUS%"
echo #define LUCID_DESCRIPTION     "%DESCRIPTION%"
) > "%~4"
