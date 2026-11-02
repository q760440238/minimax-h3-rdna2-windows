@echo off
setlocal
REM Build stable-diffusion.cpp with the HIP (ROCm) backend on Windows.
REM Toolchain: MSVC 2022 + Ninja (from VS) + ROCm 7.2 SDK extracted from the AMD installer.

set ROCM=E:\minimaxh3\bin\rocm-core\PFiles64\AMD\ROCm\7.2
set SRC=E:\minimaxh3\src\sdcpp\stable-diffusion.cpp-37bc88b5e97a213821949732ea66c58ae8ffc784
set BUILD=E:\minimaxh3\build-hip
set CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set PATH=%ROCM%\bin;%NINJA%;%PATH%
set HIP_PATH=%ROCM%
set ROCM_PATH=%ROCM%
set HIP_PLATFORM=amd

if "%1"=="configure" (
  "%CMAKE%" -S "%SRC%" -B "%BUILD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DSD_HIPBLAS=ON ^
    -DGGML_HIP=ON ^
    -DCMAKE_HIP_COMPILER="%ROCM%\bin\clang++.exe" ^
    -DCMAKE_HIP_ARCHITECTURES=gfx1030 ^
    -Dhip_DIR="%ROCM%\lib\cmake\hip" ^
  ^
    -DSD_SERVER_BUILD_FRONTEND=OFF %EXTRA_CMAKE% ^
    -DSD_BUILD_EXAMPLES=ON
  exit /b %ERRORLEVEL%
)

if "%1"=="build" (
  "%CMAKE%" --build "%BUILD%" -j 16
  exit /b %ERRORLEVEL%
)

echo usage: build_hip.bat configure ^| build
