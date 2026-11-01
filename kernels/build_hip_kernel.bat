@echo off
setlocal
REM Build HIP programs for gfx1030 with the ROCm SDK's clang (no hipcc wrapper).

set SDK=E:\minimaxh3\bin\rocm-core\PFiles64\AMD\ROCm\7.2
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%SDK%\bin;%PATH%"
set "HIP_PATH=%SDK%"

set SRC=%1
set OUT=%~n1.exe

"%SDK%\bin\clang.exe" ^
  -x hip --offload-arch=gfx1030 ^
  --hip-device-lib-path="%SDK%\amdgcn\bitcode" --rocm-path="%SDK%" ^
  --driver-mode=g++ -fuse-ld=lld --ld-path="%SDK%\bin\lld-link.exe" ^
  -I"%SDK%\include" -L"%SDK%\lib" --hip-link ^
  -O3 -DNDEBUG -std=c++17 ^
  "%SRC%" -o "%OUT%"
exit /b %ERRORLEVEL%
