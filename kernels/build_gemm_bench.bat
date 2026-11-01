@echo off
setlocal
REM Build the standalone ggml GEMM benchmark against the already-built HIP libs.

set SDK=E:\minimaxh3\bin\rocm-core\PFiles64\AMD\ROCm\7.2
set SRC=E:\minimaxh3\src\sdcpp\stable-diffusion.cpp-37bc88b5e97a213821949732ea66c58ae8ffc784
set BUILD=E:\minimaxh3\build-hip

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=%SDK%\bin;%PATH%"

REM ggml-base.lib was compiled with LLVM OpenMP (__kmpc_* ABI). The ROCm clang
REM ships no libomp, so use MSVC's LLVM OpenMP runtime import lib + DLL.
set "LIBOMP=%VCToolsInstallDir%lib\x64\libomp.lib"
if not exist "%LIBOMP%" (
  echo ERROR: libomp.lib not found at "%LIBOMP%"
  exit /b 1
)

REM Locate the matching runtime DLL (version dir varies with the VS redist).
set "LIBOMP_DLL="
for /d %%v in ("%VCToolsInstallDir%..\..\..\Redist\MSVC\*") do (
  if exist "%%v\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll" (
    set "LIBOMP_DLL=%%v\debug_nonredist\x64\Microsoft.VC143.OpenMP.LLVM\libomp140.x86_64.dll"
  )
)
if not defined LIBOMP_DLL (
  echo ERROR: libomp140.x86_64.dll not found under VS Redist\MSVC\*
  exit /b 1
)
copy /y "%LIBOMP_DLL%" "E:\minimaxh3\hip\libomp140.x86_64.dll" >nul

"%SDK%\bin\clang.exe" -x hip --offload-arch=gfx1030 ^
  --rocm-path="%SDK%" --hip-device-lib-path="%SDK%\amdgcn\bitcode" ^
  --driver-mode=g++ -fuse-ld=lld --ld-path="%SDK%\bin\lld-link.exe" ^
  -I"%SRC%\ggml\include" -I"%SRC%\ggml\src" -I"%SRC%\include" ^
  -O2 -std=gnu++17 -DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt ^
  "E:\minimaxh3\hip\gemm_bench.cpp" -x none ^
  "%BUILD%\ggml\src\ggml-hip\ggml-hip.lib" ^
  "%BUILD%\ggml\src\ggml-base.lib" ^
  -L"%SDK%\lib" -lamdhip64 -lhipblas -lrocblas ^
  "%LIBOMP%" ^
  --hip-link -o "E:\minimaxh3\hip\gemm_bench.exe"
exit /b %ERRORLEVEL%
