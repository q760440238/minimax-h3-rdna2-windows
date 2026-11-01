<#
  Reproducible, admin-free ROCm/HIP 7.2 runtime install for the Windows ROCm build
  of stable-diffusion.cpp.

  What it does:
    1. downloads the AMD HIP SDK bundle (needs the Clash/proxy if amd.com is blocked)
    2. pulls ROCm_Libs_RT.msi out of the embedded 7z payload (py7zr)
    3. extracts the MSI with an administrative install (no elevation required)
    4. copies hipblas/rocblas + Tensile kernels next to sd-cli.exe in bin\sd-rocm

  Prereq: pip install py7zr
  Usage:  .\extract_rocm.ps1 [-Proxy http://127.0.0.1:7897]
#>
param(
    [string]$Proxy = 'http://127.0.0.1:7897',
    [string]$WorkDir = 'E:\minimaxh3\bin\_rocm_work'
)

$ErrorActionPreference = 'Continue'
$url = 'https://download.amd.com/developer/eula/rocm-hub/AMD-Software-PRO-Edition-26.Q3-Win11-For-HIP.exe'
$exe = Join-Path $WorkDir 'hip-sdk.exe'
$msiOut = Join-Path $WorkDir 'msi'
$libOut = Join-Path $WorkDir 'libs'
$dst = 'E:\minimaxh3\bin\sd-rocm'

New-Item -ItemType Directory -Force -Path $WorkDir, $msiOut, $libOut | Out-Null

if (-not (Test-Path $exe)) {
    Write-Host "downloading HIP SDK bundle via $Proxy ..."
    & curl.exe -L --proxy $Proxy --retry 5 -A 'Mozilla/5.0' -o $exe $url
}
Write-Host ("bundle: {0:N0} MB" -f ((Get-Item $exe).Length / 1MB))

$py = @'
import io, os, sys, py7zr
src, outdir = sys.argv[1], sys.argv[2]
MAGIC = b"7z\xbc\xaf'\x1c"
offsets = []
with open(src, 'rb') as f:
    pos = 0
    while True:
        c = f.read(32 * 1024 * 1024)
        if not c: break
        i = c.find(MAGIC)
        while i != -1:
            offsets.append(pos + i); i = c.find(MAGIC, i + 1)
        pos += len(c)
assert offsets, 'no 7z payload found'
with open(src, 'rb') as f:
    f.seek(offsets[0]); bio = io.BytesIO(f.read())
targets = [n for n in py7zr.SevenZipFile(bio, mode='r').getnames() if n.lower().endswith('rocm_libs_rt.msi')]
print('payload targets:', targets)
with py7zr.SevenZipFile(bio, mode='r') as z:
    z.extract(path=outdir, targets=targets)
print('extracted to', outdir)
'@
$pyFile = Join-Path $WorkDir 'unpack.py'
Set-Content -Path $pyFile -Value $py -Encoding UTF8
python $pyFile $exe $msiOut

$msi = Get-ChildItem $msiOut -Recurse -Filter 'ROCm_Libs_RT.msi' | Select-Object -First 1
if (-not $msi) { throw 'ROCm_Libs_RT.msi not found in payload' }

Write-Host 'extracting MSI (administrative install, no elevation)...'
$p = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/a', "`"$($msi.FullName)`"", '/qn', "TARGETDIR=`"$libOut`"") -Wait -PassThru -NoNewWindow
if ($p.ExitCode -ne 0) { throw "msiexec failed: $($p.ExitCode)" }

$bin = Get-ChildItem $libOut -Recurse -Directory -Filter 'bin' |
       Where-Object { $_.FullName -match 'AMD\\ROCm' } | Select-Object -First 1
if (-not $bin) { throw 'ROCm bin directory not found after extraction' }

Write-Host "copying runtime from $($bin.FullName) -> $dst"
Copy-Item (Join-Path $bin.FullName '*.dll') $dst -Force
if (Test-Path (Join-Path $bin.FullName 'rocblas'))     { Copy-Item (Join-Path $bin.FullName 'rocblas') $dst -Recurse -Force }
if (Test-Path (Join-Path $bin.FullName 'hipblaslt'))   { Copy-Item (Join-Path $bin.FullName 'hipblaslt') $dst -Recurse -Force }

Push-Location $dst
& .\sd-cli.exe --list-devices
Pop-Location

Write-Host 'done. If a ROCm device is listed above, bin\sd-rocm is ready.' -ForegroundColor Green
