<#
  MiniMax-H3 local generation - backend and kernel-path aware.

  Key findings from 2026-10-31 profiling (see KERNEL-ANALYSIS.md):

  * ROCm (HIP 7.2) is the default backend: real VRAM reporting, conv3d direct on.
  * Applying a LoRA switches sd.cpp to a different kernel path.
      - On ROCm it is 3-7x SLOWER per step (47 s vs 6.7 s at 864x480/22f).
      - On Vulkan it is 6x FASTER per step (70 s vs 11.6 s) because it avoids the
        slow Q4_K MMQ shader. That is why Vulkan needs a no-op LoRA.
    => ROCm presets do NOT use a LoRA. Use -Turbo only when you want 4-step sampling.
  * --mmap saves ~8% (skips copying weights that the GPU can map directly).

  Measured end-to-end (864x480, 22f = 0.9 s of video, ROCm, no LoRA, mmap):
      8 steps  -> 85 s
     20 steps  -> 152 s
  512x288, 56f = 2.3 s of video, ROCm, no LoRA:
      8 steps  -> ~90 s
     20 steps  -> ~150 s
  864x480, 56f = 2.3 s of video (attention-heavy): need Turbo LoRA + 4 steps -> 492 s

  Presets:
    draft     512x288, 56f,  8 steps, no LoRA      ~90 s
    fast      864x480, 22f,  8 steps, no LoRA      ~85 s
    quality   864x480, 22f, 20 steps, no LoRA     ~152 s
    longhd    864x480, 56f,  4 steps + Turbo      ~492 s

  Examples:
    .\gen.ps1 -Prompt "..." -Preset fast
    .\gen.ps1 -Prompt "..." -Preset longhd
    .\gen.ps1 -Prompt "..." -Preset quality -Frames 39

  -Frames must sit on the 17k+5 grid (5/22/39/56/73...). Duration = Frames/24 s.
  Attention cost scales with tokens^2, tokens = frames/4 * H/32 * W/32.
  Keep prompts descriptive (>=25 words): short prompts measured ~2.5x slower.
#>
param(
    [Parameter(Mandatory = $true)][string]$Prompt,
    [ValidateSet('draft', 'fast', 'quality', 'longhd')][string]$Preset = 'fast',
    [ValidateSet('auto', 'rocm', 'vulkan')][string]$Backend = 'auto',
    [int]$W = 0,
    [int]$H = 0,
    [int]$Frames = 0,
    [int]$Steps = 0,
    [int]$Seed = 42,
    [double]$CfgScale = 1.0,
    [string]$Out = "",
    [switch]$Turbo,
    [switch]$NoFlashAttn,
    [switch]$FlashAttn,
    [switch]$NoMmap,
    [switch]$NoVaeTiling,
    [switch]$VaeTiling,
    [int]$MaxVram = 0,
    [string]$ExtraArgs = "",
    [string]$Te = '',
    [switch]$NoMaxVram,
    [int]$MaxVramCap = 12
)

$M = 'E:\minimaxh3\models'
$teDefault = "$M\text_encoders\qwen3vl_32b_minimax_h3-Q2_K_M.gguf"
if (-not (Test-Path $teDefault)) { throw "text encoder not found: $teDefault  (run .\download_models.ps1)" }

# --- presets ---------------------------------------------------------------
$useLora = $false
$useTurbo = $false
switch ($Preset) {
    'draft'   { if ($W -eq 0) { $W = 512 }; if ($H -eq 0) { $H = 288 }; if ($Frames -eq 0) { $Frames = 56 }; if ($Steps -eq 0) { $Steps = 8 } }
    'fast'    { if ($W -eq 0) { $W = 864 }; if ($H -eq 0) { $H = 480 }; if ($Frames -eq 0) { $Frames = 22 }; if ($Steps -eq 0) { $Steps = 8 } }
    'quality' { if ($W -eq 0) { $W = 864 }; if ($H -eq 0) { $H = 480 }; if ($Frames -eq 0) { $Frames = 22 }; if ($Steps -eq 0) { $Steps = 20 } }
    'longhd'  { if ($W -eq 0) { $W = 864 }; if ($H -eq 0) { $H = 480 }; if ($Frames -eq 0) { $Frames = 56 }; if ($Steps -eq 0) { $Steps = 8 } }
}
if ($Turbo) { $useLora = $true; $useTurbo = $true }

# --- backend selection -----------------------------------------------------
$hasRocm = Test-Path 'E:\minimaxh3\bin\sd-rocm\sd-cli.exe'
if ($Backend -eq 'auto') { $Backend = if ($hasRocm) { 'rocm' } else { 'vulkan' } }
if ($Backend -eq 'rocm' -and -not $hasRocm) { $Backend = 'vulkan' }

if ($Backend -eq 'rocm') {
    # Prefer the self-built binary: it carries the Q4_K J=128 MMQ tile for gfx1030
    # (+7.9% kernel, -4.7% end-to-end). Fall back to the official release, which is
    # self-contained (no PATH setup needed).
    $custom = 'E:\minimaxh3\build-hip\bin\sd-cli.exe'
    $sdkBin = 'E:\minimaxh3\bin\rocm-core\PFiles64\AMD\ROCm\7.2\bin'
    if ((Test-Path $custom) -and (Test-Path $sdkBin) -and -not $NoCustomBuild) {
        $sd = $custom
        $env:PATH = "$sdkBin;$env:PATH"
    } else {
        $sd = 'E:\minimaxh3\bin\sd-rocm\sd-cli.exe'
    }
    $dev = 'ROCm0'
} else {
    $sd = 'E:\minimaxh3\bin\sd-vulkan\sd-cli.exe'
    $dev = 'Vulkan0'
    $env:GGML_VK_ENABLE_MEMORY_PRIORITY = '1'
    # Vulkan only reaches the fast GEMM path with a LoRA applied; use a no-op one.
    if (-not $useLora -and (Test-Path "$M\loras\minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16.safetensors")) { $useLora = $true; $useTurbo = $false }
    if ($MaxVram -eq 0) { $MaxVram = 15 }
}

if (-not $Out) { $Out = "E:\minimaxh3\out\h3_$Preset.webm" }
$dir = Split-Path $Out -Parent
if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

$loraSpec = ''
if ($useTurbo) { $loraSpec = 'minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16:1' }
elseif ($useLora) { $loraSpec = 'minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16:0' }
$p = if ($loraSpec) { "$Prompt<lora:$loraSpec>" } else { $Prompt }

if ($Prompt.Split(' ').Count -lt 25) {
    Write-Host 'WARNING: short prompt (<25 words) - measured ~2.5x slower on this setup.' -ForegroundColor Yellow
}

$cliArgs = @(
    '-M', 'vid_gen',
    '--diffusion-model', "$M\diffusion_models\minimax_h3_fl2va_pruned-Q4_K.gguf",
    '--vae',             "$M\vae\minimax_h3_video_vae_fp16.safetensors",
    '--audio-vae',       "$M\vae\minimax_h3_audio_vae_fp32.safetensors",
    '--llm',             $(if ($Te) { $Te } else { $teDefault }),
    '-p', $p,
    '--cfg-scale', $CfgScale,
    '--steps', $Steps,
    '-s', $Seed,
    '--rng', 'cpu',
    '-W', $W, '-H', $H,
    '--fps', 24,
    '--video-frames', $Frames,
    '--offload-to-cpu'
    # lora dir appended below only when a LoRA is used
)
# WDDM/VRAM budget guard. With ~11.4 GB of weights staged, activations larger than
# ~600 MB (long clips at high resolution) get demoted to shared memory by the OS and
# elementwise ops fall to PCIe bandwidth (measured: SCALE 242 -> 9 GB/s). Capping the
# staged weight VRAM leaves room for the activations; only pays off above ~5500 tokens.
$latentT   = [math]::Max(1, [int][math]::Round(5.0 * $Frames / 17.0))
$estTokens = $latentT * [int][math]::Floor($W / 32) * [int][math]::Floor($H / 32)
if ($Backend -eq 'rocm' -and $estTokens -ge 5500 -and -not $NoMaxVram) {
    $cliArgs += @('--max-vram', "$dev=$MaxVramCap")
}
if ($loraSpec) { $cliArgs += @('--lora-model-dir', "$M\loras") }
$cliArgs += @('-o', $Out)
# Flash attention policy (re-measured 2026-11-02 AFTER the WDDM/VRAM fix):
#   ROCm  : FA is neutral at 22f (49.5 vs 49.7 s) and 25% faster at 56f
#           (184.4 -> 138.4 s); sd-cli's "auto" picks the slow path at 56f,
#           so force it on once the token count gets large.
#   Vulkan: FA must stay on (FA-off at 22f measured 70 s/step vs 40).
$useFa = ($Backend -eq 'vulkan') -or ($estTokens -ge 4000)
if ($FlashAttn) { $useFa = $true }
if ($NoFlashAttn) { $useFa = $false }
if ($useFa) { $cliArgs += @('--flash-attn', 'on') }
if (-not $NoMmap) { $cliArgs += '--mmap' }
# VAE tiling only pays off for long clips (56f: -4%; 22f: +2%).
# VAE spatial tiling is a LOSS with the current build/config: measured +15.5 s at 22f
# and +4.4 s at 56f (decode 25.0 vs 20.6 s). Keep it off unless explicitly requested.
if ($VaeTiling) { $cliArgs += '--vae-tiling' }
if ($MaxVram -gt 0) { $cliArgs += @('--max-vram', "$dev=$MaxVram") }
if ($ExtraArgs) { $cliArgs += ($ExtraArgs -split '\s+') }

Write-Host ("preset={0} backend={1} {2}x{3}x{4}f steps={5} lora={6} fa={7} mmap={8}" -f `
    $Preset, $Backend, $W, $H, $Frames, $Steps, $(if($loraSpec){$loraSpec}else{'-'}), ($useFa), (-not $NoMmap)) -ForegroundColor Cyan

$sw = [Diagnostics.Stopwatch]::StartNew()
& $sd @cliArgs 2>&1 | Out-File -FilePath "${Out}.log" -Encoding utf8
$code = $LASTEXITCODE
$sw.Stop()
if ($code -eq 0) {
    Write-Host ("OK  {0}  ({1:N1}s)" -f $Out, $sw.Elapsed.TotalSeconds) -ForegroundColor Green
} else {
    Write-Host ("FAILED exit={0}  (see {1}.log)" -f $code, $Out) -ForegroundColor Red
}
