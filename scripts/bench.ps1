<#
  Benchmark harness for MiniMax-H3 on Vulkan.
  Fixed: 864x480 / 24fps / seed 42 / same prompt, so configs are comparable.

  Examples:
    .\bench.ps1 -Name baseline -Steps 12
    .\bench.ps1 -Name turbo4 -Steps 4 -Lora "minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16:1"
    .\bench.ps1 -Name pinned -Steps 4 -NoOffload -ParamsBackend "diffusion=Vulkan0,te=cpu,vae=cpu"
#>
param(
    [Parameter(Mandatory = $true)][string]$Name,
    [int]$Steps = 12,
    [int]$Frames = 22,
    [int]$W = 864,
    [int]$H = 480,
    [int]$Seed = 42,
    [string]$Prompt = "A cute American Shorthair silver tabby kitten surfs on a tropical ocean wave, riding a white surfboard with the clear text 'sd.cpp' on it. Cinematic tracking shot, realistic water, bright sunlight, smooth motion. Add upbeat tropical surf-rock background music with cheerful drums and guitar.",
    [string]$Extra = "",
    [string]$Lora = "",
    [string]$Diffusion = "",
    [string]$ParamsBackend = "",
    [string]$Sd = 'E:\minimaxh3\bin\sd-vulkan\sd-cli.exe',
    [string]$DeviceName = 'Vulkan0',
    [string]$Llm = '',
    [switch]$FlashAttn,
    [switch]$NoOffload,
    [int]$MaxVram = 15
)

$sd = $Sd
$M  = 'E:\minimaxh3\models'
if (-not $Diffusion) { $Diffusion = "$M\diffusion_models\minimax_h3_fl2va_pruned-Q4_K.gguf" }
$out = "E:\minimaxh3\out\bench_$Name.webm"
$log = "E:\minimaxh3\out\bench_$Name.log"

$p = $Prompt
if ($Lora) { $p = "$Prompt<lora:$Lora>" }

$argv = @(
    '-M','vid_gen',
    '--diffusion-model', $Diffusion,
    '--vae', "$M\vae\minimax_h3_video_vae_fp16.safetensors",
    '--audio-vae', "$M\vae\minimax_h3_audio_vae_fp32.safetensors",
    '--llm', $(if ($Llm) { $Llm } else { "$M\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf" }),
    '-p', $p,
    '--cfg-scale', '1.0',
    '--steps', $Steps,
    '-s', $Seed,
    '--rng', 'cpu',
    '-W', $W, '-H', $H, '--fps', 24, '--video-frames', $Frames,
    '--max-vram', "$DeviceName=$MaxVram",
    '--lora-model-dir', "$M\loras",
    '-v',
    '-o', $out
)
if ($FlashAttn) { $argv += @('--flash-attn', 'on') }
if (-not $NoOffload) { $argv += @('--offload-to-cpu') }
if ($MaxVram -gt 0) { $argv += @('--max-vram', "$DeviceName=$MaxVram") }
if ($ParamsBackend) { $argv += @('--params-backend', $ParamsBackend) }
if ($Extra) { $argv += ($Extra -split '\s+') }

$sw = [Diagnostics.Stopwatch]::StartNew()
& $sd @argv 2>&1 | Out-File -FilePath $log -Encoding utf8
$sw.Stop()

$txt = (Get-Content $log -Raw) -replace '\x1b\[[0-9;]*[A-Za-z]', ''
$sample = [regex]::Matches($txt, 'sampling completed, taking ([\d.]+)s') | ForEach-Object { [double]$_.Groups[1].Value } | Select-Object -Last 1
$total  = [regex]::Matches($txt, 'generate_video completed in ([\d.]+)s') | ForEach-Object { [double]$_.Groups[1].Value } | Select-Object -Last 1
$err = ($txt -split "`r?`n" | Where-Object { $_ -match '^\[E\]' } | Select-Object -Last 2) -join ' | '
$stage = [regex]::Match($txt, 'staged compute params \(10975\.93 MB').Success

"==== {0} ====" -f $Name
"steps={0} frames={1} flash={2} nooffload={3} params_backend={4} lora={5}" -f $Steps, $Frames, $FlashAttn.IsPresent, $NoOffload.IsPresent, $ParamsBackend, $Lora
if ($sample) { "sampling : {0,7:N1} s   ({1:N1} s/step)" -f $sample, ($sample / $Steps) } else { "sampling : FAILED" }
if ($total)  { "total    : {0,7:N1} s" -f $total }
"wall     : {0,7:N1} s   full_dit_staged={1}" -f $sw.Elapsed.TotalSeconds, $stage
if ($err) { "errors   : $err" }
