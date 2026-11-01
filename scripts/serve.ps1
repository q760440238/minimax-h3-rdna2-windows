<#
  MiniMax-H3 local server (sd-server, embedded Web UI).
  The model is loaded once, so each request skips the ~20-30 s load time.

  Backend: ROCm (bin\sd-rocm) by default - measured faster on long clips.
  Use -Backend vulkan to force the Vulkan build.

  Usage:
    .\serve.ps1                 # http://127.0.0.1:1234
    .\serve.ps1 -Port 7860
    .\serve.ps1 -Backend vulkan

  In the Web UI, append this to the prompt to use the 4-step Turbo LoRA:
    <lora:minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16:1>
#>
param(
    [int]$Port = 1234,
    [string]$ListenIp = '127.0.0.1',
    [int]$MaxVram = 15,
    [ValidateSet('rocm', 'vulkan')][string]$Backend = 'rocm'
)

$M = 'E:\minimaxh3\models'

if ($Backend -eq 'rocm' -and (Test-Path 'E:\minimaxh3\bin\sd-rocm\sd-server.exe')) {
    $srv = 'E:\minimaxh3\bin\sd-rocm\sd-server.exe'
    $dev = 'ROCm0'
} else {
    $srv = 'E:\minimaxh3\bin\sd-vulkan\sd-server.exe'
    $dev = 'Vulkan0'
    $env:GGML_VK_ENABLE_MEMORY_PRIORITY = '1'
    $Backend = 'vulkan'
}

$cliArgs = @(
    '--diffusion-model', "$M\diffusion_models\minimax_h3_fl2va_pruned-Q4_K.gguf",
    '--vae',             "$M\vae\minimax_h3_video_vae_fp16.safetensors",
    '--audio-vae',       "$M\vae\minimax_h3_audio_vae_fp32.safetensors",
    '--llm',             "$M\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf",
    '--lora-model-dir',  "$M\loras",
    '--offload-to-cpu',
    '--max-vram', "$dev=$MaxVram",
    '--flash-attn', 'on',
    '--listen-ip', $ListenIp,
    '--listen-port', $Port,
    '--conditioning-cache-size', 4
)

Write-Host "backend=$Backend  ->  http://$ListenIp`:$Port   (Ctrl+C to stop)" -ForegroundColor Cyan
Write-Host "Web UI prompt suffix for Turbo LoRA: <lora:minimax_h3_fl2v_turbo_4step_v1.3_768p_comfyui_bf16:1>" -ForegroundColor DarkGray
& $srv @cliArgs
