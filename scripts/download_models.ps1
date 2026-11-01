# MiniMax-H3 权重下载（ModelScope 镜像，20+MB/s）
$ErrorActionPreference = 'Continue'
$base = 'https://modelscope.cn/models/unsloth/MiniMax-H3-GGUF/resolve/master'
$root = 'E:\minimaxh3\models'
$files = @(
    @{ url = "$base/minimax_h3_fl2va_pruned-Q4_K.gguf";          out = "$root\diffusion_models\minimax_h3_fl2va_pruned-Q4_K.gguf" },
    @{ url = "$base/qwen3vl_32b_minimax_h3-Q4_K_M.gguf";         out = "$root\text_encoders\qwen3vl_32b_minimax_h3-Q4_K_M.gguf" },
    @{ url = "$base/vae/minimax_h3_video_vae_fp16.safetensors";  out = "$root\vae\minimax_h3_video_vae_fp16.safetensors" },
    @{ url = "$base/vae/minimax_h3_audio_vae_fp32.safetensors";  out = "$root\vae\minimax_h3_audio_vae_fp32.safetensors" }
)
foreach ($f in $files) {
    $dir = Split-Path $f.out -Parent
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $sw = [Diagnostics.Stopwatch]::StartNew()
    Write-Output ("[{0:HH:mm:ss}] START {1}" -f (Get-Date), (Split-Path $f.out -Leaf))
    & curl.exe -L --retry 20 --retry-delay 5 --retry-all-errors -C - -o $f.out $f.url 2>&1 |
        Select-String -Pattern 'curl:' | ForEach-Object { $_.Line }
    $sw.Stop()
    $len = (Get-Item $f.out -ErrorAction SilentlyContinue).Length
    $mb = [math]::Round($len / 1MB, 1)
    $rate = if ($sw.Elapsed.TotalSeconds -gt 0) { [math]::Round($mb / $sw.Elapsed.TotalSeconds, 2) } else { 0 }
    Write-Output ("[{0:HH:mm:ss}] DONE  {1}  {2} MB  {3}s  {4} MB/s" -f (Get-Date), (Split-Path $f.out -Leaf), $mb, [math]::Round($sw.Elapsed.TotalSeconds, 1), $rate)
}
Write-Output "ALL_DOWNLOADS_FINISHED"
Get-ChildItem $root -Recurse -File | Select-Object FullName, @{n='MB';e={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
