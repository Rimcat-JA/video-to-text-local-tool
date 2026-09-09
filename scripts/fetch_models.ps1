<#
.SYNOPSIS
    段階 0 で必要なモデルを取得する（初回のみ・ネット接続が必要）。

.DESCRIPTION
    設計 13.1 により、通常の解析中にモデルの自動ダウンロードは行いません。
    このスクリプトは、利用者が明示的に実行したときだけ取得します。

    取得するもの:
      1) Qwen3-VL-8B-Instruct GGUF の Q4_K_M と mmproj  -> models/qwen3-vl-8b/
      2) Whisper large-v3-turbo の ggml 形式            -> models/whisper/

.PARAMETER Small
    8B の代わりに 4B（軽量構成の比較用）を取得します。

.PARAMETER SkipVision
.PARAMETER SkipAsr
    片方だけ取得したい場合に使います。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/fetch_models.ps1
#>
[CmdletBinding()]
param(
    [switch]$Small,
    [switch]$SkipVision,
    [switch]$SkipAsr
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Host '仮想環境が見つかりません。先に次を実行してください:' -ForegroundColor Yellow
    Write-Host '  py -3.12 -m venv .venv'
    Write-Host '  .venv\Scripts\python.exe -m pip install -e .'
    exit 1
}

Write-Host '=== huggingface_hub の CLI を確認します ===' -ForegroundColor Cyan
& $python -m pip install --quiet --upgrade "huggingface_hub[cli]"
if ($LASTEXITCODE -ne 0) { throw 'huggingface_hub のインストールに失敗しました。' }

# CLI の実行ファイル名は版によって hf.exe / huggingface-cli.exe のどちらかになる。
$scriptsDir = Join-Path $root '.venv\Scripts'
$hfCli = $null
foreach ($name in @('hf.exe', 'huggingface-cli.exe')) {
    $candidate = Join-Path $scriptsDir $name
    if (Test-Path $candidate) { $hfCli = $candidate; break }
}
if (-not $hfCli) {
    Write-Host 'huggingface の CLI が見つかりません。配布ページから手動で取得してください。' -ForegroundColor Red
    Write-Host '  https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF'
    Write-Host '  https://huggingface.co/ggerganov/whisper.cpp'
    exit 1
}
Write-Host "使用する CLI: $hfCli"

function Invoke-Download {
    param(
        [string]$Repo,
        [string[]]$Include,
        [string]$Dest
    )
    Write-Host ''
    Write-Host "--- $Repo -> $Dest" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $arguments = @('download', $Repo)
    foreach ($pattern in $Include) {
        $arguments += '--include'
        $arguments += $pattern
    }
    $arguments += @('--local-dir', $Dest)
    & $hfCli @arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Host "取得に失敗しました: $Repo" -ForegroundColor Red
        Write-Host 'ブラウザで配布ページを開き、必要なファイルを手動で置いてください。'
        Write-Host "  https://huggingface.co/$Repo"
        return $false
    }
    return $true
}

if (-not $SkipVision) {
    if ($Small) {
        $visionRepo = 'Qwen/Qwen3-VL-4B-Instruct-GGUF'
        $visionDest = Join-Path $root 'models\qwen3-vl-4b'
    } else {
        $visionRepo = 'Qwen/Qwen3-VL-8B-Instruct-GGUF'
        $visionDest = Join-Path $root 'models\qwen3-vl-8b'
    }
    Write-Host ''
    Write-Host '=== 画像モデル（VLM）===' -ForegroundColor Green
    Write-Host 'モデル本体（Q4_K_M）と mmproj の両方が必要です。'
    Invoke-Download -Repo $visionRepo -Include @('*Q4_K_M*', '*mmproj*') -Dest $visionDest | Out-Null
}

if (-not $SkipAsr) {
    $asrDest = Join-Path $root 'models\whisper'
    Write-Host ''
    Write-Host '=== 音声モデル（whisper.cpp 用 ggml 形式）===' -ForegroundColor Green
    Write-Host 'VLM 用の GGUF とは別形式です。'
    Invoke-Download -Repo 'ggerganov/whisper.cpp' -Include @('ggml-large-v3-turbo.bin') -Dest $asrDest | Out-Null
}

Write-Host ''
Write-Host '=== 取得結果 ===' -ForegroundColor Cyan
Get-ChildItem -Path (Join-Path $root 'models') -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Length -gt 1MB } |
    Select-Object @{N = 'ファイル'; E = { $_.FullName.Substring($root.Length + 1) } },
                  @{N = 'GiB'; E = { [math]::Round($_.Length / 1GB, 2) } } |
    Format-Table -AutoSize

Write-Host '次に不足の確認を行ってください:' -ForegroundColor Yellow
Write-Host '  .venv\Scripts\lecture-extract.exe doctor --vision-model <本体.gguf> --mmproj <mmproj.gguf> --asr-model models\whisper\ggml-large-v3-turbo.bin --asr-binary <whisper-cli のパス>'
