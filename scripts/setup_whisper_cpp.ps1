﻿<#
.SYNOPSIS
    whisper.cpp を用意する（初回のみ・ネット接続が必要）。

.DESCRIPTION
    whisper.cpp の公式リリースには、Windows 向けの Vulkan ビルド済みバイナリが
    ありません（CPU / BLAS / CUDA のみ）。AMD GPU で GPU 推論するには
    自分でビルドする必要があります。

    -Prebuilt : ビルド済みの CPU 版を取得する。コンパイラ不要。すぐ試せる。
    （既定）  : ソースを取得して Vulkan バックエンドでビルドする。
                git・cmake・C++ コンパイラ・Vulkan SDK が必要。

    設計 3.2 の通り、CPU のみの構成も維持しますが、実用速度は実測で判断します。
    まず -Prebuilt で通し、処理時間が足りなければビルドへ進むのが安全です。

.PARAMETER Prebuilt
    ビルド済みの CPU 版（whisper-bin-x64.zip）を取得して展開します。

.PARAMETER Dir
    取得先。既定は <プロジェクト>\third_party\whisper.cpp

.PARAMETER Cpu
    ソースからビルドする際に、Vulkan を使わず CPU のみでビルドします。

.PARAMETER ShowReleasesOnly
    何も取得せず、入手先と配布物の一覧だけを表示します。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_whisper_cpp.ps1 -Prebuilt
#>
[CmdletBinding()]
param(
    [switch]$Prebuilt,
    [string]$Dir,
    [switch]$Cpu,
    [switch]$ShowReleasesOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Dir) { $Dir = Join-Path $root 'third_party\whisper.cpp' }

if ($ShowReleasesOnly) {
    Write-Host '入手先:' -ForegroundColor Cyan
    Write-Host '  https://github.com/ggml-org/whisper.cpp/releases'
    Write-Host ''
    Write-Host 'Windows 向けの配布物:' -ForegroundColor Cyan
    Write-Host '  whisper-bin-x64.zip        CPU 版'
    Write-Host '  whisper-blas-bin-x64.zip   CPU + BLAS 版'
    Write-Host '  whisper-cublas-*-bin-x64.zip  NVIDIA CUDA 版（AMD GPU では使えません）'
    Write-Host ''
    Write-Host 'Vulkan のビルド済みバイナリは配布されていません。' -ForegroundColor Yellow
    Write-Host 'AMD GPU で GPU 推論する場合は、このスクリプトを -Prebuilt なしで実行してビルドしてください。'
    exit 0
}

# ---------------------------------------------------------------- prebuilt
if ($Prebuilt) {
    $dest = Join-Path $root 'third_party\whisper-bin-x64'
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $zip = Join-Path $dest 'whisper-bin-x64.zip'

    Write-Host '公開されているリリース情報を取得します...' -ForegroundColor Cyan
    $api = 'https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest'
    $release = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'lecture-extract-setup' }
    $asset = $release.assets | Where-Object { $_.name -eq 'whisper-bin-x64.zip' } | Select-Object -First 1
    if (-not $asset) {
        Write-Host 'whisper-bin-x64.zip が見つかりませんでした。' -ForegroundColor Red
        Write-Host "リリースページで配布物を確認してください: $($release.html_url)"
        exit 1
    }

    Write-Host ''
    Write-Host "  リリース: $($release.tag_name)"
    Write-Host "  ファイル: $($asset.name)  ($([math]::Round($asset.size / 1MB, 1)) MB)"
    Write-Host "  取得元  : $($asset.browser_download_url)"
    Write-Host "  展開先  : $dest"
    Write-Host ''

    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Remove-Item $zip -Force

    # main.exe は互換用の小さなスタブなので、whisper-cli.exe を優先する。
    $cli = $null
    foreach ($name in @('whisper-cli.exe', 'main.exe')) {
        $found = Get-ChildItem -Path $dest -Recurse -Filter $name -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($found) { $cli = $found; break }
    }
    if (-not $cli) {
        Write-Host '実行ファイルが見つかりません。展開先を確認してください。' -ForegroundColor Red
        Get-ChildItem -Path $dest -Recurse -Filter '*.exe' | Select-Object -ExpandProperty FullName
        exit 1
    }

    Write-Host '=== 完了（CPU 版）===' -ForegroundColor Green
    Write-Host "whisper-cli: $($cli.FullName)"
    Write-Host ''
    Write-Host 'これは CPU 推論です。GPU は使いません。' -ForegroundColor Yellow
    Write-Host '短い区間で処理時間を測り、実用速度に足りなければ -Prebuilt なしでビルドしてください。'
    Write-Host ''
    Write-Host '次の確認:'
    Write-Host "  .venv\Scripts\lecture-extract.exe doctor --asr-binary `"$($cli.FullName)`" --asr-model models\whisper\ggml-large-v3-turbo.bin --vision-model models\qwen3-vl-8b\Qwen3VL-8B-Instruct-Q4_K_M.gguf --mmproj models\qwen3-vl-8b\mmproj-Qwen3VL-8B-Instruct-F16.gguf"
    exit 0
}

# ------------------------------------------------------------ build source
$missing = @()
foreach ($tool in @('git', 'cmake')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { $missing += $tool }
}
if ($missing.Count -gt 0) {
    Write-Host "次のものが見つかりません: $($missing -join ', ')" -ForegroundColor Red
    Write-Host ''
    Write-Host 'ビルドせずにすぐ試す場合:' -ForegroundColor Yellow
    Write-Host '  powershell -ExecutionPolicy Bypass -File scripts\setup_whisper_cpp.ps1 -Prebuilt'
    Write-Host ''
    Write-Host 'ビルドする場合に必要なもの:' -ForegroundColor Yellow
    Write-Host '  winget install Git.Git'
    Write-Host '  winget install Kitware.CMake'
    Write-Host '  winget install Microsoft.VisualStudio.2022.BuildTools  （C++ ワークロードを選択）'
    Write-Host '  Vulkan SDK: https://vulkan.lunarg.com/sdk/home'
    exit 1
}

if (Test-Path (Join-Path $Dir '.git')) {
    Write-Host "既存の取得先を更新します: $Dir" -ForegroundColor Cyan
    git -C $Dir pull --ff-only
} else {
    Write-Host "whisper.cpp を取得します -> $Dir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Dir) | Out-Null
    git clone https://github.com/ggml-org/whisper.cpp $Dir
}

$commit = (git -C $Dir rev-parse --short HEAD).Trim()
Write-Host "取得したコミット: $commit" -ForegroundColor Green

$buildDir = Join-Path $Dir 'build'
if ($Cpu) {
    Write-Host 'CPU のみでビルドします。' -ForegroundColor Cyan
    cmake -B $buildDir -S $Dir
} else {
    Write-Host 'Vulkan バックエンドでビルドします。' -ForegroundColor Cyan
    cmake -B $buildDir -S $Dir -DGGML_VULKAN=ON
}
if ($LASTEXITCODE -ne 0) {
    Write-Host 'cmake の構成に失敗しました。Vulkan SDK が必要な場合があります。' -ForegroundColor Red
    Write-Host '  https://vulkan.lunarg.com/sdk/home'
    Write-Host 'CPU のみで試す場合は -Cpu を、ビルド自体を避ける場合は -Prebuilt を付けて再実行してください。'
    exit 1
}

cmake --build $buildDir -j --config Release
if ($LASTEXITCODE -ne 0) { throw 'ビルドに失敗しました。' }

$cli = Get-ChildItem -Path $buildDir -Recurse -Filter 'whisper-cli.exe' -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $cli) {
    Write-Host 'whisper-cli.exe が見つかりません。ビルド出力を確認してください。' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host '=== 完了 ===' -ForegroundColor Green
Write-Host "whisper-cli: $($cli.FullName)"
Write-Host ''
Write-Host '短い音声で必ず確認してください:' -ForegroundColor Yellow
Write-Host "  $($cli.FullName) -m models\whisper\ggml-large-v3-turbo.bin -f test.wav -l ja --output-json -of tmp\test_asr"
Write-Host ''
Write-Host "この場所を --asr-binary に渡します（成功したビルドのコミット: $commit）"
