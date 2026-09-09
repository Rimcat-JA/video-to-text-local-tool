<#
.SYNOPSIS
    whisper.cpp を取得して Vulkan バックエンドでビルドする（初回のみ・ネット接続が必要）。

.DESCRIPTION
    設計 3.2 により、AMD GPU では Vulkan 対応ビルドを第一候補にします。
    ビルドせずにビルド済みバイナリを使う場合は -ShowReleasesOnly を付けてください。

    「Vulkan 対応」という情報だけでは、この GPU・ドライバ・モデルの組み合わせで
    動作することを断定できません。ビルド後、短い音声で必ず確認してください。

.PARAMETER Dir
    取得先。既定は <プロジェクト>\third_party\whisper.cpp

.PARAMETER Cpu
    Vulkan を使わず CPU のみでビルドします。

.PARAMETER ShowReleasesOnly
    ビルドせず、ビルド済みバイナリの入手先だけを表示します。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/setup_whisper_cpp.ps1
#>
[CmdletBinding()]
param(
    [string]$Dir,
    [switch]$Cpu,
    [switch]$ShowReleasesOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Dir) { $Dir = Join-Path $root 'third_party\whisper.cpp' }

if ($ShowReleasesOnly) {
    Write-Host 'ビルド済みバイナリの入手先:' -ForegroundColor Cyan
    Write-Host '  https://github.com/ggml-org/whisper.cpp/releases'
    Write-Host ''
    Write-Host 'Windows x64 向けのアーカイブ（Vulkan 版があればそれ）を展開し、'
    Write-Host 'whisper-cli.exe の場所を --asr-binary に渡してください。'
    Write-Host 'リリースごとに資産名が異なるため、ページで実際の名前を確認してください。'
    exit 0
}

foreach ($tool in @('git', 'cmake')) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "$tool が見つかりません。" -ForegroundColor Red
        Write-Host 'ビルドせずに済ませる場合は -ShowReleasesOnly を付けて実行してください。'
        exit 1
    }
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
    Write-Host 'CPU のみで試す場合は -Cpu を付けて再実行してください。'
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
Write-Host '次に、モデルを取得して短い音声で確認してください:' -ForegroundColor Yellow
Write-Host '  powershell -ExecutionPolicy Bypass -File scripts/fetch_models.ps1 -SkipVision'
Write-Host "  $($cli.FullName) -m models\whisper\ggml-large-v3-turbo.bin -f test.wav -l ja --output-json -of tmp\test_asr"
Write-Host ''
Write-Host "この場所を --asr-binary に渡します（成功したビルドは manifest に記録されます。コミット: $commit）"
