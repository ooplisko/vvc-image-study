param(
    [ValidateSet('full', 'smoke')][string]$Profile = 'smoke',
    [ValidateSet('source', 'bundle')][string]$CodecMode = 'source',
    [int]$Workers = 4
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv 0.11.19 first: https://docs.astral.sh/uv/getting-started/installation/'
}
$uvVersion = (& uv --version).Split(' ')[1]
if ($uvVersion -ne '0.11.19') { throw "Expected uv 0.11.19, found $uvVersion" }
Push-Location $PSScriptRoot
try {
    $env:PYTHONHASHSEED = '0'
    & uv sync --locked
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & uv run --locked vvc-study reproduce --profile $Profile --codec-mode $CodecMode --workers $Workers
    exit $LASTEXITCODE
}
finally { Pop-Location }
