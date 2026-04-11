param(
    [string]$ProjectRoot = '',
    [string]$PlanCsv = '',
    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

function Get-ChunkHash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][ValidateSet('first', 'last')][string]$Mode,
        [int]$ChunkSize = 4194304
    )

    $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
    try {
        $length = [long]$fs.Length
        if ($length -le 0) {
            return ''
        }

        $bytesToRead = if ($length -lt $ChunkSize) { [int]$length } else { [int]$ChunkSize }
        $buffer = New-Object byte[] $bytesToRead
        if ($Mode -eq 'last') {
            $fs.Seek(-1 * [long]$bytesToRead, [System.IO.SeekOrigin]::End) | Out-Null
        }

        $read = $fs.Read($buffer, 0, $bytesToRead)
        if ($read -lt $bytesToRead) {
            if ($read -le 0) {
                return ''
            }
            $slice = New-Object byte[] $read
            [Array]::Copy($buffer, $slice, $read)
            $buffer = $slice
        }

        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            return ([System.BitConverter]::ToString($sha.ComputeHash($buffer))).Replace('-', '')
        }
        finally {
            $sha.Dispose()
        }
    }
    finally {
        $fs.Dispose()
    }
}

function Test-PracticalDuplicate {
    param(
        [Parameter(Mandatory = $true)][string]$PathA,
        [Parameter(Mandatory = $true)][string]$PathB
    )

    if (-not (Test-Path -LiteralPath $PathA) -or -not (Test-Path -LiteralPath $PathB)) {
        return $false
    }

    $a = Get-Item -LiteralPath $PathA
    $b = Get-Item -LiteralPath $PathB
    if ($a.Length -ne $b.Length) {
        return $false
    }

    return (
        (Get-ChunkHash -Path $PathA -Mode 'first') -eq (Get-ChunkHash -Path $PathB -Mode 'first') -and
        (Get-ChunkHash -Path $PathA -Mode 'last') -eq (Get-ChunkHash -Path $PathB -Mode 'last')
    )
}

function Get-UniqueDestinationPath {
    param(
        [Parameter(Mandatory = $true)][string]$DestinationDir,
        [Parameter(Mandatory = $true)][string]$FileName
    )

    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    $extension = [System.IO.Path]::GetExtension($FileName)
    $candidate = Join-Path $DestinationDir $FileName
    $index = 1
    while (Test-Path -LiteralPath $candidate) {
        $candidate = Join-Path $DestinationDir ("{0}__dup_{1}{2}" -f $baseName, $index, $extension)
        $index += 1
    }
    return $candidate
}

$resolvedProjectRoot = if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
}
else {
    (Resolve-Path $ProjectRoot).Path
}

$inputRoot = Join-Path $resolvedProjectRoot 'Input_Videos'
$excludeRoot = Join-Path $inputRoot 'exclude'
if ([string]::IsNullOrWhiteSpace($PlanCsv)) {
    $PlanCsv = Join-Path $resolvedProjectRoot 'Output_Results\practical_duplicate_move_plan.csv'
}

if (-not (Test-Path -LiteralPath $PlanCsv)) {
    throw "Plan CSV not found: $PlanCsv"
}

if (-not (Test-Path -LiteralPath $excludeRoot)) {
    if ($WhatIf) {
        Write-Host "[WhatIf] Would create $excludeRoot"
    }
    else {
        New-Item -ItemType Directory -Path $excludeRoot | Out-Null
    }
}

$plan = Import-Csv $PlanCsv
$total = $plan.Count
$moved = 0
$deleted = 0
$skipped = 0
$renamed = 0
$processed = 0

Write-Host "Plan: $PlanCsv"
Write-Host "Candidates: $total"
Write-Host "Target exclude folder: $excludeRoot"

foreach ($row in $plan) {
    $processed += 1
    $remaining = $total - $processed
    $sourcePath = Join-Path $inputRoot $row.MoveToExcludePath
    $fileName = [System.IO.Path]::GetFileName($row.MoveToExcludePath)
    $defaultTarget = Join-Path $excludeRoot $fileName

    $status = "{0}/{1} remaining={2}" -f $processed, $total, $remaining
    Write-Progress -Activity 'Moving practical duplicate candidates to Input_Videos\exclude' -Status $status -PercentComplete (($processed / [double]$total) * 100)

    if (-not (Test-Path -LiteralPath $sourcePath)) {
        $skipped += 1
        Write-Host "[SKIP] missing: $($row.MoveToExcludePath)"
        continue
    }

    if (-not (Test-Path -LiteralPath $defaultTarget)) {
        if ($WhatIf) {
            Write-Host "[WhatIf][MOVE] $($row.MoveToExcludePath) -> exclude\\$fileName"
        }
        else {
            Move-Item -LiteralPath $sourcePath -Destination $defaultTarget
        }
        $moved += 1
        continue
    }

    if (Test-PracticalDuplicate -PathA $sourcePath -PathB $defaultTarget) {
        if ($WhatIf) {
            Write-Host "[WhatIf][DELETE] $($row.MoveToExcludePath) (duplicate of exclude\\$fileName)"
        }
        else {
            Remove-Item -LiteralPath $sourcePath -Force
        }
        $deleted += 1
        continue
    }

    $uniqueTarget = Get-UniqueDestinationPath -DestinationDir $excludeRoot -FileName $fileName
    $renamed += 1
    if ($WhatIf) {
        Write-Host "[WhatIf][MOVE-RENAME] $($row.MoveToExcludePath) -> $(Split-Path -Leaf $uniqueTarget)"
    }
    else {
        Move-Item -LiteralPath $sourcePath -Destination $uniqueTarget
    }
    $moved += 1
}

Write-Progress -Activity 'Moving practical duplicate candidates to Input_Videos\exclude' -Completed
Write-Host ''
Write-Host 'Summary'
Write-Host "Processed: $processed / $total"
Write-Host "Moved: $moved"
Write-Host "Deleted due to duplicate name already in exclude: $deleted"
Write-Host "Renamed on move due to conflicting non-duplicate filename: $renamed"
Write-Host "Skipped: $skipped"
