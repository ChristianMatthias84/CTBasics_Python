param(
    [string]$MatlabRoot = 'd:\Repos\CTPhysics_Main_Tools_Users_CH\Matlab\siemens.CTBasics',
    [string]$PythonRoot = 'd:\Repos\CTPhysics_Main_Tools_Users_CH\Python\siemens.CTBasics'
)

$ErrorActionPreference = 'Stop'
$docsRoot = Join-Path $PythonRoot 'docs\migration'

function Get-ModuleTarget {
    param([string]$RelativePath)

    $normalized = $RelativePath -replace '\\', '/'
    if ($normalized -like 'Descriptors/*') { return 'ctbasics.descriptors' }
    if ($normalized -like 'Simulation/*') { return 'ctbasics.simulation' }
    if ($normalized -like 'Reconstruction/*') { return 'ctbasics.reconstruction' }
    if ($normalized -like 'Preprocessing/*') { return 'ctbasics.preprocessing' }
    if ($normalized -like 'ReadAndWrite/*') { return 'ctbasics.io' }
    if ($normalized -like 'RawData/*') { return 'ctbasics.rawdata' }
    if ($normalized -like 'MultiCore/*') { return 'ctbasics.parallel' }
    if ($normalized -like 'Memory/*') { return 'ctbasics.memory' }
    if ($normalized -like 'ImageManipulation/*') { return 'ctbasics.image_manipulation' }
    if ($normalized -like 'Visualization/*') { return 'ctbasics.visualization' }
    if ($normalized -like 'Apps/*') { return 'ctbasics.apps' }
    if ($normalized -like 'LecturesAndTestScripts/*') { return 'examples.lectures' }
    if ($normalized -like 'Benchmarks/*') { return 'benchmarks' }
    if ($normalized -like 'Projects/Static CT/*') { return 'projects.static_ct' }
    if ($normalized -like 'Scripts/*') { return 'examples.scripts' }
    if ($normalized -eq 'init_CTBasics.m') { return 'ctbasics.bootstrap' }
    if ($normalized -eq 'RunAllScripts.m') { return 'examples.run_all' }
    return 'ctbasics.misc'
}

function Get-SymbolInfo {
    param([string]$FilePath)

    $head = Get-Content -LiteralPath $FilePath -TotalCount 30 -ErrorAction SilentlyContinue | Out-String
    if ($head -match '^(?s).*^\s*classdef\s+(?:\([^\)]*\)\s*)?(?<name>[A-Za-z]\w*)') {
        return @($Matches.name, 'class')
    }
    if ($head -match '^(?s).*^\s*function\s+(?:\[[^\]]*\]\s*=\s*)?(?<name>[A-Za-z]\w*)') {
        return @($Matches.name, 'function')
    }
    return @([System.IO.Path]::GetFileNameWithoutExtension($FilePath), 'script')
}

$manifestRows = Get-ChildItem -LiteralPath $MatlabRoot -Recurse -File |
    Where-Object { $_.FullName -notmatch '\\.git(\\|$)' } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($MatlabRoot.Length + 1) -replace '\\', '/'
        [pscustomobject]@{
            relative_path = $relative
            sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant()
            size_bytes = $_.Length
        }
    }

$inventoryRows = Get-ChildItem -LiteralPath $MatlabRoot -Recurse -File -Filter *.m |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($MatlabRoot.Length + 1) -replace '\\', '/'
        $symbolInfo = Get-SymbolInfo -FilePath $_.FullName
        $deps = ''
        $status = 'not-started'
        if ($relative -eq 'RawData/ReadRawDataFunctions.m') {
            $deps = 'readXACB;readTable;xml2struct;TFS toolbox'
            $status = 'blocked'
        }
        if ($relative -eq 'init_CTBasics.m') {
            $deps = 'userpath CTBasicsConfig.txt;TFS toolbox'
        }
        if ($relative -eq 'RunAllScripts.m') {
            $deps = 'IOHelperFunctions;all lecture/project scripts'
        }

        [pscustomobject]@{
            matlab_file = $relative
            symbol_method_or_scenario = $symbolInfo[0]
            variants = $symbolInfo[1]
            python_target = Get-ModuleTarget $relative
            dependencies = $deps
            test_id = 'M0-SEED'
            status = $status
        }
    }

$inventoryRows += [pscustomobject]@{
    matlab_file = 'external/readXACB'
    symbol_method_or_scenario = 'readXACB'
    variants = 'external-function'
    python_target = 'ctbasics.rawdata'
    dependencies = 'external TFS/raw-data toolbox; binary interface unknown'
    test_id = 'M0-SEED'
    status = 'blocked'
}
$inventoryRows += [pscustomobject]@{
    matlab_file = 'external/readTable'
    symbol_method_or_scenario = 'readTable'
    variants = 'external-function'
    python_target = 'ctbasics.rawdata'
    dependencies = 'external TFS/raw-data toolbox; binary interface unknown'
    test_id = 'M0-SEED'
    status = 'blocked'
}
$inventoryRows += [pscustomobject]@{
    matlab_file = 'external/xml2struct'
    symbol_method_or_scenario = 'xml2struct'
    variants = 'external-function'
    python_target = 'ctbasics.rawdata'
    dependencies = 'external MATLAB toolbox; binary interface unknown'
    test_id = 'M0-SEED'
    status = 'blocked'
}

$manifestPath = Join-Path $docsRoot 'original_manifest.csv'
$inventoryPath = Join-Path $docsRoot 'inventory.csv'
$manifestRows | ConvertTo-Csv -NoTypeInformation | Set-Content -Encoding UTF8 -LiteralPath $manifestPath
$inventoryRows | ConvertTo-Csv -NoTypeInformation | Set-Content -Encoding UTF8 -LiteralPath $inventoryPath

Write-Host ('manifest_rows=' + $manifestRows.Count)
Write-Host ('inventory_rows=' + $inventoryRows.Count)
