$m = 'D:\Repos\CTPhysics_Main_Tools_Users_CH\Matlab\siemens.CTBasics'
$p = 'D:\Repos\CTPhysics_Main_Tools_Users_CH\Python\siemens.CTBasics\CTBasics'

$pyIdx = @{}
Get-ChildItem $p -Recurse -Filter *.py -File |
    Where-Object { $_.FullName -notmatch '__pycache__' } |
    ForEach-Object { $pyIdx[$_.BaseName] = (Get-Content $_.FullName | Measure-Object -Line).Lines }

Get-ChildItem $m -Recurse -Filter *.m -File | ForEach-Object {
    [pscustomobject]@{
        Ordner = $_.Directory.FullName.Replace($m, '').TrimStart('\')
        Datei  = $_.BaseName
        MATLAB = (Get-Content $_.FullName | Measure-Object -Line).Lines
        Python = $(if ($pyIdx.ContainsKey($_.BaseName)) { $pyIdx[$_.BaseName] } else { 0 })
    }
} | Sort-Object MATLAB -Descending | ForEach-Object {
    '{0,6} {1,6}  {2}\{3}' -f $_.MATLAB, $_.Python, $_.Ordner, $_.Datei
}
