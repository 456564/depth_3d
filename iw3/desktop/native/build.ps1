$ErrorActionPreference = 'Stop'
$compiler = (Get-Command cl -ErrorAction Stop).Source
$vc = Split-Path (Split-Path (Split-Path (Split-Path $compiler)))
$kits = 'C:\Program Files (x86)\Windows Kits\10'
$sdkVersion = (Get-ChildItem -LiteralPath (Join-Path $kits 'Include') -Directory | Sort-Object Name -Descending | Select-Object -First 1).Name
$sdkInclude = Join-Path $kits ('Include\' + $sdkVersion)
$sdkLib = Join-Path $kits ('Lib\' + $sdkVersion)
$buildArgs = @('/nologo', '/LD', '/MT', '/EHsc', '/std:c++17', '/O2', '/W4', '/utf-8',
    ('/I' + (Join-Path $vc 'include')), ('/I' + (Join-Path $sdkInclude 'ucrt')),
    ('/I' + (Join-Path $sdkInclude 'shared')), ('/I' + (Join-Path $sdkInclude 'um')),
    ('/I' + (Join-Path $sdkInclude 'winrt')),
    (Join-Path $PSScriptRoot 'process_audio.cpp'), ('/Fo' + (Join-Path $PSScriptRoot 'quest_process_audio.obj')),
    '/link', ('/OUT:' + (Join-Path $PSScriptRoot 'quest_process_audio.dll')),
    ('/IMPLIB:' + (Join-Path $PSScriptRoot 'quest_process_audio.lib')),
    ('/LIBPATH:' + (Join-Path $vc 'lib\x64')), ('/LIBPATH:' + (Join-Path $sdkLib 'ucrt\x64')),
    ('/LIBPATH:' + (Join-Path $sdkLib 'um\x64')), 'ole32.lib', 'mmdevapi.lib', 'uuid.lib')
& $compiler @buildArgs
if ($LASTEXITCODE -ne 0) { throw ('Native build failed: ' + $LASTEXITCODE) }
