@echo off
setlocal

call "%~dp0setenv.bat"
set "IW3_4K60_EXPERIMENT=1"
if not defined IW3_TRT_SITE set "IW3_TRT_SITE=%~dp0..\.Codex\research\chain-timing\depth-trt\verified-site"
if not defined IW3_TRT_PLAN set "IW3_TRT_PLAN=%~dp0..\.Codex\research\chain-timing\depth-trt\artifacts\amp-v2-s-392x686-op20.plan"
if not defined IW3_TRT_CACHE set "IW3_TRT_CACHE=%~dp0..\.Codex\research\chain-timing\depth-trt\artifacts\rtx-5f80873d1839a3f2.bin"
pushd "%NUNIF_DIR%" || exit /b 1
start "iw3 4K60 experiment" pythonw -m iw3.desktop.gui
popd
exit /b 0
