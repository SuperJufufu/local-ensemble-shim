@echo off
rem Local Heavy server (Bonsai x3 lenses + minicpm5) -- OpenAI-compatible shim
rem port 8020, model id "dual-heavy", used by DSH provider "local-heavy"
rem NOTE: keep this file ASCII-only; cmd parses .bat with the OEM code page,
rem       so non-ASCII text here can corrupt the command lines.
cd /d "%~dp0"
start "" /min python heavy_server.py --port 8020
echo Started Local Heavy server on http://127.0.0.1:8020/v1
echo Log: %~dp0heavy-server-trace.log
echo Stop it with stop-heavy-server.bat
