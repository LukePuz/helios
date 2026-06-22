# Creates a "Trading Bot" shortcut on your Desktop that double-clicks to launch everything.
# Run once: right-click this file -> "Run with PowerShell"

$botDir    = Split-Path -Parent $MyInvocation.MyCommand.Path
$batFile   = Join-Path $botDir "launch.bat"
$desktop   = [Environment]::GetFolderPath("Desktop")
$shortcut  = Join-Path $desktop "Trading Bot.lnk"

$wsh  = New-Object -ComObject WScript.Shell
$lnk  = $wsh.CreateShortcut($shortcut)
$lnk.TargetPath       = $batFile
$lnk.WorkingDirectory = $botDir
$lnk.WindowStyle      = 1   # Normal window
$lnk.Description      = "Launch Trading Bot + Dashboard"
$lnk.Save()

Write-Host ""
Write-Host "  Shortcut created: $shortcut"
Write-Host "  Double-click 'Trading Bot' on your Desktop to start everything."
Write-Host ""
