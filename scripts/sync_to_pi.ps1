Write-Host "Syncing Unrealbot Vision & Autonomy folder to Raspberry Pi..." -ForegroundColor Cyan

# Determine source directory (directory of this script, but pointing to software folder)
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) {
    $ScriptDir = ".\scripts"
}
$SrcDir = "$ScriptDir\..\software"

# Target Raspberry Pi configuration
$PiUser = "pi"
$PiHost = "192.168.1.90" # Modify this if your Pi IP is different
$PiDest = "/home/pi/Documents/Unrealbot-ABIDE-ENTHRALLED/"

Write-Host "Source Directory: $SrcDir" -ForegroundColor Gray
Write-Host "Target Host: $PiUser@$PiHost:$PiDest" -ForegroundColor Gray

# Copy the core folders and files
scp -r "$SrcDir\vision" "${PiUser}@${PiHost}:${PiDest}"
scp -r "$SrcDir\tflite_models" "${PiUser}@${PiHost}:${PiDest}"
scp "$SrcDir\robot_main.py" "${PiUser}@${PiHost}:${PiDest}"
scp "$SrcDir\requirements.txt" "${PiUser}@${PiHost}:${PiDest}"

if ($LASTEXITCODE -eq 0) {
    Write-Host "Sync Complete!" -ForegroundColor Green
    Write-Host "To install dependencies on the Pi, run:" -ForegroundColor Yellow
    Write-Host "  pip install -r ~/Documents/Unrealbot-ABIDE-ENTHRALLED/requirements.txt" -ForegroundColor Yellow
} else {
    Write-Host "Sync Failed! Check connection to $PiHost." -ForegroundColor Red
}

Write-Host "Press any key to close..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
