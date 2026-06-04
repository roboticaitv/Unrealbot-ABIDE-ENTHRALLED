Write-Host "Syncing Unrealbot Vision folder to Raspberry Pi..." -ForegroundColor Cyan

# Copies the vision folder, ONNX models, and robot_main to the Pi. 
# It assumes your Pi folder is located at ~/Documents/Unrealbot-ABIDE-ENTHRALLED/
scp -r "C:\Users\caser\OneDrive\Documents\GitHub\Unrealbot-ABIDE-ENTHRALLED\vision" pi@192.168.1.90:/home/pi/Documents/Unrealbot-ABIDE-ENTHRALLED/
scp -r "C:\Users\caser\OneDrive\Documents\GitHub\Unrealbot-ABIDE-ENTHRALLED\ONNX_models" pi@192.168.1.90:/home/pi/Documents/Unrealbot-ABIDE-ENTHRALLED/
scp "C:\Users\caser\OneDrive\Documents\GitHub\Unrealbot-ABIDE-ENTHRALLED\robot_main.py" pi@192.168.1.90:/home/pi/Documents/Unrealbot-ABIDE-ENTHRALLED/

if ($LASTEXITCODE -eq 0) {
    Write-Host "Sync Complete!" -ForegroundColor Green
} else {
    Write-Host "Sync Failed!" -ForegroundColor Red
}

Write-Host "Press any key to close..."
$Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") | Out-Null
