#!/bin/bash
echo "Configurando el arranque automático de Unrealbot..."

# Crear el archivo del servicio de systemd
cat << 'EOF' | sudo tee /etc/systemd/system/unrealbot.service > /dev/null
[Unit]
Description=Unrealbot ABIDE Autonomous System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/Desktop/Unrealbot-ABIDE-ENTHRALLED
Environment="PATH=/home/pi/Desktop/abide_env_313/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
# Arranca el sistema principal (robot_main.py) usando el entorno virtual con TFLite
ExecStart=/home/pi/Desktop/abide_env_313/bin/python /home/pi/Desktop/Unrealbot-ABIDE-ENTHRALLED/software/robot_main.py
Restart=on-failure
RestartSec=5
StandardOutput=syslog
StandardError=syslog
SyslogIdentifier=unrealbot

[Install]
WantedBy=multi-user.target
EOF

# Recargar systemd y habilitar el servicio
sudo systemctl daemon-reload
sudo systemctl enable unrealbot.service

echo "--------------------------------------------------------"
echo "¡Éxito! El sistema Unrealbot arrancará solo cada que enciendas la Raspberry Pi 5."
echo ""
echo "Comandos útiles para ti:"
echo "Ver si está corriendo:  sudo systemctl status unrealbot"
echo "Ver los logs en vivo:   journalctl -u unrealbot -f"
echo "Apagarlo manualmente:   sudo systemctl stop unrealbot"
echo "Encenderlo manualmente: sudo systemctl start unrealbot"
echo "--------------------------------------------------------"
