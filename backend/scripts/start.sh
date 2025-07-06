#!/bin/bash

echo "Iniciando SafeCode Backend..."

# Configurar SSL si está habilitado
if [ "$ENABLE_SSL" = "true" ]; then
    echo "SSL habilitado, configurando certificados..."
    /usr/local/bin/setup-ssl.sh
fi

# Obtener IP externa para logs
EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip 2>/dev/null || echo "unknown")

echo "Servidor disponible en:"
echo "   HTTP:  http://${EXTERNAL_IP}:8000"
if [ "$ENABLE_SSL" = "true" ]; then
    echo "   HTTPS: https://${EXTERNAL_IP}:8443"
fi

# Iniciar la aplicación
if [ "$ENABLE_SSL" = "true" ] && [ -f "/etc/ssl/certs/server.crt" ]; then
    echo "Iniciando con SSL..."
    python main.py
else
    echo "Iniciando sin SSL..."
    uvicorn main:app --host 0.0.0.0 --port 8000
fi
