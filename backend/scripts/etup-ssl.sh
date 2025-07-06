#!/bin/bash

# Script para configurar SSL en el backend
echo "Configurando SSL para el backend..."

# Crear directorios necesarios
mkdir -p /etc/ssl/certs /etc/ssl/private

# Generar certificado autofirmado
if [ ! -f "/etc/ssl/certs/server.crt" ] || [ ! -f "/etc/ssl/private/server.key" ]; then
    echo "Generando certificado SSL autofirmado..."
    # Obtener IP externa de la VM
    EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
    INTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip)
    
    # Crear archivo de configuración para el certificado
    cat > /tmp/ssl.conf <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = US
ST = CA
L = Mountain View
O = SafeCode
OU = Backend
CN = ${EXTERNAL_IP}

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
IP.1 = ${EXTERNAL_IP}
IP.2 = ${INTERNAL_IP}
IP.3 = 127.0.0.1
DNS.1 = localhost
EOF

    # Generar clave privada y certificado
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /etc/ssl/private/server.key \
        -out /etc/ssl/certs/server.crt \
        -config /tmp/ssl.conf \
        -extensions v3_req

    # Establecer permisos correctos
    chmod 600 /etc/ssl/private/server.key
    chmod 644 /etc/ssl/certs/server.crt
    
    echo "Certificado SSL generado exitosamente"
    echo "IP Externa: ${EXTERNAL_IP}"
    echo "IP Interna: ${INTERNAL_IP}"
else
    echo "Certificados SSL ya existen"
fi

# Mostrar información del certificado
echo "Información del certificado:"
openssl x509 -in /etc/ssl/certs/server.crt -text -noout | grep -E "(Subject:|DNS:|IP Address:)"
