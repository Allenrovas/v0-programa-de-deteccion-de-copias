#!/bin/bash

# Script para configurar SSL en la VM de GCP
echo "🔧 Configurando SSL en VM de GCP..."

# Actualizar sistema
sudo apt-get update

# Instalar dependencias
sudo apt-get install -y openssl curl

# Configurar firewall para HTTPS
echo "Configurando reglas de firewall..."
gcloud compute firewall-rules create allow-https-8443 \
    --allow tcp:8443 \
    --source-ranges 0.0.0.0/0 \
    --description "Allow HTTPS traffic on port 8443" \
    --quiet 2>/dev/null || echo "Regla de firewall ya existe"

# Obtener información de la VM
EXTERNAL_IP=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip)
ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)
INSTANCE_NAME=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)

echo "Información de la VM:"
echo "IP Externa: ${EXTERNAL_IP}"
echo "Zona: ${ZONE}"
echo "Instancia: ${INSTANCE_NAME}"

# Crear directorio para el proyecto si no existe
mkdir -p ~/safecode-backend
cd ~/safecode-backend

# Descargar docker-compose si no existe
if [ ! -f "docker-compose.yml" ]; then
    echo "Descargando configuración..."
    # Aquí deberías clonar tu repositorio o copiar los archivos
    echo "Asegúrate de tener los archivos del proyecto en ~/safecode-backend"
fi

echo "Configuración completada"
echo ""
echo "URLs de acceso:"
echo "HTTP:  http://${EXTERNAL_IP}:8000"
echo "HTTPS: https://${EXTERNAL_IP}:8443"
echo ""
echo "Próximos pasos:"
echo "1. Copia los archivos del proyecto a ~/safecode-backend"
echo "2. Ejecuta: docker-compose up -d"
echo "3. Actualiza la URL del frontend para usar HTTPS"
