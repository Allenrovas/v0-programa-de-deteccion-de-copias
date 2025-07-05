# Usar Node.js como imagen base
FROM node:18-alpine as build

# Establecer directorio de trabajo
WORKDIR /app

# Copiar package.json y package-lock.json
COPY package*.json ./

# Instalar dependencias
RUN npm ci --only=production

# Copiar el código fuente
COPY . .

# Construir la aplicación
RUN npm run build

# Etapa de producción - usar Node.js para servir archivos estáticos
FROM node:18-alpine as production

WORKDIR /app

# Instalar serve globalmente para servir archivos estáticos
RUN npm install -g serve

# Copiar los archivos construidos desde la etapa de build
COPY --from=build /app/dist ./dist

# Exponer el puerto (Cloud Run usa la variable PORT)
EXPOSE 8080

# Comando para iniciar la aplicación
CMD ["serve", "-s", "dist", "-l", "8080"]
