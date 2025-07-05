# ----------- BUILD STAGE -------------
FROM node:18-alpine AS build

WORKDIR /app

# Copiar solo los archivos de dependencias
COPY package*.json ./

# Instalar TODAS las dependencias (incluidas las dev)
RUN npm ci

# Copiar el resto del código
COPY . .

# Ejecutar build
RUN npm run build

# ----------- PRODUCTION STAGE -------------
FROM nginx:alpine AS production

# Copiar archivos estáticos al servidor Nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Exponer el puerto 80
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
