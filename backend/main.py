from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, ground_truth, adaptive_learning, feedback
from dotenv import load_dotenv
import uvicorn
import ssl
import os

load_dotenv()

app = FastAPI(
    title="Detector de Copias",
    description="API para detectar similitudes entre fragmentos de código",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(upload.router)
app.include_router(ground_truth.router)
app.include_router(adaptive_learning.router)
app.include_router(feedback.router)

@app.get("/")
async def root():
    return {"message": "Bienvenido al API de Detección de Copias", "ssl_enabled": True}

# Configuración SSL global
ssl_context = None
cert_file = "/etc/ssl/certs/server.crt"
key_file = "/etc/ssl/private/server.key"

if os.path.exists(cert_file) and os.path.exists(key_file):
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_context.load_cert_chain(cert_file, key_file)
    port = 8443
else:
    port = 8000

@app.get("/health")
async def health_check():
    return {"status": "healthy", "ssl": True if ssl_context else False}

if __name__ == "__main__":
    if ssl_context:
        print("SSL habilitado")
    else:
        print("Ejecutando sin SSL en puerto 8000")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,  # Deshabilitado para producción
        ssl_keyfile=key_file if ssl_context else None,
        ssl_certfile=cert_file if ssl_context else None
    )
