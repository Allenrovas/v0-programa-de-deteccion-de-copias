# Punto de entrada de la API
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload

app = FastAPI(
    title="Detector de Copias",
    description="API para detectar similitudes entre fragmentos de código",
    version="0.1.0"
)

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(upload.router)

@app.get("/")
async def root():
    return {"message": "Bienvenido al API de Detección de Copias"}