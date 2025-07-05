from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime

router = APIRouter(prefix="/api/feedback", tags=["feedback"])

@router.post("/send")
async def send_feedback(
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
):
    """
    Envía feedback por correo electrónico usando Gmail SMTP con imágenes adjuntas
    """
    try:
        # Configuración de Gmail SMTP
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = os.getenv("GMAIL_USER")
        sender_password = os.getenv("GMAIL_APP_PASSWORD")
        recipient_email = os.getenv("RECIPIENT_EMAIL", sender_email)
        
        if not sender_email or not sender_password:
            raise HTTPException(
                status_code=500, 
                detail="Configuración de email no encontrada. Verifica las variables de entorno GMAIL_USER y GMAIL_APP_PASSWORD."
            )
        
        # Crear el mensaje
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"Nuevo Feedback de SafeCode - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Cuerpo del email
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        body = f"""
            Nuevo feedback recibido en SafeCode:

            Fecha: {timestamp}

            Mensaje:
            {message}

            ---
            Este mensaje fue enviado desde el sistema SafeCode de detección de copias.
            Proyecto de graduación - USAC FIUSAC ECYS
        """
        
        # Adjuntar el cuerpo del mensaje
        msg.attach(MIMEText(body, "plain", "utf-8"))
        
        # Procesar imágenes adjuntas si existen
        if images:
            for image in images:
                if image.filename:
                    # Leer el contenido del archivo
                    content = await image.read()
                    
                    # Crear adjunto
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(content)
                    encoders.encode_base64(part)
                    
                    # Agregar header
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {image.filename}'
                    )
                    
                    # Adjuntar al mensaje
                    msg.attach(part)
        
        # Enviar el email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        return {
            "success": True,
            "message": "Feedback enviado correctamente",
            "details": {
                "timestamp": timestamp,
                "images_count": len(images) if images else 0
            }
        }
        
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(
            status_code=500,
            detail="Error de autenticación con Gmail. Verifica tu App Password de Gmail."
        )
    except smtplib.SMTPException as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error enviando email: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )

@router.get("/test")
async def test_email_config():
    """
    Endpoint para probar la configuración de email
    """
    sender_email = os.getenv("GMAIL_USER")
    sender_password = os.getenv("GMAIL_APP_PASSWORD")
    recipient_email = os.getenv("RECIPIENT_EMAIL")
    
    return {
        "gmail_user_configured": bool(sender_email),
        "gmail_password_configured": bool(sender_password),
        "recipient_configured": bool(recipient_email),
        "sender_email": sender_email if sender_email else "No configurado",
        "recipient_email": recipient_email if recipient_email else "Mismo que sender"
    }
