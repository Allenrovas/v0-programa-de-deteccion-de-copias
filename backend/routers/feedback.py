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

def create_html_email_template(message: str, timestamp: str, images_count: int = 0) -> str:
    """
    Crea un template HTML con estética dark mode similar al frontend
    """
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Nuevo Feedback - SafeCode</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                background-color: #0f172a;
                color: #e2e8f0;
                line-height: 1.6;
                padding: 20px;
            }}
            
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                border: 1px solid #334155;
            }}
            
            .header {{
                background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
                padding: 30px;
                text-align: center;
                position: relative;
                overflow: hidden;
            }}
            
            .header::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
                opacity: 0.3;
            }}
            
            .header h1 {{
                color: white;
                font-size: 28px;
                font-weight: 700;
                margin-bottom: 8px;
                position: relative;
                z-index: 1;
            }}
            
            .header p {{
                color: rgba(255, 255, 255, 0.9);
                font-size: 16px;
                position: relative;
                z-index: 1;
            }}
            
            .content {{
                padding: 40px 30px;
            }}
            
            .info-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .info-item {{
                background: #1e293b;
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #334155;
                text-align: center;
            }}
            
            .info-item .label {{
                color: #94a3b8;
                font-size: 14px;
                font-weight: 500;
                margin-bottom: 8px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            
            .info-item .value {{
                color: #e2e8f0;
                font-size: 18px;
                font-weight: 600;
            }}
            
            .message-section {{
                background: #1e293b;
                border-radius: 12px;
                padding: 25px;
                border: 1px solid #334155;
                margin-bottom: 30px;
            }}
            
            .message-section h3 {{
                color: #3b82f6;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 15px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .message-section h3::before {{
                content: '💬';
                font-size: 20px;
            }}
            
            .message-text {{
                background: #0f172a;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #3b82f6;
                color: #e2e8f0;
                font-size: 16px;
                white-space: pre-wrap;
                word-wrap: break-word;
            }}
            
            .attachments {{
                background: #1e293b;
                border-radius: 12px;
                padding: 20px;
                border: 1px solid #334155;
                margin-bottom: 30px;
            }}
            
            .attachments h3 {{
                color: #10b981;
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .attachments h3::before {{
                content: '📎';
                font-size: 18px;
            }}
            
            .attachment-count {{
                background: #10b981;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 14px;
                font-weight: 600;
            }}
            
            .footer {{
                background: #0f172a;
                padding: 25px 30px;
                text-align: center;
                border-top: 1px solid #334155;
            }}
            
            .footer p {{
                color: #64748b;
                font-size: 14px;
                margin-bottom: 8px;
            }}
            
            .footer .project-info {{
                color: #3b82f6;
                font-weight: 600;
                font-size: 16px;
            }}
            
            .badge {{
                display: inline-block;
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                color: white;
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-top: 10px;
            }}
            
            @media (max-width: 600px) {{
                .info-grid {{
                    grid-template-columns: 1fr;
                }}
                
                .container {{
                    margin: 10px;
                }}
                
                .header, .content, .footer {{
                    padding: 20px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>SafeCode</h1>
                <p>Sistema de Detección de Copias</p>
            </div>
            
            <div class="content">
                <div class="info-grid">
                    <div class="info-item">
                        <div class="label">Fecha y Hora</div>
                        <div class="value">{timestamp}</div>
                    </div>
                </div>
                
                <div class="message-section">
                    <h3>Mensaje de Feedback</h3>
                    <div class="message-text">{message}</div>
                </div>
                
                {f'''
                <div class="attachments">
                    <h3>Archivos Adjuntos</h3>
                    <span class="attachment-count">{images_count} archivo(s) adjunto(s)</span>
                </div>
                ''' if images_count > 0 else ''}
            </div>
            
            <div class="footer">
                <p>Este mensaje fue enviado desde el sistema SafeCode</p>
                <div class="project-info">Proyecto de Graduación - USAC FIUSAC ECYS</div>
                <div class="badge">Feedback Automático</div>
            </div>
        </div>
    </body>
    </html>
    """

@router.post("/send")
async def send_feedback(
    message: str = Form(...),
    images: Optional[List[UploadFile]] = File(None)
):
    """
    Envía feedback por correo electrónico usando Gmail SMTP con imágenes adjuntas y HTML estilizado
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
        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"Nuevo Feedback de SafeCode - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        images_count = len(images) if images else 0
        
        # Crear versión HTML
        html_content = create_html_email_template(message, timestamp, images_count)
        
        # Crear versión texto plano como fallback
        text_content = f"""
Nuevo feedback recibido en SafeCode:

Fecha: {timestamp}
Archivos adjuntos: {images_count}

Mensaje:
{message}

---
Este mensaje fue enviado desde el sistema SafeCode de detección de copias.
Proyecto de graduación - USAC FIUSAC ECYS
        """
        
        # Adjuntar ambas versiones
        part_text = MIMEText(text_content, "plain", "utf-8")
        part_html = MIMEText(html_content, "html", "utf-8")
        
        msg.attach(part_text)
        msg.attach(part_html)
        
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
                "images_count": images_count,
                "email_format": "HTML + Text"
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