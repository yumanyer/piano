import asyncio
import json
import traceback
import os
import pathlib
from aiohttp import web, WSMsgType
from email.mime.text import MIMEText
import smtplib
from dotenv import load_dotenv
import logging  # Importa logging

logging.basicConfig(level=logging.DEBUG)  # Configura el logging al principio

# Cargar variables de entorno
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")

# --- Estado Global ---
connected_clients = set()
client_limits = {}  # Estructura: {ip: {"likes": int, "suggestions": int}}

# --- Funciones auxiliares ---
async def send_to_clients(message):
    if connected_clients:
        tasks = []
        for client in connected_clients:
            if not client.closed:
                tasks.append(client.send_json(message))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            disconnected_clients = set()
            for client, result in zip(connected_clients, results):
                if isinstance(result, Exception):
                    disconnected_clients.add(client)
            connected_clients.difference_update(disconnected_clients)

# --- Paths base ---
BASE_DIR = pathlib.Path(__file__).parent
FRONTEND_DIR = BASE_DIR / 'frontend'
SOUNDS_DIR = BASE_DIR / 'sounds'

# --- Index ---
async def serve_index(request):
    index_path = FRONTEND_DIR / 'index.html'
    try:
        if not index_path.is_file():
            return web.Response(text="Interfaz no encontrada.", status=404)
        with open(index_path, 'r', encoding='utf-8') as f:
            return web.Response(text=f.read(), content_type='text/html')
    except Exception as e:
        print(f"Error cargando index.html: {e}")
        return web.Response(text="Error interno al cargar interfaz.", status=500)

# --- WebSocket ---
async def aiohttp_websocket_handler(request):
    global connected_clients
    ws = web.WebSocketResponse(max_msg_size=2*1024*1024)
    await ws.prepare(request)
    remote_addr = request.remote
    print(f"Cliente WebSocket conectado: {remote_addr}")
    connected_clients.add(ws)

    try:
        await ws.send_json({"type": "status", "message": "Conectado al servidor."})

        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    print(f"Mensaje de {remote_addr}: {data}")

                    if data.get('type') == 'finger_event':
                        print(f"Dedo {data.get('finger_id')} -> {data.get('state')}")
                    elif data.get('type') == 'set_camera_index':
                        print("Cámara cambiada (ignorado).")
                except Exception as e:
                    print(f"Error procesando mensaje: {e}")
                    traceback.print_exc()
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                print(f"Cliente desconectado: {remote_addr}")
    except Exception as e:
        print(f"Error WebSocket: {e}")
    finally:
        connected_clients.discard(ws)
        print(f"WebSocket cerrado: {remote_addr}")
    return ws

# Para almacenar los likes por IP (en memoria, por simplicidad)
ip_likes = {}

# Manejar el "like" desde el frontend
async def handle_like(request):

    client_ip = request.remote
    if client_ip in ip_likes:
        return web.json_response({'status': 'error', 'message': 'Ya has dado like.'}, status=400)
    
    ip_likes[client_ip] = True  # Marcamos que la IP ya dio like
    return web.json_response({'status': 'success', 'message': 'Like recibido.'})
# Ruta para verificar si el usuario ya ha dado like
async def check_like(request):
    client_ip = request.remote
    if client_ip in ip_likes:
        return web.json_response({'status': 'success', 'message': 'Ya diste like.'})
    else:
        return web.json_response({'status': 'error', 'message': 'Aún no has dado like.'})
# --- Sugerencias ---
async def handle_suggestion(request):
    try:
        client_ip = request.remote
        client_data = client_limits.setdefault(client_ip, {"likes": 0, "suggestions": 0})
        if client_data["suggestions"] >= 3:
            return web.json_response({"success": False, "message": "Máximo de sugerencias alcanzado."}, status=429)

        data = await request.json()
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"success": False, "message": "Texto vacío."}, status=400)

        msg = MIMEText(text)
        msg['Subject'] = f"Sugerencia de {client_ip}"
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_USER

        try: # Encierra la conexión SMTP en un bloque try...except
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                logging.debug("Conectando al servidor SMTP...")
                server.set_debuglevel(1)  # Activar el modo de depuración de smtplib
                logging.debug(f"Intentando login con usuario: {EMAIL_USER}")
                server.login(EMAIL_USER, EMAIL_PASS)
                logging.debug("Login exitoso.")
                server.send_message(msg)
                logging.debug("Correo enviado.")

        except Exception as e:  # Captura cualquier excepción que ocurra
            logging.error(f"Error al enviar correo: {e}")
            traceback.print_exc() # Imprime el traceback completo

        client_data["suggestions"] += 1
        print(f"Sugerencia de {client_ip}")
        return web.json_response({"success": True, "message": "¡Gracias por tu sugerencia!"})
    except Exception as e:
        print(f"Error en sugerencia: {e}")
        traceback.print_exc()
        return web.json_response({"success": False, "message": "Error interno"}, status=500)
# --- Estado de IP ---
async def handle_status(request):
    client_ip = request.remote
    client_data = client_limits.setdefault(client_ip, {"likes": 0, "suggestions": 0})
    return web.json_response(client_data)

# --- Inicializar servidor ---
async def start_servers():
    app = web.Application()
    app.router.add_get("/", serve_index)
    app.router.add_get("/ws", aiohttp_websocket_handler)
    app.router.add_post("/like", handle_like)
    app.router.add_get("/check_like", check_like)
    app.router.add_post("/suggest", handle_suggestion)
    app.router.add_get("/status", handle_status)
    app.router.add_static("/sounds", path=SOUNDS_DIR, name="sounds")
    app.router.add_static("/", path=FRONTEND_DIR, name="frontend_static")

    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"Servidor en http://{host}:{port}")
    print(f"WebSocket en ws://{host}:{port}/ws")

    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        print("Servidor detenido.")
    finally:
        await runner.cleanup()

# --- Ejecutar ---
if __name__ == "__main__":
    try:
        asyncio.run(start_servers())
    except KeyboardInterrupt:
        print("Interrumpido por teclado.")
    except Exception as e:
        print("Error fatal:")
        traceback.print_exc()
    finally:
        print("Servidor cerrado.")
