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

ip_likes = {}
total_likes = 0  # Nuevo contador global

async def handle_like(request):
    global total_likes
    client_ip = request.remote
    if client_ip in ip_likes:
        return web.json_response({
            'status': 'error',
            'message': 'Ya has dado like.',
            'has_liked': True,
            'total_likes': total_likes
        }, status=400)

    ip_likes[client_ip] = True
    total_likes += 1  # Incrementar contador
    return web.json_response({
        'status': 'success',
        'message': 'Like recibido.',
        'has_liked': True,
        'total_likes': total_likes
    })

async def check_like(request):
    client_ip = request.remote
    has_liked = client_ip in ip_likes
    return web.json_response({
        'status': 'success' if has_liked else 'error',
        'has_liked': has_liked,
        'total_likes': total_likes
    })

async def like_status(request):
    client_ip = request.remote
    client_data = client_limits.get(client_ip, {"likes": 0})
    liked = client_data["likes"] >= 1
    return web.json_response({"liked": liked})

# Ruta para obtener el estado de las sugerencias y likes
async def handle_status(request):
    client_ip = request.remote
    client_data = client_limits.setdefault(client_ip, {"likes": 0, "suggestions": 0})
    return web.json_response(client_data)

# Manejar el envío de sugerencias
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

        # Enviar correo
        try:
            msg = MIMEText(f"Sugerencia de IP {client_ip}:\n\n{text}")
            msg["Subject"] = "Nueva sugerencia en Manus-Tiles"
            msg["From"] = EMAIL_USER
            msg["To"] = EMAIL_USER  # Puede ser otro correo si querés

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(msg)

        except Exception as e:
            logging.error("Error al enviar correo: %s", e)
            return web.json_response({"success": False, "message": "No se pudo enviar el correo."}, status=500)

        # Incrementar contador local si el correo se envió bien
        client_data["suggestions"] += 1
        logging.info(f"[SUGGESTION] {client_ip} envió: {text}")

        return web.json_response({"success": True, "message": "¡Gracias por tu sugerencia!"})

    except Exception as e:
        logging.error("Error interno en handle_suggestion: %s", e)
        return web.json_response({"success": False, "message": "Error interno"}, status=500)

# --- Inicializar servidor ---
async def start_servers():
    app = web.Application()
    app.router.add_get("/", serve_index)
    app.router.add_get("/ws", aiohttp_websocket_handler)
    app.router.add_post("/like", handle_like)
    app.router.add_get("/like/status", like_status)
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
