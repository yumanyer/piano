import asyncio
import json
import traceback
import os
import pathlib
from aiohttp import web, WSMsgType

# --- Estado Global Simplificado ---
connected_clients = set() # Almacena objetos aiohttp.web.WebSocketResponse

# --- Función send_to_clients (Puede mantenerse por si necesitas broadcast, pero no se usa activamente ahora) ---
async def send_to_clients(message):
    """Envía un mensaje JSON a todos los clientes WebSocket conectados usando aiohttp."""
    if connected_clients:
        tasks = []
        for client in connected_clients:
            if not client.closed:
                tasks.append(client.send_json(message))
            else:
                 print(f"Cliente ya cerrado encontrado en send_to_clients.")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            disconnected_clients = set()
            client_list = list(connected_clients)

            for i, result in enumerate(results):
                if i < len(client_list):
                    client = client_list[i]
                    if isinstance(result, Exception):
                        remote_addr = client.remote_address if hasattr(client, 'remote_address') else "desconocido"
                        print(f"Error al enviar a {remote_addr}: {result}. Marcando cliente para eliminar.")
                        disconnected_clients.add(client)
                        if not client.closed:
                             # Podrías intentar cerrar aquí, pero aiohttp a menudo lo maneja
                             # await client.close()
                             pass

            connected_clients.difference_update(disconnected_clients)


# --- Definir rutas base y estáticas ---
BASE_DIR = pathlib.Path(__file__).parent
FRONTEND_DIR = BASE_DIR / 'frontend'
SOUNDS_DIR = BASE_DIR / 'sounds'

# --- Handler para servir index.html ---
async def serve_index(request):
    """Sirve el archivo index.html principal desde la carpeta frontend."""
    index_path = FRONTEND_DIR / 'index.html'
    try:
        # Verificar si el archivo existe primero
        if not index_path.is_file():
             print(f"ERROR: No se encontró {index_path}")
             return web.Response(text="Interfaz no encontrada (index.html no existe).", status=404)

        with open(index_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    except Exception as e:
         print(f"ERROR: Leyendo index.html ({index_path}): {e}")
         return web.Response(text="Error interno al cargar la interfaz.", status=500)


# --- Manejador WebSocket para aiohttp (Simplificado) ---
async def aiohttp_websocket_handler(request):
    """Maneja conexiones WebSocket. Ahora principalmente recibe eventos del cliente."""
    global connected_clients # Solo necesitamos el set de clientes

    ws = web.WebSocketResponse(max_msg_size=2*1024*1024) # Aumentado por si acaso
    await ws.prepare(request)

    remote_addr = request.remote
    print(f"Cliente WebSocket conectado (aiohttp): {remote_addr}")
    connected_clients.add(ws)

    try:
        # Mensaje de bienvenida
        await ws.send_json({"type": "status", "message": "Conectado al servidor."})

        # Bucle para recibir mensajes del cliente
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    print(f"Mensaje recibido de {remote_addr} (aiohttp): {data}")

                    # --- Procesar mensajes del cliente ---
                    if data.get('type') == 'finger_event':
                        # El cliente ahora nos dice qué dedo cambió de estado
                        finger_id = data.get('finger_id')
                        state = data.get('state')
                        print(f"Evento recibido: Dedo {finger_id} -> {state}")
                        # Podrías hacer algo aquí si fuera necesario (sincronizar, etc.)
                        # Ejemplo: retransmitir a otros clientes (requiere lógica adicional)
                        # await send_to_clients(data) # <-- ¡Cuidado con bucles infinitos si retransmites!

                    elif data.get('type') == 'set_camera_index':
                        print("Mensaje 'set_camera_index' ignorado (manejo local en cliente).")

                    else:
                         print(f"Mensaje no reconocido recibido del cliente: {data}")
                # --- Fin procesamiento ---

                except json.JSONDecodeError:
                    print(f"Mensaje no JSON recibido de {remote_addr} (aiohttp): {msg.data}")
                except Exception as e:
                    print(f"Error procesando mensaje de {remote_addr} (aiohttp): {e}")
                    traceback.print_exc()

            elif msg.type == WSMsgType.ERROR:
                print(f'Conexión WebSocket cerrada con excepción {ws.exception()} para {remote_addr}')
            elif msg.type == WSMsgType.CLOSE:
                 print(f"Cliente {remote_addr} solicitó cierre (CLOSE).")
            elif msg.type == WSMsgType.CLOSED:
                print(f"Conexión ya cerrada para {remote_addr} (CLOSED).")

    except asyncio.CancelledError:
        print(f"Tarea de manejo de WebSocket cancelada para {remote_addr}.")
        if not ws.closed:
             await ws.close(code=1012, message='Server shutdown')
    except Exception as e:
        print(f"Error inesperado en manejador WebSocket aiohttp para {remote_addr}:")
        traceback.print_exc()
    finally:
        print(f"Cliente WebSocket desconectado (aiohttp): {remote_addr}")
        connected_clients.discard(ws)

    print(f"Finalizado el manejador WebSocket para {remote_addr}")
    return ws

# --- Función para iniciar el servidor aiohttp ---
async def start_servers():
    """Configura e inicia el servidor aiohttp para manejar HTTP, WebSocket y archivos estáticos."""
    app = web.Application()

    # --- CONFIGURACIÓN DE RUTAS (ORDEN IMPORTANTE) ---
    # 1. Ruta WebSocket (específica)
    app.router.add_get("/ws", aiohttp_websocket_handler)

    # 2. Ruta para servir index.html en la raíz (específica)
    app.router.add_get("/", serve_index)

    # 3. Servir archivos estáticos desde la carpeta 'sounds' bajo el prefijo '/sounds'
    app.router.add_static('/sounds', path=SOUNDS_DIR, name='sounds', show_index=False)
    print(f"--- Sirviendo sonidos desde: {SOUNDS_DIR} bajo la ruta /sounds ---")

    # 4. Servir archivos estáticos desde la carpeta 'frontend' (CSS, JS, etc.)
    app.router.add_static('/', path=FRONTEND_DIR, name='frontend_static', show_index=False)
    print(f"--- Sirviendo otros archivos estáticos (CSS, JS) desde: {FRONTEND_DIR} bajo la ruta / ---")

    # --- Inicio del servidor ---
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0"
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"--- Servidor Unificado (HTTP y WebSocket) iniciado en http://{host}:{port} ---")
    print(f"--- Endpoint WebSocket disponible en ws://{host}:{port}/ws ---")
    print(f"--- Esperando conexiones ---")

    # --- Mantener servidor corriendo ---
    try:
        # Mantenerse vivo indefinidamente hasta CancelledError
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        print("Tarea start_servers cancelada.")
    finally:
        print("Deteniendo servidor aiohttp...")
        await runner.cleanup()
        print("Servidor aiohttp detenido.")

# --- Punto de entrada principal ---
if __name__ == "__main__":
    try:
        print("Iniciando la aplicación principal asyncio (Servidor Simplificado)...")
        asyncio.run(start_servers())
    except KeyboardInterrupt:
        print("\nInterrupción por teclado (Ctrl+C) detectada. Iniciando cierre limpio...")
    except Exception as e:
        print("\n--- Error Fatal en el Nivel Principal ---")
        traceback.print_exc()
    finally:
        print("--- Aplicación Finalizada ---")