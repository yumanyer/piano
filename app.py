import cv2
import mediapipe as mp
import asyncio
# import websockets # <--- Eliminado
import json
import base64
import traceback
import os # <--- Añadido para leer el puerto
from aiohttp import web, WSMsgType # <--- Añadido WSMsgType

# LANDMARK (sin cambios)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Dedo abajo (sin cambios)
def dedo_abajo(landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx, is_thumb=False):
    try:
        finger_tip = landmarks[finger_tip_idx]
        if is_thumb:
            compare_landmark = landmarks[mp_hands.HandLandmark.THUMB_IP]
        else:
            compare_landmark = landmarks[finger_pip_idx]
        return finger_tip.y > compare_landmark.y
    except IndexError:
        print(f"Error: Índice fuera de rango al acceder a landmarks. Indices: tip={finger_tip_idx}, pip={finger_pip_idx}, mcp={finger_mcp_idx}. ¿Hay {len(landmarks)} landmarks?")
        return False
    except Exception as e:
        print(f"Error inesperado en dedo_abajo: {e}")
        traceback.print_exc()
        return False

# --- Estado Global (sin cambios) ---
connected_clients = set() # Ahora almacenará objetos aiohttp.web.WebSocketResponse
finger_state = [False] * 10
selected_camera_index = 0
camera_task = None

# --- MODIFICADO: send_to_clients para aiohttp ---
async def send_to_clients(message):
    """Envía un mensaje JSON a todos los clientes WebSocket conectados usando aiohttp."""
    if connected_clients:
        # aiohttp prefiere enviar el objeto dict directamente con send_json
        # json_message = json.dumps(message) # No es necesario para send_json
        tasks = []
        for client in connected_clients:
            # Verificar si el cliente sigue conectado antes de intentar enviar
            if not client.closed:
                # Usar send_json para enviar dicts directamente
                tasks.append(client.send_json(message))
            else:
                # Si ya está cerrado, podemos marcarlo para eliminar
                # (aunque el manejador principal también lo hará al detectar el cierre)
                 print(f"Cliente ya cerrado encontrado en send_to_clients.")


        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            disconnected_clients = set()
            client_list = list(connected_clients) # Iterar sobre copia

            for i, result in enumerate(results):
                # Verificar si el índice i es válido para client_list (puede haber clientes ya cerrados filtrados arriba)
                if i < len(client_list):
                    client = client_list[i]
                    if isinstance(result, Exception):
                        # Aiohttp podría cerrar la conexión automáticamente en algunos errores,
                        # pero es bueno registrarlo y asegurarse de eliminarlo del set.
                        remote_addr = client.remote_address if hasattr(client, 'remote_address') else "desconocido" # Obtener IP si está disponible
                        print(f"Error al enviar a {remote_addr}: {result}. Marcando cliente para eliminar.")
                        disconnected_clients.add(client)
                        # Opcionalmente, intentar cerrar explícitamente si aún no está cerrado
                        if not client.closed:
                             await client.close()


            # Eliminar clientes desconectados después de la iteración
            connected_clients.difference_update(disconnected_clients)

# --- process_camera (sin cambios internos, pero usará el send_to_clients modificado) ---
async def process_camera():
    global finger_state, selected_camera_index, camera_task

    print(f"Intentando abrir cámara con índice: {selected_camera_index}")
    cap = cv2.VideoCapture(selected_camera_index, cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"Error crítico: No se pudo abrir la cámara con índice {selected_camera_index}.")
        await send_to_clients({"type": "error", "message": f"Failed to open camera index {selected_camera_index}"})
        camera_task = None
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Cámara {selected_camera_index} abierta ({frame_width}x{frame_height} @ {fps if fps > 0 else 'N/A'} FPS). Iniciando detección...")
    await send_to_clients({"type": "status", "message": f"Camera {selected_camera_index} opened."})

    last_sent_state = list(finger_state)

    try:
        with mp_hands.Hands(
                model_complexity=0,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                max_num_hands=2
        ) as hands:
            while cap.isOpened() and len(connected_clients) > 0:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"Error: No se pudo leer el frame (ret={ret}, frame is None={frame is None}). Deteniendo cámara.")
                        break

                    # 1. Preprocesamiento
                    frame = cv2.flip(frame, 1)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False

                    # 2. Detección con Mediapipe
                    results = hands.process(rgb_frame)
                    draw_frame = frame.copy()

                    # 3. Procesamiento de resultados
                    processed_fingers_this_frame = set()
                    if results.multi_hand_landmarks and results.multi_handedness:
                        hand_positions = []
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            x_coords = [lm.x for lm in hand_landmarks.landmark]
                            hand_center_x = sum(x_coords) / len(x_coords)
                            hand_label = handedness.classification[0].label
                            hand_positions.append((hand_center_x, hand_landmarks, hand_label))

                        hand_positions.sort(key=lambda x: x[0])

                        for i, (center_x, hand_landmarks, hand_label) in enumerate(hand_positions):
                            if len(hand_positions) == 2:
                                hand_id_start = 6 if i == 0 else 1
                            else:
                                hand_id_start = 1

                            finger_tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
                            finger_pip = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
                            finger_mcp = [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]

                            for j in range(5):
                                h, w, _ = draw_frame.shape
                                try:
                                    tip_landmark = hand_landmarks.landmark[finger_tips[j]]
                                    mcp_landmark = hand_landmarks.landmark[finger_mcp[j]]
                                    pip_landmark = hand_landmarks.landmark[finger_pip[j]]

                                    x_tip = int(tip_landmark.x * w)
                                    y_tip = int(tip_landmark.y * h)
                                    x_mcp = int(mcp_landmark.x * w)
                                    y_mcp = int(mcp_landmark.y * h)

                                    cv2.circle(draw_frame, (x_tip, y_tip), 8, (0, 0, 255), -1)
                                    cv2.circle(draw_frame, (x_mcp, y_mcp), 8, (0, 255, 255), -1)

                                    numero_dedo = hand_id_start + j
                                    cv2.putText(draw_frame, str(numero_dedo), (x_tip - 15, y_tip - 15),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                    is_down = dedo_abajo(
                                        hand_landmarks.landmark,
                                        finger_tips[j], finger_pip[j], finger_mcp[j], is_thumb=(j == 0)
                                    )
                                    global_finger_index = numero_dedo - 1
                                    if 0 <= global_finger_index < 10:
                                        processed_fingers_this_frame.add(global_finger_index)
                                        finger_state[global_finger_index] = is_down
                                    else:
                                         print(f"Advertencia: Índice de dedo global fuera de rango: {global_finger_index} (Número dedo: {numero_dedo})")
                                except IndexError:
                                     print(f"Error de índice procesando dedo {j} de mano {i}. Landmarks disponibles: {len(hand_landmarks.landmark)}")
                                     continue

                            mp_drawing.draw_landmarks(
                                 draw_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 landmark_drawing_spec=None,
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                            )

                    for k in range(10):
                        if k not in processed_fingers_this_frame:
                            finger_state[k] = False

                    # 4. Comparar y enviar cambios
                    state_changed = False
                    for k in range(10):
                        if finger_state[k] != last_sent_state[k]:
                            state_changed = True
                            event_type = "finger_down" if finger_state[k] else "finger_up"
                            await send_to_clients({"type": event_type, "finger_id": k}) # <--- Llamará al modificado
                    if state_changed:
                        last_sent_state = list(finger_state)

                    # 5. Enviar frame
                    if connected_clients:
                        try:
                            ret_encode, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ret_encode:
                                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                                await send_to_clients({"type": "video_frame", "image": jpg_as_text}) # <--- Llamará al modificado
                        except Exception as e:
                            print(f"Error codificando/enviando video frame: {e}")

                    await asyncio.sleep(0.01)

                except asyncio.CancelledError:
                    print("Bucle interno de process_camera cancelado.")
                    raise
                except Exception as e_inner:
                    print(f"\n--- Error en el bucle interno de process_camera ({e_inner}) ---")
                    traceback.print_exc()
                    await asyncio.sleep(1)

    except asyncio.CancelledError:
        print("Tarea process_camera cancelada externamente.")
    except Exception as e_outer:
        print("\n--- Error mayor en process_camera ---")
        traceback.print_exc()
        await send_to_clients({"type": "error", "message": "Camera processing failed."})
    finally:
        print("Liberando cámara...")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        print("Procesamiento de cámara detenido.")
        camera_task = None


# --- ELIMINADO: async def handler(websocket): ---
# La lógica ahora estará en aiohttp_websocket_handler


# --- Handler HTTP (sin cambios) ---
async def http_handler(request):
    """Manejador simple para peticiones HTTP GET en la raíz."""
    print("Petición HTTP recibida en /")
    return web.Response(text="Servidor WebSocket y HTTP activos.")


# --- NUEVO: Manejador WebSocket para aiohttp ---
async def aiohttp_websocket_handler(request):
    """Maneja las conexiones WebSocket entrantes usando aiohttp."""
    global finger_state, selected_camera_index, camera_task, connected_clients

    ws = web.WebSocketResponse(max_msg_size=2*1024*1024) # Crear instancia de WebSocketResponse
    await ws.prepare(request) # Preparar la conexión (handshake)

    remote_addr = request.remote
    print(f"Cliente WebSocket conectado (aiohttp): {remote_addr}")
    connected_clients.add(ws) # Añadir al set de clientes

    try:
        # Enviar estado inicial al nuevo cliente
        initial_state_msg = {"type": "initial_state", "fingers": finger_state}
        await ws.send_json(initial_state_msg)
        print(f"Enviado estado inicial a {remote_addr}")

        # Enviar estado/índice de cámara actual
        is_running = camera_task is not None and not camera_task.done()
        await ws.send_json({"type": "camera_status", "index": selected_camera_index, "running": is_running})

        # Bucle para recibir mensajes del cliente
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    print(f"Mensaje recibido de {remote_addr} (aiohttp): {data}")

                    if data.get('type') == 'set_camera_index':
                        new_index = int(data.get('index', -1))
                        if new_index < 0:
                            print(f"Índice de cámara inválido recibido: {data.get('index')}")
                            continue # Ignorar mensaje inválido

                        print(f"Solicitud para usar cámara con índice: {new_index} de {remote_addr}")

                        needs_restart = False
                        is_running = camera_task is not None and not camera_task.done() # Recalcular estado actual

                        if not is_running:
                            print("Tarea de cámara no activa. Iniciando...")
                            needs_restart = True
                        elif new_index != selected_camera_index:
                            print(f"Índice de cámara diferente solicitado ({new_index} vs {selected_camera_index}). Reiniciando...")
                            needs_restart = True
                        else:
                            print(f"Cámara {new_index} ya seleccionada y corriendo. No se reinicia.")
                            # Reenviar estado por si acaso
                            await ws.send_json({"type": "initial_state", "fingers": finger_state})

                        if needs_restart:
                            selected_camera_index = new_index

                            # Cancelar tarea anterior si existe
                            if camera_task and not camera_task.done():
                                print("Cancelando tarea de cámara anterior...")
                                camera_task.cancel()
                                try:
                                    await camera_task
                                except asyncio.CancelledError:
                                    print("Tarea de cámara anterior cancelada.")
                                camera_task = None

                            # Iniciar nueva tarea (solo si hay clientes)
                            if connected_clients: # Verificar si todavía hay clientes (podría haberse desconectado mientras se procesaba)
                                print(f"Iniciando nueva tarea process_camera con índice {selected_camera_index}...")
                                camera_task = asyncio.create_task(process_camera())
                                # Informar a TODOS los clientes del cambio
                                await send_to_clients({"type": "camera_status", "index": selected_camera_index, "running": True})
                            else:
                                print("No hay clientes conectados, no se inicia la cámara.")
                                # Informar estado (no corriendo) aunque no haya clientes (podría haber otro que se conecte)
                                await send_to_clients({"type": "camera_status", "index": selected_camera_index, "running": False})

                    else:
                         print(f"Mensaje no reconocido o sin acción (aiohttp): {data}")

                except json.JSONDecodeError:
                    print(f"Mensaje no JSON recibido de {remote_addr} (aiohttp): {msg.data}")
                except ValueError:
                     print(f"Error convirtiendo índice de cámara a entero. Mensaje: {msg.data}")
                except asyncio.CancelledError:
                     print(f"Tarea handler aiohttp cancelada para {remote_addr}")
                     raise # Re-lanzar para que el finally se ejecute
                except Exception as e:
                    print(f"Error procesando mensaje de {remote_addr} (aiohttp): {e}")
                    traceback.print_exc()
                    # Podríamos enviar un error al cliente si la conexión sigue activa
                    # await ws.send_json({"type": "error", "message": "Internal server error processing message"})

            elif msg.type == WSMsgType.ERROR:
                print(f'Conexión WebSocket cerrada con excepción {ws.exception()} para {remote_addr}')
                # El bucle terminará automáticamente
            elif msg.type == WSMsgType.CLOSE:
                 print(f"Cliente {remote_addr} solicitó cierre (CLOSE).")
                 # El bucle terminará automáticamente
            elif msg.type == WSMsgType.CLOSED:
                print(f"Conexión ya cerrada para {remote_addr} (CLOSED).")
                 # El bucle terminará automáticamente

    except asyncio.CancelledError:
        print(f"Tarea de manejo de WebSocket cancelada para {remote_addr}.")
        # Asegurar que se cierre la conexión si la tarea es cancelada externamente
        if not ws.closed:
             await ws.close(code=1012, message='Server shutdown') # 1012 = Service Restart
    except Exception as e:
        print(f"Error inesperado en manejador WebSocket aiohttp para {remote_addr}:")
        traceback.print_exc()
    finally:
        print(f"Cliente WebSocket desconectado (aiohttp): {remote_addr}")
        # Asegurarse de eliminarlo del set
        connected_clients.discard(ws) # Usar discard es más seguro que remove
        # Si era el último cliente, cancelar la tarea de cámara
        if not connected_clients and camera_task and not camera_task.done():
             print("Último cliente (aiohttp) desconectado, cancelando tarea de cámara...")
             camera_task.cancel()
             camera_task = None # Resetear

    print(f"Finalizado el manejador WebSocket para {remote_addr}")
    return ws # Devuelve el objeto WebSocketResponse


# --- ELIMINADO: async def start_websocket_server(): ---


# --- MODIFICADO: Función para iniciar el servidor aiohttp (HTTP y WS) ---
async def start_servers():
    """Configura e inicia el servidor aiohttp para manejar HTTP y WebSocket."""
    global camera_task # Asegurarse de poder cancelar al final

    # Configurar aplicación aiohttp
    app = web.Application()
    app.router.add_get("/", http_handler)         # Ruta raíz para HTTP
    app.router.add_get("/ws", aiohttp_websocket_handler) # NUEVA ruta para WebSocket

    runner = web.AppRunner(app)
    await runner.setup()

    # Determinar el puerto (Render usa $PORT, default a 8080 localmente)
    port = int(os.environ.get("PORT", 8080))
    host = "0.0.0.0" # Escuchar en todas las interfaces

    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f"--- Servidor Unificado (HTTP y WebSocket) iniciado en http://{host}:{port} ---")
    print(f"--- Endpoint WebSocket disponible en ws://{host}:{port}/ws ---")
    print(f"--- Esperando conexiones HTTP y WebSocket ---")

    # Mantener la aplicación corriendo (similar a wait_closed)
    # Podemos esperar indefinidamente hasta que se cancele o interrumpa
    try:
        # Bucle infinito para mantener el servidor activo hasta cancelación
        while True:
            await asyncio.sleep(3600) # Dormir por largos periodos, despertará si se cancela
    except asyncio.CancelledError:
        print("Tarea start_servers cancelada.")
    finally:
        # Limpiar recursos de aiohttp
        print("Deteniendo servidor aiohttp...")
        await runner.cleanup()
        print("Servidor aiohttp detenido.")
        # Cancelar tarea de cámara si sigue activa al salir
        if camera_task and not camera_task.done():
             print("Cancelando tarea de cámara pendiente al cerrar servidor...")
             camera_task.cancel()
             try:
                 await camera_task
             except asyncio.CancelledError:
                 print("Tarea de cámara cancelada durante el cierre.")


if __name__ == "__main__":
    try:
        print("Iniciando la aplicación principal asyncio (Servidor Unificado)...")
        asyncio.run(start_servers()) # Ejecutar el servidor unificado
    except KeyboardInterrupt:
        print("\nInterrupción por teclado (Ctrl+C) detectada. Iniciando cierre limpio...")
    except Exception as e:
        print("\n--- Error Fatal en el Nivel Principal ---")
        traceback.print_exc()
    finally:
        print("--- Aplicación Finalizada ---")