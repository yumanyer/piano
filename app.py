import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import traceback
# --- NUEVA IMPORTACIÓN ---
from aiohttp import web

# LANDMARK
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# landmarks = la punta de los dedos de la mano
# finger_tip = nudillos de la mano

def dedo_abajo(landmarks, finger_tip_idx, finger_pip_idx, finger_mcp_idx, is_thumb=False):
    try:
        finger_tip = landmarks[finger_tip_idx]
        if is_thumb:
            # Para el pulgar, a menudo es mejor comparar con el nudillo IP (InterPhalangeal)
            compare_landmark = landmarks[mp_hands.HandLandmark.THUMB_IP]
        else:
            # Para otros dedos, comparar con el nudillo PIP (Proximal InterPhalangeal)
            compare_landmark = landmarks[finger_pip_idx]
        # El dedo está 'abajo' si la punta está más abajo (mayor coordenada Y) que el nudillo de comparación
        return finger_tip.y > compare_landmark.y
    except IndexError:
        print(f"Error: Índice fuera de rango al acceder a landmarks. Indices: tip={finger_tip_idx}, pip={finger_pip_idx}, mcp={finger_mcp_idx}. ¿Hay {len(landmarks)} landmarks?")
        return False
    except Exception as e:
        print(f"Error inesperado en dedo_abajo: {e}")
        traceback.print_exc()
        return False

#  WebSocket
connected_clients = set()
finger_state = [False] * 10
selected_camera_index = 0 # index de la camara que se usara por defecto
camera_task = None # <--- la camara existe pero todavia no asignamos niguna

async def send_to_clients(message):
    if connected_clients:
        json_message = json.dumps(message)
        # Usar asyncio.gather para enviar a todos concurrentemente
        tasks = [client.send(json_message) for client in connected_clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        disconnected_clients = set()
        # Usar una copia de la lista de clientes para iterar de forma segura
        client_list = list(connected_clients)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client = client_list[i]
                print(f"Error al enviar a {client.remote_address}: {result}. Eliminando cliente.")
                # Marcar para eliminar, no eliminar durante la iteración
                disconnected_clients.add(client)

        # Eliminar clientes desconectados después de la iteración
        connected_clients.difference_update(disconnected_clients)


async def process_camera():
    global finger_state, selected_camera_index, camera_task

    print(f"Intentando abrir cámara con índice: {selected_camera_index}")
    # Intentar abrir con backend preferido si hay problemas
    cap = cv2.VideoCapture(selected_camera_index, cv2.CAP_ANY) # O cv2.CAP_DSHOW en Windows, cv2.CAP_V4L2 en Linux

    if not cap.isOpened():
        print(f"Error crítico: No se pudo abrir la cámara con índice {selected_camera_index}.")
        await send_to_clients({"type": "error", "message": f"Failed to open camera index {selected_camera_index}"})
        camera_task = None # Marcar la tarea como no existente
        return

    # Intentar configurar propiedades, pero continuar si fallan algunas
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # Puede devolver 0 si no es soportado
    print(f"Cámara {selected_camera_index} abierta ({frame_width}x{frame_height} @ {fps if fps > 0 else 'N/A'} FPS). Iniciando detección...")
    await send_to_clients({"type": "status", "message": f"Camera {selected_camera_index} opened."})

    last_sent_state = list(finger_state)

    try:
        with mp_hands.Hands(
                model_complexity=0,        # Más rápido, menos preciso
                min_detection_confidence=0.6, # Umbral para detectar una mano
                min_tracking_confidence=0.6, # Umbral para seguir la mano una vez detectada
                max_num_hands=2            # Detectar hasta 2 manos
        ) as hands:
            while cap.isOpened() and len(connected_clients) > 0:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print(f"Error: No se pudo leer el frame (ret={ret}, frame is None={frame is None}). Deteniendo cámara.")
                        break # Salir del bucle si falla la lectura

                    # 1. Preprocesamiento
                    frame = cv2.flip(frame, 1) # Espejar horizontalmente
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convertir a RGB para MediaPipe
                    rgb_frame.flags.writeable = False # Optimización: marcar como no escribible

                    # 2. Detección con Mediapipe
                    results = hands.process(rgb_frame)

                    # Crear frame para dibujar (copia del frame original BGR)
                    draw_frame = frame.copy()

                    # 3. Procesamiento de resultados, asignación de números y actualización del estado de dedos
                    processed_fingers_this_frame = set() # Dedos detectados en ESTE frame

                    if results.multi_hand_landmarks and results.multi_handedness:
                        hand_positions = [] # Lista para almacenar (centro_x, landmarks, label)
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            # Calcular el centro horizontal aproximado de la mano
                            x_coords = [lm.x for lm in hand_landmarks.landmark]
                            hand_center_x = sum(x_coords) / len(x_coords)
                            hand_label = handedness.classification[0].label # 'Left' o 'Right'
                            hand_positions.append((hand_center_x, hand_landmarks, hand_label))

                        # Ordenar manos de izquierda a derecha en la imagen (menor x a mayor x)
                        hand_positions.sort(key=lambda x: x[0])

                        for i, (center_x, hand_landmarks, hand_label) in enumerate(hand_positions):
                             # Asignar hand_id:
                             # Si hay 2 manos, la de la izquierda (índice 0 en la lista ordenada) es 6-10.
                             # Si hay 2 manos, la de la derecha (índice 1) es 1-5.
                             # Si hay 1 mano, siempre es 1-5.
                             # NOTA: Esto asume que la cámara no está invertida de forma extraña.
                             # 'Left' en handedness se refiere a la mano real (izquierda del usuario).
                             # Si la cámara está espejada (como es común), la mano IZQUIERDA del usuario aparecerá a la DERECHA de la imagen.
                             # Ajustamos la lógica basada en la POSICIÓN en la imagen.
                             if len(hand_positions) == 2:
                                hand_id_start = 6 if i == 0 else 1 # Mano a la izquierda de la IMAGEN -> 6-10, derecha -> 1-5
                             else: # Solo una mano detectada
                                hand_id_start = 1 # Siempre 1-5 si solo hay una

                             # Definir puntos clave para cada dedo (pulgar a meñique)
                             finger_tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
                             finger_pip = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
                             finger_mcp = [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]

                             for j in range(5): # Iterar por los 5 dedos (0=pulgar, 4=meñique)
                                h, w, _ = draw_frame.shape # Obtener dimensiones del frame para convertir coordenadas normalizadas
                                try:
                                    tip_landmark = hand_landmarks.landmark[finger_tips[j]]
                                    mcp_landmark = hand_landmarks.landmark[finger_mcp[j]]
                                    pip_landmark = hand_landmarks.landmark[finger_pip[j]] # Necesario para dedo_abajo

                                    x_tip = int(tip_landmark.x * w)
                                    y_tip = int(tip_landmark.y * h)
                                    x_mcp = int(mcp_landmark.x * w)
                                    y_mcp = int(mcp_landmark.y * h)

                                    # Dibujar la punta del dedo en rojo (círculo más grande)
                                    cv2.circle(draw_frame, (x_tip, y_tip), 8, (0, 0, 255), -1)
                                    # Dibujar el nudillo (MCP) en amarillo
                                    cv2.circle(draw_frame, (x_mcp, y_mcp), 8, (0, 255, 255), -1)

                                    # Asignar número al dedo (hand_id_start + j) y mostrarlo cerca de la punta
                                    numero_dedo = hand_id_start + j
                                    cv2.putText(draw_frame, str(numero_dedo), (x_tip - 15, y_tip - 15),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                    # Actualizar el estado del dedo usando la función dedo_abajo
                                    is_down = dedo_abajo(
                                        hand_landmarks.landmark,
                                        finger_tips[j], # Índice del landmark de la punta
                                        finger_pip[j],  # Índice del landmark PIP (o IP para pulgar)
                                        finger_mcp[j],  # Índice del landmark MCP
                                        is_thumb=(j == 0) # Indicar si es el pulgar
                                    )
                                    global_finger_index = numero_dedo - 1 # Convertir a índice 0-9
                                    if 0 <= global_finger_index < 10:
                                        processed_fingers_this_frame.add(global_finger_index)
                                        finger_state[global_finger_index] = is_down
                                    else:
                                         print(f"Advertencia: Índice de dedo global fuera de rango: {global_finger_index} (Número dedo: {numero_dedo})")


                                except IndexError:
                                     print(f"Error de índice procesando dedo {j} de mano {i}. Landmarks disponibles: {len(hand_landmarks.landmark)}")
                                     continue # Saltar al siguiente dedo

                             # Dibujar las conexiones entre landmarks en blanco
                             mp_drawing.draw_landmarks(
                                 draw_frame,
                                 hand_landmarks,
                                 mp_hands.HAND_CONNECTIONS,
                                 landmark_drawing_spec=None, # No dibujar puntos individuales (ya los dibujamos)
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2) # Conexiones blancas
                             )

                    # Marcar como 'up' (False) los dedos que no fueron detectados en este frame
                    # Esto asegura que si una mano desaparece, sus dedos se marquen como 'up'
                    for k in range(10):
                        if k not in processed_fingers_this_frame:
                            finger_state[k] = False

                    # 4. Comparar y enviar cambios de estado a los clientes conectados
                    state_changed = False
                    for k in range(10):
                        if finger_state[k] != last_sent_state[k]:
                            state_changed = True
                            event_type = "finger_down" if finger_state[k] else "finger_up"
                            # Enviar evento individual por cada cambio
                            await send_to_clients({"type": event_type, "finger_id": k})
                    if state_changed:
                        last_sent_state = list(finger_state) # Actualizar último estado enviado

                    # 5. Enviar el frame procesado (con dibujos) a los clientes vía WebSocket
                    if connected_clients:
                        try:
                            # Codificar frame a JPG con calidad moderada para reducir tamaño
                            ret_encode, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ret_encode:
                                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                                await send_to_clients({"type": "video_frame", "image": jpg_as_text})
                        except Exception as e:
                            print(f"Error codificando/enviando video frame: {e}")

                    # Ceder control brevemente para que otras tareas (como websockets) puedan ejecutarse
                    await asyncio.sleep(0.01) # Pequeña pausa para evitar consumir 100% CPU y permitir I/O

                except asyncio.CancelledError:
                    print("Bucle interno de process_camera cancelado.")
                    raise # Re-lanzar para que el bloque finally externo se ejecute
                except Exception as e_inner:
                    print(f"\n--- Error en el bucle interno de process_camera ({e_inner}) ---")
                    traceback.print_exc()
                    await asyncio.sleep(1) # Pausa tras error interno antes de reintentar

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
        # Asegurarse de que la tarea global se marque como no existente al finalizar
        camera_task = None


async def handler(websocket):
    global finger_state, selected_camera_index, camera_task, connected_clients

    remote_addr = websocket.remote_address
    print(f"Cliente conectado: {remote_addr}")
    connected_clients.add(websocket)
    # Cada cliente necesita saber si ya procesó el índice inicial
    # Para evitar que un cliente cambie la cámara de otro accidentalmente
    # si envían el mensaje al mismo tiempo. El ÚLTIMO en establecerlo gana.
    # O mejor, solo permitir cambiar si la cámara no está activa o el índice es diferente.

    try:
        # Enviar estado inicial de los dedos al nuevo cliente al conectarse
        initial_state_msg = {"type": "initial_state", "fingers": finger_state}
        await websocket.send(json.dumps(initial_state_msg))
        print(f"Enviado estado inicial a {remote_addr}")

        # Enviar índice de cámara actual si ya está corriendo
        if camera_task and not camera_task.done():
             await websocket.send(json.dumps({"type": "camera_status", "index": selected_camera_index, "running": True}))
        else:
             await websocket.send(json.dumps({"type": "camera_status", "index": selected_camera_index, "running": False}))


        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Mensaje recibido de {remote_addr}: {data}")

                if data.get('type') == 'set_camera_index':
                    new_index = int(data.get('index', -1)) # Usar -1 como inválido
                    if new_index < 0:
                        print(f"Índice de cámara inválido recibido: {data.get('index')}")
                        continue

                    print(f"Solicitud para usar cámara con índice: {new_index} de {remote_addr}")

                    # Cambiar/Iniciar cámara SOLO si el índice es diferente al actual
                    # O si la tarea de cámara no está corriendo actualmente
                    needs_restart = False
                    if camera_task is None or camera_task.done():
                        print("Tarea de cámara no activa. Iniciando...")
                        needs_restart = True
                    elif new_index != selected_camera_index:
                        print(f"Índice de cámara diferente solicitado ({new_index} vs {selected_camera_index}). Reiniciando...")
                        needs_restart = True
                    else:
                        print(f"Cámara {new_index} ya seleccionada y corriendo. No se reinicia.")
                        # Reenviar estado por si acaso el cliente se perdió algo
                        await websocket.send(json.dumps({"type": "initial_state", "fingers": finger_state}))


                    if needs_restart:
                        selected_camera_index = new_index

                        # Cancelar tarea anterior si existe y está corriendo
                        if camera_task and not camera_task.done():
                            print("Cancelando tarea de cámara anterior...")
                            camera_task.cancel()
                            try:
                                await camera_task # Esperar a que se cancele completamente
                            except asyncio.CancelledError:
                                print("Tarea de cámara anterior cancelada.")
                            camera_task = None # Resetear la variable global

                        # Iniciar nueva tarea de cámara (solo si hay clientes conectados)
                        if connected_clients:
                            print(f"Iniciando nueva tarea process_camera con índice {selected_camera_index}...")
                            # Asegurarse de que la tarea se asigna a la variable global
                            camera_task = asyncio.create_task(process_camera())
                            # Informar a todos los clientes sobre el cambio/inicio
                            await send_to_clients(json.dumps({"type": "camera_status", "index": selected_camera_index, "running": True}))

                        else:
                             print("No hay clientes conectados, no se inicia la cámara aunque se solicitó.")
                             # Informar estado (no corriendo)
                             await send_to_clients(json.dumps({"type": "camera_status", "index": selected_camera_index, "running": False}))


                # Podrías añadir más tipos de mensajes aquí si es necesario
                # elif data.get('type') == 'otro_comando':
                #    ...

                else:
                     print(f"Mensaje no reconocido o sin acción: {data}")

            except json.JSONDecodeError:
                print(f"Mensaje no JSON recibido de {remote_addr}: {message}")
            except ValueError:
                 print(f"Error convirtiendo índice de cámara a entero. Mensaje: {message}")
            except asyncio.CancelledError:
                 print(f"Tarea handler cancelada para {remote_addr}")
                 raise # Re-lanzar para que el finally se ejecute correctamente
            except Exception as e:
                print(f"Error procesando mensaje de {remote_addr}: {e}")
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Cliente desconectado correctamente: {remote_addr}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Conexión de cliente cerrada con error: {remote_addr} - {e}")
    except Exception as e:
        print(f"Error inesperado en handler para {remote_addr}:")
        traceback.print_exc()
    finally:
        print(f"Eliminando cliente: {remote_addr}")
        connected_clients.remove(websocket)
        # Si era el último cliente, cancelar la tarea de cámara para liberar recursos
        if not connected_clients and camera_task and not camera_task.done():
             print("Último cliente desconectado, cancelando tarea de cámara...")
             camera_task.cancel()
             # No es estrictamente necesario esperar aquí, pero reseteamos la variable
             camera_task = None


# --- NUEVO HANDLER HTTP ---
async def http_handler(request):
    """Manejador simple para peticiones HTTP GET en la raíz."""
    print("Petición HTTP recibida en /")
    return web.Response(text="Servidor WebSocket y HTTP activos.")

# --- FUNCIÓN PRINCIPAL DEL SERVIDOR WEBSOCKET (renombrada de main a start_websocket_server) ---
async def start_websocket_server():
    """Inicia y gestiona el servidor WebSocket."""
    global camera_task # Indicar que usamos la global
    host = "0.0.0.0"  # Escucha en todas las interfaces de red
    port = 8765       # Puerto para WebSocket

    print(f"--- Iniciando Servidor WebSocket en ws://{host}:{port} ---")
    # El inicio de la cámara ahora se maneja cuando un cliente envía 'set_camera_index'
    print(f"--- Esperando conexión de cliente para seleccionar e iniciar cámara ---")

    # Crear y iniciar el servidor WebSocket
    # Aumentar max_size para permitir frames de video más grandes
    server = await websockets.serve(handler, host, port, max_size=2*1024*1024)

    print("Servidor WebSocket iniciado y escuchando...")

    try:
        # Mantener el servidor WebSocket corriendo hasta que se cierre o cancele
        await server.wait_closed()
    except asyncio.CancelledError:
         print("Servidor WebSocket principal cancelado.")
    finally:
        print("Cerrando servidor WebSocket...")
        # Cancelar la tarea de cámara si sigue corriendo al cerrar el servidor WS
        if camera_task and not camera_task.done():
            print("Cancelando tarea de cámara pendiente al cerrar WS...")
            camera_task.cancel()
            try:
                await camera_task # Esperar a que la cancelación termine
            except asyncio.CancelledError:
                print("Tarea de cámara cancelada durante el cierre del WS.")
        print("Servidor WebSocket cerrado.")


# --- NUEVA FUNCIÓN PARA INICIAR AMBOS SERVIDORES ---
async def start_servers():
    """Configura e inicia los servidores HTTP (aiohttp) y WebSocket (websockets)."""
    # Configurar servidor HTTP
    http_app = web.Application()
    http_app.router.add_get("/", http_handler) # Ruta raíz para HTTP
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    # Escuchar en 0.0.0.0 para aceptar conexiones de cualquier IP
    http_site = web.TCPSite(http_runner, "0.0.0.0", 8080) # Puerto 8080 para HTTP
    await http_site.start()
    print(f"--- Servidor HTTP iniciado en http://0.0.0.0:8080 ---")

    # Iniciar el servidor WebSocket (nuestra función original renombrada)
    websocket_task = asyncio.create_task(start_websocket_server())

    # Mantener esta función corriendo hasta que algo falle o se cancele
    # Podríamos esperar a que la tarea del websocket termine,
    # o simplemente esperar indefinidamente hasta una interrupción.
    try:
        # Esperar a que la tarea del WebSocket termine (o sea cancelada)
        await websocket_task
        print("Tarea del servidor WebSocket finalizada.")
    except asyncio.CancelledError:
        print("Tarea start_servers cancelada.")
        if not websocket_task.done():
            websocket_task.cancel()
            await websocket_task # Esperar cancelación
    finally:
        # Limpiar recursos de aiohttp
        print("Deteniendo servidor HTTP...")
        await http_runner.cleanup()
        print("Servidor HTTP detenido.")


if __name__ == "__main__":
    try:
        print("Iniciando la aplicación principal asyncio (HTTP + WebSocket)...")
        # --- REEMPLAZO DE LA LLAMADA ---
        # asyncio.run(main()) # <--- Línea original comentada
        asyncio.run(start_servers()) # <--- Nueva línea para iniciar ambos servidores
    except KeyboardInterrupt:
        print("\nInterrupción por teclado (Ctrl+C) detectada. Iniciando cierre limpio...")
    except Exception as e:
        print("\n--- Error Fatal en el Nivel Principal ---")
        traceback.print_exc()
    finally:
        print("--- Aplicación Finalizada ---")