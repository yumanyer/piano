import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import base64
import traceback

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
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                client = list(connected_clients)[i] 
                print(f"Error al enviar a {client.remote_address}: {result}. Eliminando cliente.")
                disconnected_clients.add(client)

        # Eliminar clientes desconectados fuera del bucle de iteración
        connected_clients.difference_update(disconnected_clients)

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
    print(f"Cámara {selected_camera_index} abierta ({frame_width}x{frame_height} @ {fps:.2f} FPS). Iniciando detección...")
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
                        print("Error: No se pudo leer el frame. Deteniendo cámara., frame is None={frame is None}")
                        break  # Salir del bucle si falla la lectura
                    
                    # 1. Preprocesamiento
                    frame = cv2.flip(frame, 1)
                    # aca converirmo de bgr a rgba
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame.flags.writeable = False

                    # 2. Detección con Mediapipe
                    results = hands.process(rgb_frame)

                    # Crear frame para dibujar (copia del frame original)
                    draw_frame = frame.copy()

                    # 3. Procesamiento de resultados, asignación de números y actualización del estado de dedos
                    processed_fingers_this_frame = set()  # Dedos detectados en ESTE frame

                    if results.multi_hand_landmarks and results.multi_handedness:

                        hand_positions = []  # Lista para almacenar posiciones de las manos
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            x_coords = [lm.x for lm in hand_landmarks.landmark]
                            hand_center = sum(x_coords) / len(x_coords)
                            hand_positions.append((hand_center, hand_landmarks, handedness.classification[0].label))
                        # Ordenar manos de izquierda a derecha según su centro
                        hand_positions.sort(key=lambda x: x[0])
                    
                        for i, (center, hand_landmarks, hand_label) in enumerate(hand_positions):
                            # Asignar hand_id: si hay dos manos, la izquierda tendrá IDs 6-10 y la derecha 1-5; si solo hay una, se asigna 1-5.
                            hand_id = 6 if (len(hand_positions) == 2 and i == 0) else 1
                    
                            # Índices para la punta, PIP y MCP de los dedos
                            finger_tips = [4, 8, 12, 16, 20]
                            finger_pip = [
                                mp_hands.HandLandmark.THUMB_IP,
                                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                                mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                                mp_hands.HandLandmark.RING_FINGER_PIP,
                                mp_hands.HandLandmark.PINKY_PIP
                            ]
                            finger_mcp = [3, 5, 9, 13, 17]
                    
                            for j in range(5):
                                h, w, _ = frame.shape
                                x_tip = int(hand_landmarks.landmark[finger_tips[j]].x * w)
                                y_tip = int(hand_landmarks.landmark[finger_tips[j]].y * h)
                                x_mcp = int(hand_landmarks.landmark[finger_mcp[j]].x * w)
                                y_mcp = int(hand_landmarks.landmark[finger_mcp[j]].y * h)
                    
                                # Dibujar la punta del dedo en rojo (círculo más grande)
                                cv2.circle(draw_frame, (x_tip, y_tip), 8, (0, 0, 255), -1)
                                # Dibujar el nudillo (MCP) en amarillo
                                cv2.circle(draw_frame, (x_mcp, y_mcp), 8, (0, 255, 255), -1)
                    
                                # Asignar número al dedo (hand_id + j) y mostrarlo en la punta
                                numero = hand_id + j
                                cv2.putText(draw_frame, str(numero), (x_tip - 10, y_tip - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                                # Actualizar el estado del dedo
                                is_down = dedo_abajo(
                                    hand_landmarks.landmark,
                                    finger_tips[j],
                                    finger_pip[j],
                                    finger_mcp[j],
                                    is_thumb=(j == 0)
                                )
                                global_finger_index = numero - 1  # Índice 0-based
                                processed_fingers_this_frame.add(global_finger_index)
                                finger_state[global_finger_index] = is_down
                    
                            # Dibujar las conexiones entre landmarks en blanco usando un estilo personalizado.
                            mp_drawing.draw_landmarks(
                                draw_frame,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                landmark_drawing_spec=None,  # No dibuja círculos adicionales en cada landmark
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                            )
                    
                    # Marcar como 'up' (False) los dedos que no fueron detectados en este frame
                    for k in range(10):
                        if k not in processed_fingers_this_frame:
                            finger_state[k] = False
                    
                    # 4. Comparar y enviar cambios de estado a los clientes conectados
                    state_changed = False
                    for k in range(10):
                        if finger_state[k] != last_sent_state[k]:
                            state_changed = True
                            event_type = "finger_down" if finger_state[k] else "finger_up"
                            await send_to_clients({"type": event_type, "finger_id": k})
                    if state_changed:
                        last_sent_state = list(finger_state)  # Actualizar último estado enviado
                    
                    # 5. Enviar el frame procesado (con dibujos) a los clientes vía WebSocket
                    if connected_clients:
                        try:
                            ret_encode, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            if ret_encode:
                                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                                await send_to_clients({"type": "video_frame", "image": jpg_as_text})
                        except Exception as e:
                            print(f"Error codificando/enviando video: {e}")
                    
                    # Ceder control para que otras tareas puedan ejecutarse
                    await asyncio.sleep(0.1)
                    
                except asyncio.CancelledError:
                    print("Bucle interno de process_camera cancelado.")
                    raise  # Re-lanzar para que el bloque finally externo se ejecute
                except Exception as e_inner:
                    print("\n--- Error en el bucle interno de process_camera ---")
                    traceback.print_exc()
                    await asyncio.sleep(1)  # Pausa tras error interno
    
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


async def handler(websocket):
    global finger_state, selected_camera_index, camera_task, connected_clients

    # Ya no se necesita 'path', así que eliminamos la referencia
    print(f"Cliente conectado: {websocket.remote_address}")
    connected_clients.add(websocket)
    initial_camera_index_received = False # Flag local para este cliente

    try:
        # Enviar estado inicial de los dedos al nuevo cliente
        initial_state_msg = {"type": "initial_state", "fingers": finger_state}
        await websocket.send(json.dumps(initial_state_msg))
        print(f"Enviado estado inicial a {websocket.remote_address}")

        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"Mensaje recibido de {websocket.remote_address}: {data}")

                if data.get('type') == 'set_camera_index' and not initial_camera_index_received:
                    new_index = int(data.get('index', 0))
                    print(f"Índice de cámara solicitado por {websocket.remote_address}: {new_index}")

                    # Si la cámara seleccionada cambia O si la tarea no está corriendo
                    if new_index != selected_camera_index or camera_task is None or camera_task.done():
                        selected_camera_index = new_index
                        initial_camera_index_received = True # Marcar como recibido para este cliente

                        # Cancelar tarea anterior si existe y está corriendo
                        if camera_task and not camera_task.done():
                            print("Índice de cámara cambiado o reinicio solicitado. Cancelando tarea anterior...")
                            camera_task.cancel()
                            try:
                                await camera_task # Esperar a que se cancele
                            except asyncio.CancelledError:
                                print("Tarea de cámara anterior cancelada.")
                            camera_task = None # Resetear

                        # Iniciar nueva tarea de cámara (solo si hay clientes)
                        if connected_clients:
                            print(f"Iniciando tarea process_camera con índice {selected_camera_index}...")
                            camera_task = asyncio.create_task(process_camera())
                        else:
                             print("No hay clientes conectados, no se inicia la cámara.")

                    else:
                         print(f"Índice de cámara ({new_index}) ya está en uso y la tarea está corriendo.")
                         # Podríamos reenviar el estado inicial por si acaso
                         await websocket.send(json.dumps({"type": "initial_state", "fingers": finger_state}))


                else:
                     print(f"Mensaje no manejado o índice ya procesado: {data}")

            except json.JSONDecodeError:
                print(f"Mensaje no JSON recibido de {websocket.remote_address}: {message}")
            except ValueError:
                 print(f"Error convirtiendo índice de cámara a entero: {data.get('index')}")
            except Exception as e:
                print(f"Error procesando mensaje de {websocket.remote_address}: {e}")
                traceback.print_exc()

    except websockets.exceptions.ConnectionClosedOK:
        print(f"Cliente desconectado correctamente: {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"Conexión de cliente cerrada con error: {websocket.remote_address} - {e}")
    except Exception as e:
        print(f"Error inesperado en handler para {websocket.remote_address}:")
        traceback.print_exc()
    finally:
        print(f"Eliminando cliente: {websocket.remote_address}")
        connected_clients.remove(websocket)
        # Si era el último cliente, cancelar la tarea de cámara
        if not connected_clients and camera_task and not camera_task.done():
             print("Último cliente desconectado, cancelando tarea de cámara...")
             camera_task.cancel()
             # No necesitamos esperar aquí necesariamente, pero podemos resetear
             # camera_task = None



async def main():
    global camera_task # Indicar que usamos la global
    host = "127.0.0.1" # Usar IP explícita puede ser más robusto
    port = 8765

    print(f"--- Iniciando Servidor WebSocket en ws://{host}:{port} ---")
    print(f"--- Esperando conexión de cliente para seleccionar cámara ---")

    # Crear y iniciar el servidor
    server = await websockets.serve(handler, host, port, max_size=2*1024*1024) # Aumentar max_size por si acaso

    print("Servidor WebSocket iniciado. Esperando conexiones...")

    try:
        # Mantener el servidor corriendo hasta que se cierre
        await server.wait_closed()
    except asyncio.CancelledError:
         print("Servidor principal cancelado.")
    finally:
        print("Cerrando servidor WebSocket...")
        # Cancelar la tarea de cámara si sigue corriendo al cerrar el servidor
        # Ahora 'camera_task' siempre existe (puede ser None)
        if camera_task and not camera_task.done():
            print("Cancelando tarea de cámara pendiente...")
            camera_task.cancel()
            try:
                await camera_task # Esperar a que la cancelación termine
            except asyncio.CancelledError:
                print("Tarea de cámara cancelada al cerrar.")
        print("Servidor WebSocket cerrado.")


if __name__ == "__main__":
    try:
        print("Iniciando la aplicación principal asyncio...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupción por teclado (Ctrl+C) detectada. Iniciando cierre...")
    except Exception as e:
        print("\n--- Error Fatal en el Nivel Principal ---")
        traceback.print_exc()
    finally:
        print("--- Aplicación Finalizada ---")