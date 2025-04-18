// --- Elementos del DOM ---
const modal = document.getElementById('infoModal');
const showModalBtn = document.getElementById('showModal');
const closeModalBtn = document.getElementById('closeModal');
const confirmModalBtn = document.getElementById('confirmModal');
const mainContainer = document.querySelector('.container');
const mainPanel = document.getElementById('mainPanel');
const circleButton = document.getElementById('fullscreenBtn');
const videoElement = document.getElementById('videoFeed'); // <video>
const canvasElement = document.getElementById('outputCanvas'); // <canvas>
const canvasCtx = canvasElement.getContext('2d');
const connectionStatusElement = document.getElementById('connectionStatus');
const fileInputs = document.querySelectorAll('.custom-input');
const saveButton = document.getElementById('saveButton');
const playButton = document.getElementById('playButton');

// --- Elementos para Selección de Cámara ---
const cameraSetupDiv = document.getElementById('cameraSetup');
const requestCameraBtn = document.getElementById('requestCameraButton');
const cameraSelectList = document.getElementById('cameraSelect');
const confirmCameraBtn = document.getElementById('confirmCameraButton');
const cameraStatusMsg = document.getElementById('cameraStatusMessage');

// --- Configuración WebSocket ---
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const wsHost = window.location.host;
const WS_URL = `${wsProtocol}//${wsHost}/ws`;
console.log(`Connecting WebSocket to: ${WS_URL}`);
let socket;

// --- Configuración Web Audio API ---
let audioContext;
const soundBuffers = new Map();
const defaultSoundFiles = [ // Rutas relativas a la raíz del servidor ahora
    '/sounds/DO.wav', '/sounds/RE.wav', '/sounds/MI.wav', '/sounds/FA.wav', '/sounds/SOL.wav',
    '/sounds/LA.wav', '/sounds/SI.wav', '/sounds/DO%23.wav', '/sounds/RE%23.wav', '/sounds/FA%23.wav'
];
let userSoundURLs = {};
defaultSoundFiles.forEach((url, index) => { userSoundURLs[index] = url; });
let soundsLoadedCount = 0;
let totalSoundsToLoad = defaultSoundFiles.length;
let audioContextResumed = false;

// --- Estado MediaPipe y Finger State ---
let hands = null; // Instancia de MediaPipe Hands
let currentFingerState = Array(10).fill(false);
let lastFingerState = Array(10).fill(false);
let selectedDeviceId = null;
let videoStream = null;
let animationFrameId = null; // Para controlar el bucle de animación

// --- Constantes para MediaPipe ---
// Asegúrate que los landmarks de MediaPipe JS coincidan con los usados en Python
// Puedes encontrarlos en la documentación de MediaPipe JS o inspeccionando el objeto 'results'
const MP_HAND_LANDMARKS = {
    THUMB_TIP: 4, THUMB_IP: 3, THUMB_MCP: 1, // Revisar estos índices si fallan
    INDEX_FINGER_TIP: 8, INDEX_FINGER_PIP: 6, INDEX_FINGER_MCP: 5,
    MIDDLE_FINGER_TIP: 12, MIDDLE_FINGER_PIP: 10, MIDDLE_FINGER_MCP: 9,
    RING_FINGER_TIP: 16, RING_FINGER_PIP: 14, RING_FINGER_MCP: 13,
    PINKY_TIP: 20, PINKY_PIP: 18, PINKY_MCP: 17
};

// --- Inicialización ---
window.addEventListener('load', () => {
    console.log('Página cargada.');
    loadUserSounds();
    initializeMediaPipeHands(); // Inicializar instancia de Hands
    if (cameraSetupDiv) cameraSetupDiv.style.display = 'flex';
    if (mainContainer) mainContainer.style.display = 'none';
    addNumberClickListeners();
    setupWebSocket(); // Conectar WebSocket al cargar la página
});

// --- Inicializar MediaPipe Hands ---
function initializeMediaPipeHands() {
    // Verificar si las variables globales de MediaPipe (Hands, HAND_CONNECTIONS) existen
    if (typeof Hands === 'undefined' || typeof HAND_CONNECTIONS === 'undefined') {
        console.error("MediaPipe Hands no cargado. Verifica los scripts CDN en el HTML.");
        // Mostrar error al usuario
        if(cameraStatusMsg) cameraStatusMsg.textContent = "Error: No se pudo cargar MediaPipe. Recarga la página.";
        if(requestCameraBtn) requestCameraBtn.disabled = true;
        return;
    }

    hands = new Hands({locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }});

    hands.setOptions({
      maxNumHands: 2,
      modelComplexity: 0, // 0 es el más rápido
      minDetectionConfidence: 0.6,
      minTrackingConfidence: 0.6
    });

    hands.onResults(onResults); // Asignar el callback
    console.log('MediaPipe Hands inicializado.');
}

// --- Listener para cargar sonidos personalizados ---
fileInputs.forEach(input => {
    input.addEventListener('change', async function(event) {
        const file = event.target.files[0];
        const key = parseInt(event.target.dataset.key);
        const parentItem = this.closest('.file-item');
        let label = parentItem ? parentItem.querySelector('.file-name') : null;
        let indicator = parentItem ? parentItem.querySelector('.custom-indicator') : null;

        if (label) label.remove();
        if (indicator) indicator.remove(); // Remover indicador anterior si existe

        if (file && audioContext) {
            const fileName = file.name;
            const fileURL = URL.createObjectURL(file);
            console.log(`Archivo seleccionado para tecla ${key + 1}: ${fileName}`);

            label = document.createElement('span');
            label.className = 'file-name';
            label.textContent = ` (${fileName})`;
            this.parentNode.insertBefore(label, this.nextSibling);

            try {
                userSoundURLs[key] = fileURL; // Guardar URL temporalmente
                const arrayBuffer = await file.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                soundBuffers.set(key, audioBuffer);
                console.log(`Sonido personalizado para tecla ${key + 1} cargado.`);
                // No necesitamos forzar recarga aquí, se hizo al seleccionar
            } catch (error) {
                console.error(`Error procesando archivo de audio para tecla ${key + 1}:`, error);
                alert(`Error al cargar sonido: ${error.message}. Restaurando default.`);
                userSoundURLs[key] = defaultSoundFiles[key]; // Restaurar URL default
                if (label) label.remove();
                // Cargar el buffer por defecto si falla el personalizado
                loadSound(defaultSoundFiles[key], key, () => {}, () => {}, true); // Forzar recarga del default
            }
        } else if (!audioContext) {
            alert("Audio no inicializado. Haz clic en la página e intenta de nuevo.");
            if (label) label.remove();
            event.target.value = null;
        } else { // No se seleccionó archivo (o se canceló)
            const defaultURL = defaultSoundFiles[key];
            if (userSoundURLs[key] !== defaultURL) { // Si había un custom antes
                 userSoundURLs[key] = defaultURL; // Restaurar URL default
                 console.log(`Restaurado sonido por defecto para tecla ${key + 1}`);
                 // Recargar buffer por defecto
                 loadSound(defaultURL, key, () => {
                      console.log(`Buffer por defecto para tecla ${key+1} recargado.`);
                 }, () => {}, true); // Forzar recarga del default
            }
            if (label) label.remove();
        }
    });
});

// --- Función para guardar sonidos ---
function saveUserSounds() {
    try {
        // Crear objeto solo con URLs no-blob para guardar
        const urlsToSave = {};
        Object.keys(userSoundURLs).forEach(key => {
            if (!userSoundURLs[key].startsWith('blob:')) {
                urlsToSave[key] = userSoundURLs[key];
            } else {
                // Opcional: podrías guardar la URL default si era un blob
                urlsToSave[key] = defaultSoundFiles[key];
            }
        });
        localStorage.setItem("userSoundSettings", JSON.stringify(urlsToSave)); // Guardar solo no-blobs
        alert("Configuración de sonidos guardada.");
    } catch (error) {
        console.error("Error guardando sonidos en localStorage:", error);
        alert("Error al guardar configuración.");
    }
}

// --- Función para cargar sonidos ---
function loadUserSounds() {
    const savedSettings = localStorage.getItem("userSoundSettings");
    // Reiniciar userSoundURLs con defaults primero
    userSoundURLs = {};
    defaultSoundFiles.forEach((url, index) => { userSoundURLs[index] = url; });

    if (savedSettings) {
        try {
            const parsedSettings = JSON.parse(savedSettings);
            console.log("Configuración de sonidos cargada.");
            // Sobrescribir defaults con los guardados
            Object.keys(parsedSettings).forEach(keyIndexStr => {
                const keyIndex = parseInt(keyIndexStr);
                if (userSoundURLs.hasOwnProperty(keyIndex)) { // Asegurarse que el índice existe
                    userSoundURLs[keyIndex] = parsedSettings[keyIndexStr];
                }
            });
        } catch (error) {
            console.error("Error cargando sonidos desde localStorage:", error);
            localStorage.removeItem("userSoundSettings"); // Limpiar si está corrupto
        }
    } else {
         console.log("No se encontró configuración guardada. Usando defaults.");
    }

    // Actualizar UI para reflejar si se cargó algo diferente al default (VISUAL)
    fileInputs.forEach(input => {
        const key = parseInt(input.dataset.key);
        const parentItem = input.closest('.file-item');
        let label = parentItem ? parentItem.querySelector('.file-name') : null;
        let indicator = parentItem ? parentItem.querySelector('.custom-indicator') : null;
        if (label) label.remove();
        if (indicator) indicator.remove();

        if (userSoundURLs[key] !== defaultSoundFiles[key]) {
             // Si la URL cargada no es la default (asumimos es una personalizada guardada)
             // Opcional: Mostrar nombre o indicación
             indicator = document.createElement('span');
             indicator.className = 'custom-indicator';
             indicator.textContent = ' (custom)';
             indicator.style.fontSize = '0.8em';
             indicator.style.marginLeft = '5px';
             input.parentNode.insertBefore(indicator, input.nextSibling);
        }
    });
}

// --- Función para añadir listeners a los números ---
function addNumberClickListeners() {
    document.querySelectorAll(".file-item-num").forEach(item => {
        const key = parseInt(item.dataset.key);
        if (!isNaN(key)) {
            item.addEventListener("click", function () {
                console.log(`Clic en número ${key + 1}, intentando reproducir sonido...`);
                if (!audioContextResumed) {
                    console.warn("AudioContext no activo. Intentando reanudar...");
                    resumeAudioContext(); // Intenta activar el audio
                }
                playSound(key);
            });
        } else {
            console.warn("Elemento .file-item-num sin data-key válido:", item);
        }
    });
}


// --- Event Listener para el botón de solicitar cámara ---
if (requestCameraBtn) {
    requestCameraBtn.addEventListener('click', async () => {
        if (cameraStatusMsg) cameraStatusMsg.textContent = 'Solicitando permiso y buscando cámaras...';
        await listCameras();
    });
}

// --- Función para listar cámaras ---
async function listCameras() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
        if (cameraStatusMsg) cameraStatusMsg.textContent = "Error: enumerateDevices() no soportado.";
        return;
    }
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        if (videoDevices.length === 0) {
             if (cameraStatusMsg) cameraStatusMsg.textContent = 'No se encontraron cámaras.';
             return;
        }

        // Permiso necesario para nombres?
        let needsPermission = videoDevices.some(device => !device.label);
        if (needsPermission) {
             if (cameraStatusMsg) cameraStatusMsg.textContent = 'Permiso necesario para nombres...';
             try {
                 const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                 tempStream.getTracks().forEach(track => track.stop());
                 // Volver a listar
                 await listCameras();
                 return;
             } catch (permError) {
                 console.error("Error al obtener permiso getUserMedia:", permError);
                 if (cameraStatusMsg) cameraStatusMsg.textContent = `Error permiso: ${permError.name}.`;
                 if(requestCameraBtn) requestCameraBtn.style.display = 'inline-block';
                 if(cameraSelectList) cameraSelectList.style.display = 'none';
                 if(confirmCameraBtn) confirmCameraBtn.style.display = 'none';
                 return;
             }
        }

        // Llenar selector
        if (cameraSelectList) {
            cameraSelectList.innerHTML = '';
            videoDevices.forEach((device) => { // Ya no usamos índice para OpenCV
                const option = document.createElement('option');
                option.value = device.deviceId; // USAR deviceId como VALOR
                option.text = device.label || `Cámara ${cameraSelectList.options.length + 1}`;
                option.dataset.deviceId = device.deviceId; // Redundante pero OK
                cameraSelectList.appendChild(option);
            });
            cameraSelectList.style.display = 'inline-block';
            if (confirmCameraBtn) confirmCameraBtn.style.display = 'inline-block';
            if (requestCameraBtn) requestCameraBtn.style.display = 'none';
            if (cameraStatusMsg) cameraStatusMsg.textContent = 'Selecciona una cámara:';
        }

    } catch (error) {
        console.error('Error listando cámaras:', error);
        let message = `Error dispositivos: ${error.name || error.message}`;
        if (error.name === 'NotAllowedError') message = 'Permiso denegado.';
        else if (error.name === 'NotFoundError') message = 'No se encontró cámara.';
         if (cameraStatusMsg) { cameraStatusMsg.textContent = message; cameraStatusMsg.style.color = 'red'; }
         if(requestCameraBtn) requestCameraBtn.style.display = 'inline-block';
         if(cameraSelectList) cameraSelectList.style.display = 'none';
         if(confirmCameraBtn) confirmCameraBtn.style.display = 'none';
    }
}



// ---  Event Listener Confirmar Cámara ---
if (confirmCameraBtn) {
    confirmCameraBtn.addEventListener('click', async () => {
        const selectedOption = cameraSelectList.options[cameraSelectList.selectedIndex];
        if (!selectedOption || !selectedOption.value) {
            if (cameraStatusMsg) cameraStatusMsg.textContent = 'Selecciona una cámara.';
            return;
        }

        selectedDeviceId = selectedOption.value; // Obtener deviceId del valor del option
        const selectedCameraLabel = selectedOption.text;

        if (cameraStatusMsg) cameraStatusMsg.textContent = `Iniciando cámara: ${selectedCameraLabel}...`;
        confirmCameraBtn.disabled = true;
        requestCameraBtn.disabled = true; // Deshabilitar ambos mientras inicia

        try {
            // Detener stream anterior si existe
            stopMediaPipeProcessing(); // Detiene bucle y cámara

            // Obtener el stream de video del usuario
            videoStream = await navigator.mediaDevices.getUserMedia({
                video: { deviceId: { exact: selectedDeviceId }, width: 640, height: 480 } // Pedir resolución
            });

            videoElement.srcObject = videoStream;
            videoElement.onloadedmetadata = () => {
                canvasElement.width = videoElement.videoWidth;
                canvasElement.height = videoElement.videoHeight;
                console.log(`Stream cámara iniciado (${videoElement.videoWidth}x${videoElement.videoHeight}).`);

                // Ocultar setup, mostrar UI
                if (cameraSetupDiv) cameraSetupDiv.style.display = 'none';
                if (mainContainer) mainContainer.style.display = 'flex';
                if (connectionStatusElement) updateConnectionStatusBasedOnAudio(); // Actualiza estado general

                // Iniciar Audio y Sonidos (si no se hizo antes)
                initializeAudio();

                // Conectar WebSocket si no está conectado
                setupWebSocket();

                // Iniciar el bucle de procesamiento de MediaPipe
                startMediaPipeProcessing();

                confirmCameraBtn.disabled = false; // Habilitar botones de nuevo
                requestCameraBtn.disabled = false;
            };
            videoElement.onerror = (err) => {
                console.error("Error en el elemento video:", err);
                if (cameraStatusMsg) cameraStatusMsg.textContent = `Error interno del video.`;
                confirmCameraBtn.disabled = false;
                requestCameraBtn.disabled = false;
                 resetToCameraSelection(`Error al mostrar video de '${selectedCameraLabel}'.`);
            }

        } catch (error) {
            console.error(`Error al iniciar getUserMedia (${selectedCameraLabel}):`, error);
            if (cameraStatusMsg) cameraStatusMsg.textContent = `Error al iniciar '${selectedCameraLabel}': ${error.name}.`;
            confirmCameraBtn.disabled = false;
            requestCameraBtn.disabled = false;
            resetToCameraSelection(null); // Resetea UI sin mensaje de error específico
        }
    });
}

// --- Bucle de procesamiento MediaPipe ---
async function startMediaPipeProcessing() {
    if (!hands || !videoElement.srcObject || !videoStream?.active) {
        console.warn("Intento de iniciar MediaPipe sin cámara activa o Hands no listo.");
        stopMediaPipeProcessing(); // Asegurarse que esté detenido
        return;
    }
    console.log("Iniciando bucle MediaPipe...");
    if(connectionStatusElement) connectionStatusElement.textContent = "Procesando cámara..."; // Indicar que está activo

    async function processFrame() {
        // Si el stream ya no está activo, detener el bucle
        if (!videoStream || !videoStream.active) {
            console.log("Stream de video detenido, parando bucle MediaPipe.");
            stopMediaPipeProcessing();
            return;
        }
        // Si el video está listo para ser procesado
        if (videoElement.readyState >= 2) {
            try {
                // Enviar frame a MediaPipe
                await hands.send({image: videoElement});
            } catch (error) {
                console.error("Error en hands.send:", error);
                stopMediaPipeProcessing(); // Detener si hay error
                resetToCameraSelection("Error procesando video con MediaPipe.");
                return; // Salir del bucle
            }
        }
        // Solicitar el siguiente frame si el stream sigue activo
        if (videoStream?.active) {
            animationFrameId = requestAnimationFrame(processFrame);
        } else {
             stopMediaPipeProcessing(); // Detener si stream se inactivó
        }
    }
    // Cancelar frame anterior por si acaso y empezar de nuevo
    if (animationFrameId) cancelAnimationFrame(animationFrameId);
    animationFrameId = requestAnimationFrame(processFrame);
}

// --- Callback de Resultados MediaPipe ---
function onResults(results) {
    // Limpiar canvas
    canvasCtx.save(); // Guardar estado actual del contexto (importante)
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

    // NO dibujar la imagen de video aquí si el <video> está visible.
    // Si el <video> está oculto (visibility: hidden), sí necesitas dibujarlo:
    // canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    // Reiniciar estado de dedos para este frame
    currentFingerState.fill(false);
    const processedFingersThisFrame = new Set();

    if (results.multiHandLandmarks && results.multiHandedness) {
        const handData = [];
        for (let i = 0; i < results.multiHandLandmarks.length; i++) {
            const landmarks = results.multiHandLandmarks[i];
            const centerX = landmarks.reduce((sum, lm) => sum + lm.x, 0) / landmarks.length;
            handData.push({ centerX, landmarks, handedness: results.multiHandedness[i].label });
        }
        handData.sort((a, b) => a.centerX - b.centerX); // Ordenar por posición X (izquierda a derecha en imagen)

        for (let i = 0; i < handData.length; i++) {
            const { landmarks } = handData[i];

             // *** CORRECCIÓN ASIGNACIÓN DE MANO ***
             // Izquierda en imagen (i=0) es MANO DERECHA del usuario (1-5)
             // Derecha en imagen (i=1) es MANO IZQUIERDA del usuario (6-10)
             const handednessLabel = handData[i].handedness; // "Right" o "Left"
             const handIdStart = (handednessLabel === "Left") ? 1 : 6;
             // ************************************

            // --- Dibujar conexiones y landmarks ---
            // Espejar el CANVAS TEMPORALMENTE para que el dibujo coincida con el video espejado
            canvasCtx.save(); // Guardar estado antes de transformar
            canvasCtx.translate(canvasElement.width, 0); // Mover origen a la derecha
            canvasCtx.scale(-1, 1); // Escalar inversamente en X (espejar)

            if (typeof drawConnectors !== 'undefined' && typeof HAND_CONNECTIONS !== 'undefined') {
                // Dibujar conexiones con coordenadas normales (la transformación del canvas hace el resto)
                drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS,
                               {color: '#FFFFFF', lineWidth: 2});

                // Dibujar landmarks individuales (opcional, elige un color)
                 drawLandmarks(canvasCtx, landmarks,
                                {color: '#FF0000', // Puntos rojos
                                 fillColor: '#FF0000',
                                 lineWidth: 1,
                                 radius: 4}); // Ajusta tamaño
                 // Dibujar nudillos PIP con otro color
                 const pipIndices = [MP_HAND_LANDMARKS.THUMB_IP, MP_HAND_LANDMARKS.INDEX_FINGER_PIP, MP_HAND_LANDMARKS.MIDDLE_FINGER_PIP, MP_HAND_LANDMARKS.RING_FINGER_PIP, MP_HAND_LANDMARKS.PINKY_PIP];
                 const pipLandmarks = pipIndices.map(index => landmarks[index]).filter(Boolean); // Obtener landmarks PIP existentes
                 drawLandmarks(canvasCtx, pipLandmarks,
                                 {color: '#FFFF00', // Puntos amarillos
                                 fillColor: '#FFFF00',
                                  lineWidth: 1,
                                  radius: 4});


            } else { console.warn("drawConnectors o HAND_CONNECTIONS no definidos."); }

            // --- Lógica y dibujo de números (DENTRO del canvas espejado) ---
            const fingerTips = [MP_HAND_LANDMARKS.THUMB_TIP, MP_HAND_LANDMARKS.INDEX_FINGER_TIP, MP_HAND_LANDMARKS.MIDDLE_FINGER_TIP, MP_HAND_LANDMARKS.RING_FINGER_TIP, MP_HAND_LANDMARKS.PINKY_TIP];
            const fingerPip = [MP_HAND_LANDMARKS.THUMB_IP, MP_HAND_LANDMARKS.INDEX_FINGER_PIP, MP_HAND_LANDMARKS.MIDDLE_FINGER_PIP, MP_HAND_LANDMARKS.RING_FINGER_PIP, MP_HAND_LANDMARKS.PINKY_PIP];

            for (let j = 0; j < 5; j++) {
                const tipLandmark = landmarks[fingerTips[j]];
                const pipLandmark = landmarks[fingerPip[j]];

                if (tipLandmark && pipLandmark) {
                    // Calcular coordenadas en el canvas (NORMALES, antes de la transformación)
                    // La transformación del canvas se encarga de mapearlas visualmente.
                    const xTip = tipLandmark.x * canvasElement.width;
                    const yTip = tipLandmark.y * canvasElement.height;

                    // Dibujar número (también se dibujará espejado por la transformación del canvas)
                    const numeroDedo = handIdStart + j;
                    canvasCtx.fillStyle = 'white';
                    canvasCtx.font = 'bold 18px Arial';
                    // Ajustar la alineación del texto para que quede bien al espejarse
                    canvasCtx.textAlign = 'center'; // Centrar texto horizontalmente
                    canvasCtx.fillText(numeroDedo.toString(), xTip, yTip - 15); // Poner encima de la punta

                    // Lógica dedo abajo (no cambia)
                    const isDown = dedoAbajoJS(landmarks, fingerTips[j], fingerPip[j], j === 0);
                    const globalFingerIndex = numeroDedo - 1;

                    if (globalFingerIndex >= 0 && globalFingerIndex < 10) {
                        processedFingersThisFrame.add(globalFingerIndex);
                        currentFingerState[globalFingerIndex] = isDown;
                    }
                }
            }
            // --- Fin lógica y dibujo números ---

            canvasCtx.restore(); // *** IMPORTANTE: Restaurar estado del canvas (quita el espejado) ***
        }
    }

    // Marcar como 'up' dedos no procesados (sin cambios)
    for (let k = 0; k < 10; k++) { if (!processedFingersThisFrame.has(k)) { currentFingerState[k] = false; } }

    // Comparar estado y enviar eventos / reproducir sonido / actualizar UI (sin cambios)
    for (let k = 0; k < 10; k++) {
        if (currentFingerState[k] !== lastFingerState[k]) {
            const eventType = currentFingerState[k] ? "down" : "up";
            updateFingerUI(k, currentFingerState[k]);
            if (currentFingerState[k]) { playSound(k); }
            if (socket && socket.readyState === WebSocket.OPEN) {
                try { socket.send(JSON.stringify({ type: 'finger_event', finger_id: k, state: eventType })); }
                catch (e) { console.error("Error enviando WS:", e); }
            }
        }
    }
    lastFingerState = [...currentFingerState]; // Actualizar estado (sin cambios)
    // canvasCtx.restore(); // Ya no es necesario un restore global aquí
}
// --- Lógica dedoAbajo en JS ---
function dedoAbajoJS(landmarks, fingerTipIdx, fingerPipIdx, isThumb = false) { // fingerPipIdx se sigue pasando pero se ignorará para el pulgar
    try {
        const fingerTip = landmarks[fingerTipIdx];

        // *** MODIFICADO: Seleccionar el punto de comparación ***
        // Si es pulgar (isThumb), usa THUMB_MCP (índice 1).
        // Si no, usa el índice PIP/IP pasado (fingerPipIdx).
        const compareIndex = isThumb ? MP_HAND_LANDMARKS.THUMB_IP : fingerPipIdx;
        const compareLandmark = landmarks[compareIndex];
        // ****************************************************

        if (!fingerTip || !compareLandmark) {
            // Puedes añadir un log si quieres ver cuándo falta un landmark
            // console.warn(` dedoAbajoJS: Landmark faltante - Tip: ${fingerTipIdx}, Compare: ${compareIndex}`);
            return false; // Landmark no detectado
        }

        // Dedo abajo si Y de la punta es MAYOR (más abajo en pantalla) que Y del punto de comparación
        return fingerTip.y > compareLandmark.y;
    } catch (error) {
        console.error("Error en dedoAbajoJS:", error);
        return false;
    }
}

// --- Funciones WebSocket (Simplificadas) ---
function setupWebSocket() {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        console.log("WebSocket ya está conectado o conectándose.");
        // Si ya está conectado, actualizar estado visual
         if (connectionStatusElement && socket.readyState === WebSocket.OPEN) {
             updateConnectionStatusBasedOnAudio();
         }
        return;
    }
    console.log(`Intentando conectar a ${WS_URL}...`);
    if (connectionStatusElement) {
        connectionStatusElement.textContent = 'Conectando al servidor...';
        connectionStatusElement.style.color = 'orange';
    }
    socket = new WebSocket(WS_URL);

    socket.onopen = () => {
        console.log('WebSocket conectado.');
        if (connectionStatusElement) {
             updateConnectionStatusBasedOnAudio(); // Actualizar estado
        }
    };
    socket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        // No resetear a cámara aquí, podría ser error temporal
        if (connectionStatusElement) {
            connectionStatusElement.textContent = 'Error WebSocket';
            connectionStatusElement.style.color = 'red';
        }
        socket = null; // Permitir reintentar conexión?
    };
    socket.onclose = (event) => {
        console.log('WebSocket desconectado:', event.code, event.reason);
        if (connectionStatusElement) {
            connectionStatusElement.textContent = `Desconectado (${event.code})`;
            connectionStatusElement.style.color = 'red';
        }
        // No resetear a cámara automáticamente, solo indicar desconexión
        socket = null;
    };
    socket.onmessage = handleWebSocketMessage;
}

// --- Manejador de Mensajes WebSocket (Simplificado) ---
function handleWebSocketMessage(event) {
    try {
        const data = JSON.parse(event.data);
        console.log("Mensaje Backend:", data);

        // Solo procesar mensajes que NO sean generados localmente
        if (data.type === 'status') {
             // Mostrar estado del servidor si es relevante
             if (connectionStatusElement && !connectionStatusElement.textContent.includes('Cargando')) {
                 connectionStatusElement.textContent += ` | Servidor: ${data.message}`;
             }
        } else if (data.type === 'error') { // Errores generales del backend
             console.error("Error Backend:", data.message);
             if (connectionStatusElement) {
                 connectionStatusElement.textContent = `Error Servidor: ${data.message}`;
                 connectionStatusElement.style.color = 'red';
             }
        }
        // Ignorar otros tipos (finger_event, video_frame, etc.)

    } catch (e) {
        console.error("Error procesando mensaje WebSocket:", e, event.data);
    }
}

// --- Funciones Web Audio API (Sin cambios) ---
function initializeAudio() {
    if (!audioContext) {
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log("AudioContext creado:", audioContext.state);
            if (audioContext.state === 'suspended') { resumeAudioContext(); } // Intentar reanudar si inicia suspendido
            else { audioContextResumed = true; preloadSounds(); }
        } catch (e) {
            console.error("Error AudioContext:", e);
            if(connectionStatusElement) {connectionStatusElement.textContent += " (Error Audio)"; connectionStatusElement.style.color = 'red';}
            fileInputs.forEach(input => input.disabled = true);
            alert("Error al inicializar audio.");
        }
    } else if (audioContext.state === 'suspended') { resumeAudioContext(); }
    else if (!audioContextResumed && soundsLoadedCount === 0) { preloadSounds(); } // Si ya está corriendo pero no se cargaron
}
function resumeAudioContext() {
    if (!audioContext || audioContext.state !== 'suspended') return;
    audioContext.resume().then(() => {
        console.log("AudioContext reanudado.");
        audioContextResumed = true;
        if (soundsLoadedCount === 0) preloadSounds(); else updateConnectionStatusBasedOnAudio();
    }).catch(e => console.error("Error reanudando AudioContext:", e));
}
function preloadSounds() {
    if (!audioContext || !audioContextResumed) return;
    if (soundsLoadedCount >= totalSoundsToLoad && totalSoundsToLoad > 0) { console.log("Sonidos ya precargados."); updateConnectionStatusBasedOnAudio(); return; }
    console.log("Precargando sonidos...");
    updateConnectionStatusBasedOnAudio();

    soundsLoadedCount = 0;
    totalSoundsToLoad = Object.keys(userSoundURLs).length;
    let soundsSuccessfullyLoaded = 0;
    let soundsFailedToLoad = 0;

    if (totalSoundsToLoad === 0) { updateConnectionStatusBasedOnAudio(); return; }

    const promises = Object.keys(userSoundURLs).map(keyIndexStr => {
        const keyIndex = parseInt(keyIndexStr);
        const url = userSoundURLs[keyIndex];
        return new Promise((resolve, reject) => {
            loadSound(url, keyIndex, resolve, reject, false); // Usar promesas en loadSound
        });
    });

    Promise.allSettled(promises).then(results => {
         results.forEach(result => {
             if (result.status === 'fulfilled') soundsSuccessfullyLoaded++;
             else soundsFailedToLoad++;
         });
         soundsLoadedCount = soundsSuccessfullyLoaded; // Actualizar contador global
         console.log(`Precarga finalizada. Éxitos: ${soundsSuccessfullyLoaded}, Fallos: ${soundsFailedToLoad}`);
         updateConnectionStatusBasedOnAudio();
    });
}
function loadSound(url, index, successCallback, errorCallback, forceReload = false) {
    // Convertir a promesa para Promise.allSettled
    return new Promise((resolve, reject) => {
        if (!audioContext) return reject(new Error("AudioContext no disponible"));
        if (!forceReload && soundBuffers.has(index)) {
            console.log(`Sonido ${index} ya en buffer.`);
            return resolve(); // Resuelve si ya está cargado y no se fuerza
        }
        fetch(url)
            .then(response => {
                if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
                return response.arrayBuffer();
            })
            .then(arrayBuffer => audioContext.decodeAudioData(arrayBuffer))
            .then(audioBuffer => {
                soundBuffers.set(index, audioBuffer);
                console.log(`Sonido ${index} cargado (${url.split('/').pop()})`);
                if(successCallback) successCallback(); // Llamar callback original si existe
                resolve(); // Resuelve la promesa
            })
            .catch(error => {
                console.error(`Error cargando ${index} (${url}):`, error);
                soundBuffers.delete(index);
                if(errorCallback) errorCallback(error); // Llamar callback original si existe
                reject(error); // Rechaza la promesa
            });
    });
}
function playSound(index) {
    if (!audioContextResumed) { resumeAudioContext(); return; } // Intentar reanudar si no está activo
    const buffer = soundBuffers.get(index);
    if (buffer) {
        try {
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.start(0);
        } catch (e) { console.error(`Error reproduciendo ${index}:`, e); }
    } else { console.warn(`Buffer no encontrado: ${index}.`); }
}

// --- Funciones UI (sin cambios funcionales mayores) ---
function openModal() { if (modal) modal.style.display = 'flex'; }
function closeModal() { if (modal) modal.style.display = 'none'; }
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        (mainPanel || document.documentElement).requestFullscreen().catch(err => console.error(`Error fullscreen: ${err.message}`, err));
    } else if (document.exitFullscreen) { document.exitFullscreen(); }
}
document.addEventListener('fullscreenchange', () => {
    const isFullscreen = !!document.fullscreenElement;
    if (mainPanel) mainPanel.classList.toggle('fullscreen', isFullscreen);
    if (circleButton) circleButton.textContent = isFullscreen ? '✕' : '⛶';
});
function updateFingerUI(index, isDown) {
    const listItem = document.getElementById(`finger-item-${index}`);
    if (listItem) {
        const indicator = listItem.querySelector('.finger-status-indicator');
        if (indicator) indicator.style.backgroundColor = isDown ? 'lime' : 'transparent';
    }
}
function updateConnectionStatusBasedOnAudio() {
    if (!connectionStatusElement) return;
    let statusText = "";
    let statusColor = "grey";

    if (socket && socket.readyState === WebSocket.OPEN) {
        statusText = "Conectado";
        statusColor = "lime";
        if (!audioContext) { statusText += " | Audio Error"; statusColor = 'red'; }
        else if (!audioContextResumed) { statusText += " | Clic para activar audio"; statusColor = 'yellow'; }
        else if (soundsLoadedCount < totalSoundsToLoad) {
            let failed = totalSoundsToLoad - soundsLoadedCount;
            statusText += ` | Cargando sonidos (${soundsLoadedCount}/${totalSoundsToLoad})`;
            if(failed > 0) statusText += ` (${failed} fallaron)`;
            statusColor = 'orange';
        } else { statusText += " | Sonidos Listos"; }
    } else if (socket && socket.readyState === WebSocket.CONNECTING) {
        statusText = "Conectando...";
        statusColor = "orange";
    } else {
        statusText = "Desconectado";
        statusColor = "red";
    }
    connectionStatusElement.textContent = statusText;
    connectionStatusElement.style.color = statusColor;
}
function resetToCameraSelection(message) {
    stopMediaPipeProcessing();
    if (socket && socket.readyState === WebSocket.OPEN) { socket.close(); }
    socket = null;
    if (videoStream) { videoStream.getTracks().forEach(track => track.stop()); videoStream = null; }
    if (videoElement) videoElement.srcObject = null;
    if (canvasCtx) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    currentFingerState.fill(false); lastFingerState.fill(false);
    for (let i = 0; i < 10; i++) { updateFingerUI(i, false); }
    if (mainContainer) mainContainer.style.display = 'none';
    if (cameraSetupDiv) cameraSetupDiv.style.display = 'flex';
    if (requestCameraBtn) { requestCameraBtn.style.display = 'inline-block'; requestCameraBtn.disabled = false; }
    if (cameraSelectList) cameraSelectList.style.display = 'none';
    if (confirmCameraBtn) { confirmCameraBtn.style.display = 'none'; confirmCameraBtn.disabled = false; }
    if (connectionStatusElement) connectionStatusElement.textContent = "Desconectado";
    if (cameraStatusMsg) {
        cameraStatusMsg.textContent = message || 'Selecciona cámara para iniciar.';
        cameraStatusMsg.style.color = message && message.toLowerCase().includes('error') ? 'red' : 'inherit';
    }
    selectedDeviceId = null;
    updateConnectionStatusBasedOnAudio(); // Actualiza estado a Desconectado
}
function stopMediaPipeProcessing() {
    console.log("Deteniendo MediaPipe y cámara...");
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    if (videoElement) videoElement.srcObject = null;
    if (canvasCtx) canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
     // Opcional: Cerrar instancia de Hands si planeas recrearla
     // if (hands) hands.close(); // Consultar documentación si existe este método
}

// --- Event Listeners UI ---
if (showModalBtn) showModalBtn.addEventListener('click', openModal);
if (closeModalBtn) closeModalBtn.addEventListener('click', closeModal);
if (confirmModalBtn) confirmModalBtn.addEventListener('click', closeModal);
if (circleButton) circleButton.addEventListener('click', toggleFullscreen); // Botón dedicado

window.addEventListener('click', (event) => { // Click para activar audio y cerrar modal
    if (modal && event.target === modal) closeModal();
    if (!audioContextResumed && audioContext && audioContext.state === 'suspended') resumeAudioContext();
});
document.addEventListener('keydown', (event) => { // Escape para salir de fullscreen
    if (event.key === 'Escape' && document.fullscreenElement) {
        if (circleButton) circleButton.textContent = '⛶';
         if(mainPanel) mainPanel.classList.remove('fullscreen');
    }
});
if (saveButton) saveButton.addEventListener('click', saveUserSounds);
if (playButton) playButton.addEventListener('click', () => { // Botón Tocar -> Fullscreen y activar audio
    toggleFullscreen();
    if (!audioContextResumed) initializeAudio();
});


document.addEventListener('DOMContentLoaded', function() {
    const heartIcon = document.getElementById('heartIcon');
    const heartCount = document.getElementById('heartCount');
    const suggestionBox = document.getElementById('suggestionBox');
    const submitSuggestionButton = document.getElementById('submitSuggestion');
  
    let likeCount = 0;
  
    heartIcon.addEventListener('click', function() {
      likeCount++;
      heartCount.textContent = likeCount;
      heartIcon.style.color = '#e74c3c'; // Cambia a rojo al dar like
    });
  
    submitSuggestionButton.addEventListener('click', function() {
      const suggestion = suggestionBox.value;
      if (suggestion.trim() !== '') {
        // Aquí puedes enviar la sugerencia a un servidor o almacenarla localmente
        alert('Sugerencia enviada: ' + suggestion);  // Reemplaza esto con tu lógica de envío
        suggestionBox.value = ''; // Limpia el textarea
      } else {
        alert('Por favor, escribe una sugerencia.');
      }
    });
  });

// Añadir el control del "like"
let hasLiked = false;  // Variable para verificar si ya se ha dado like

// Función para enviar like
function sendLike() {
    const likeButton = document.getElementById('likeButton');
    const heartIcon = document.getElementById('heartIcon');

    if (!hasLiked) {  // Solo si no ha dado like
        likeButton.disabled = true;  // Deshabilita el botón
        heartIcon.style.color = 'red';  // Cambia el color del corazón al rojo
        const likeData = JSON.stringify({type: 'like'});

        fetch('/like', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: likeData
        }).then(response => {
            if (response.ok) {
                hasLiked = true;  // Marca que ya se ha dado like
                alert('Like enviado.');
            } else {
                likeButton.disabled = false;
                heartIcon.style.color = '#ccc';  // Restaurar color si ocurre un error
                alert('Error al enviar like.');
            }
        }).catch(error => {
            likeButton.disabled = false;
            heartIcon.style.color = '#ccc';  // Restaurar color si hay error
            alert('Error al enviar like.');
        });
    } else {
        alert('Ya has dado un like.');
    }
}
  // Enviar sugerencia
  function sendSuggestion() {
    const suggestionButton = document.getElementById('suggestionButton');
    const suggestionBox = document.getElementById('suggestionBox'); // Definir suggestionBox correctamente
  
    if (suggestionButton && suggestionBox) {
      suggestionButton.disabled = true;
  
      const suggestionData = JSON.stringify({ type: 'suggest', text: suggestionBox.value });
  
      fetch('/suggest', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: suggestionData
      }).then(response => {
        if (response.ok) {
          alert('Sugerencia enviada.');
        } else {
          alert('Error al enviar sugerencia.');
        }
      }).catch(error => {
        alert('Error al enviar sugerencia.');
      }).finally(() => {
        suggestionButton.disabled = false;
      });
    }
  }
  
// Bloqueo de contexto/teclas (opcional)
// document.addEventListener("contextmenu", event => event.preventDefault());
// document.addEventListener("keydown", event => { if ((event.ctrlKey && (event.key === "u" || event.key === "s")) || event.key === "F12") event.preventDefault(); });