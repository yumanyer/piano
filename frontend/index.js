// --- Elementos del DOM ---
const modal = document.getElementById('infoModal');
const showModalBtn = document.getElementById('showModal');
const closeModalBtn = document.getElementById('closeModal');
const confirmModalBtn = document.getElementById('confirmModal');
const mainContainer = document.querySelector('.container');
const mainPanel = document.getElementById('mainPanel');
const circleButton = document.getElementById('fullscreenBtn'); // Botón dedicado a pantalla completa
const videoElement = document.getElementById('videoFeed');
const connectionStatusElement = document.getElementById('connectionStatus');
const fileInputs = document.querySelectorAll('.custom-input');
const saveButton = document.getElementById('saveButton'); // Ya estaba declarado
const playButton = document.getElementById('playButton'); // Ya estaba declarado

// --- Elementos para Selección de Cámara ---
const cameraSetupDiv = document.getElementById('cameraSetup');
const requestCameraBtn = document.getElementById('requestCameraButton');
const cameraSelectList = document.getElementById('cameraSelect');
const confirmCameraBtn = document.getElementById('confirmCameraButton');
const cameraStatusMsg = document.getElementById('cameraStatusMessage');

// --- Configuración WebSocket ---
const WS_URL = 'ws://localhost:8765';
let socket;

// --- Configuración Web Audio API ---
let audioContext;
const soundBuffers = new Map(); // Almacenará los AudioBuffers (default y custom)
const defaultSoundFiles = [ // Rutas a sonidos por defecto (índices 0-9)
    '../sounds/DO.wav', '../sounds/RE.wav', '../sounds/MI.wav', '../sounds/FA.wav', '../sounds/SOL.wav',
    '../sounds/LA.wav', '../sounds/SI.wav', '../sounds/DO%23.wav', '../sounds/RE%23.wav', '../sounds/FA%23.wav'
];
// Mapeo para URLs de sonidos  - Índice 0-9
let userSoundURLs = {};
defaultSoundFiles.forEach((url, index) => {
    userSoundURLs[index] = url; // Inicializar con las URLs por defecto
});

let soundsLoadedCount = 0;
let totalSoundsToLoad = defaultSoundFiles.length;
let audioContextResumed = false;

// --- Inicialización ---
window.addEventListener('load', () => {
    console.log('Página cargada.');
    // Cargar configuración de sonidos guardada ANTES de pre-cargar
    loadUserSounds();
    // Mostrar configuración de cámara
    if (cameraSetupDiv) cameraSetupDiv.style.display = 'flex';
    if (mainContainer) mainContainer.style.display = 'none';

    // Añadir listener para reproducir sonido al hacer clic en el número
    addNumberClickListeners();
});

// --- Listener para cargar sonidos personalizados ---
fileInputs.forEach(input => {
    input.addEventListener('change', async function(event) {
        const file = event.target.files[0];
        const key = parseInt(event.target.dataset.key); 
        const parentItem = this.closest('.file-item');
        let label = parentItem ? parentItem.querySelector('.file-name') : null;

        if (label) {
            label.remove();
        }

        if (file && audioContext) { // Necesitamos audioContext para decodificar
            const fileName = file.name;
            const fileURL = URL.createObjectURL(file); // Crear Blob URL

            console.log(`Archivo seleccionado para tecla ${key + 1}: ${fileName}`);

            // Mostrar nombre de archivo
            label = document.createElement('span');
            label.className = 'file-name';
            label.textContent = ` (${fileName})`;
            this.parentNode.insertBefore(label, this.nextSibling);

            try {
                // 1. Actualizar la URL del usuario (para guardar/cargar estado)
                userSoundURLs[key] = fileURL;

                // 2. Cargar y decodificar el audio para Web Audio API
                const arrayBuffer = await file.arrayBuffer();
                const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

                // 3. Almacenar el AudioBuffer para reproducción inmediata
                soundBuffers.set(key, audioBuffer);
                console.log(`Sonido personalizado para tecla ${key + 1} cargado y listo.`);


                 if (soundsLoadedCount >= totalSoundsToLoad) { 
                    console.log(`Sonido personalizado para ${key+1} reemplazó al default.`);
                 }


            } catch (error) {
                console.error(`Error al procesar el archivo de audio para la tecla ${key + 1}:`, error);
                alert(`Error al cargar el sonido: ${error.message}`);
                userSoundURLs[key] = defaultSoundFiles[key];
                 if (label) label.remove();
            }

        } else if (!audioContext) {
            alert("El sistema de audio no está inicializado. Intenta interactuar con la página (ej. clic) y vuelve a intentarlo.");
             if (label) label.remove();
            event.target.value = null; 
        } else {

            const defaultURL = defaultSoundFiles[key];
            if (userSoundURLs[key] !== defaultURL) {
                 userSoundURLs[key] = defaultURL;
                 console.log(`Restaurado sonido por defecto para tecla ${key + 1}`);
                 // Recargar el buffer por defecto
                 loadSound(defaultURL, key, () => {
                      console.log(`Buffer por defecto para tecla ${key+1} recargado.`);
                 }, true); // forceReload = true
            }
        }
    });
});

// --- Función para guardar sonidos (URLs) en localStorage ---
function saveUserSounds() {
    try {
        // Solo guardamos las URLs (incluyendo Blob URLs temporales)
        localStorage.setItem("userSoundURLs", JSON.stringify(userSoundURLs));
        alert("Configuración de sonidos guardada.\nNota: Los sonidos personalizados deberán volver a seleccionarse si recargas la página.");
    } catch (error) {
        console.error("Error guardando sonidos en localStorage:", error);
        alert("Error al guardar la configuración de sonidos.");
    }
}

// --- Función para cargar sonidos (URLs) desde localStorage ---
function loadUserSounds() {
    const savedURLs = localStorage.getItem("userSoundURLs");
    if (savedURLs) {
        try {
            const parsedURLs = JSON.parse(savedURLs);


            userSoundURLs = parsedURLs;
            console.log("Configuración de sonidos cargada desde localStorage.");

            Object.keys(parsedURLs).forEach(keyIndexStr => {
                const keyIndex = parseInt(keyIndexStr);
                if (parsedURLs[keyIndex].startsWith('blob:')) {
                    const input = document.querySelector(`.custom-input[data-key="${keyIndex}"]`);
                    const parentItem = input ? input.closest('.file-item') : null;
                     if(parentItem){
                        // Añadir indicación visual de que había un sonido custom aquí
                        let indicator = parentItem.querySelector('.custom-indicator');
                        if(!indicator) {
                           indicator = document.createElement('span');
                           indicator.className = 'custom-indicator';
                           indicator.textContent = ' (custom - recargar)';
                           indicator.style.fontSize = '0.8em';
                           indicator.style.marginLeft = '5px';
                           input.parentNode.insertBefore(indicator, input.nextSibling);
                        }
                     }

                    userSoundURLs[keyIndex] = defaultSoundFiles[keyIndex];
                }
            });

        } catch (error) {
            console.error("Error cargando sonidos desde localStorage:", error);
            localStorage.removeItem("userSoundURLs");
             
             userSoundURLs = {};
             defaultSoundFiles.forEach((url, index) => { userSoundURLs[index] = url; });
        }
    } else {
         console.log("No se encontró configuración de sonidos guardada. Usando defaults.");
        
          userSoundURLs = {};
          defaultSoundFiles.forEach((url, index) => { userSoundURLs[index] = url; });
    }
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
                // Intentar reproducir incluso si se acaba de intentar reanudar
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

// --- Función para listar cámaras (sin cambios respecto a tu versión más reciente) ---
async function listCameras() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        if (videoDevices.length === 0) {
             if (cameraStatusMsg) cameraStatusMsg.textContent = 'No se encontraron cámaras. Verifica los permisos o si hay alguna conectada.';
             return;
        }

        // Verifica si necesitamos permiso para obtener nombres (labels)
        let needsPermission = videoDevices.every(device => !device.label);

        if (needsPermission) {
             if (cameraStatusMsg) cameraStatusMsg.textContent = 'Permiso necesario para listar nombres. Intentando obtener permiso...';
             try {
                 // Pedir permiso temporal
                 const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                 tempStream.getTracks().forEach(track => track.stop()); // Detenerlo inmediatamente
                 // Volver a listar ahora que (esperemos) tenemos permiso
                 await listCameras(); // Llamada recursiva
                 return; // Salir de esta ejecución
             } catch (permError) {
                 console.error("Error al obtener permiso para getUserMedia:", permError);
                 if (cameraStatusMsg) cameraStatusMsg.textContent = `Error al obtener permiso: ${permError.name}. Recarga e intenta de nuevo.`;
                 // Mostrar el botón de intentar de nuevo si falló el permiso
                 if(requestCameraBtn) requestCameraBtn.style.display = 'inline-block';
                 if(cameraSelectList) cameraSelectList.style.display = 'none';
                 if(confirmCameraBtn) confirmCameraBtn.style.display = 'none';
                 return;
             }
        }

        if (cameraSelectList) {
            cameraSelectList.innerHTML = ''; 
            videoDevices.forEach((device, index) => {
                const option = document.createElement('option');
                option.value = index; // Usamos el ÍNDICE como valor para OpenCV
                option.text = device.label || `Cámara ${index + 1}`;
                option.dataset.deviceId = device.deviceId; // Guardamos deviceId para pruebas
                cameraSelectList.appendChild(option);
            });

            // Mostrar selector y botón de confirmar, ocultar el de buscar
            cameraSelectList.style.display = 'inline-block';
            if (confirmCameraBtn) confirmCameraBtn.style.display = 'inline-block';
            if (requestCameraBtn) requestCameraBtn.style.display = 'none';
            if (cameraStatusMsg) cameraStatusMsg.textContent = 'Selecciona una cámara:';
        }

    } catch (error) {
        console.error('Error en enumerateDevices o flujo de permisos:', error);
        let message = `Error al acceder a dispositivos: ${error.name || error.message}`;
        if (error.name === 'NotAllowedError') {
            message = 'Permiso denegado para acceder a la cámara.';
        } else if (error.name === 'NotFoundError') {
            message = 'No se encontró ninguna cámara compatible.';
        }
         if (cameraStatusMsg) {
             cameraStatusMsg.textContent = message;
             cameraStatusMsg.style.color = 'red';
         }
          // Mostrar botón de reintentar en caso de error
         if(requestCameraBtn) requestCameraBtn.style.display = 'inline-block';
         if(cameraSelectList) cameraSelectList.style.display = 'none';
         if(confirmCameraBtn) confirmCameraBtn.style.display = 'none';
    }
}


// --- Función para probar cámara (sin cambios) ---
async function testCamera(deviceId) {
    let stream = null;
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: 640, height: 480 } // Pedir resolución baja para test rápido
        });
        console.log("Prueba de cámara exitosa para deviceId:", deviceId);
        return true;
    } catch (error) {
        console.error(`Error al probar la cámara con deviceId ${deviceId}:`, error.name, error.message);
        return false;
    } finally {
        // Asegurarse de detener el stream de prueba
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            console.log("Stream de prueba detenido.");
        }
    }
}


// --- Event Listener para el botón de confirmar cámara (con prueba) ---
if (confirmCameraBtn) {
    confirmCameraBtn.addEventListener('click', async () => {
        const selectedIndex = cameraSelectList.value;
        if (selectedIndex === '' || selectedIndex === null) {
            if (cameraStatusMsg) cameraStatusMsg.textContent = 'Por favor, selecciona una cámara de la lista.';
            return;
        }

        const selectedOption = cameraSelectList.options[cameraSelectList.selectedIndex];
        const selectedCameraLabel = selectedOption.text;
        const selectedDeviceId = selectedOption.dataset.deviceId;
        const cameraIndexToSend = parseInt(selectedIndex, 10);

        if (cameraStatusMsg) cameraStatusMsg.textContent = `Probando cámara: ${selectedCameraLabel}...`;
        confirmCameraBtn.disabled = true; // Deshabilitar botón mientras prueba

        const cameraWorks = await testCamera(selectedDeviceId);

        if (!cameraWorks) {
            if (cameraStatusMsg) cameraStatusMsg.textContent = `No se pudo acceder a '${selectedCameraLabel}'. Intenta con otra cámara o verifica permisos.`;
            confirmCameraBtn.disabled = false; // Habilitar botón de nuevo
            return;
        }

        // Si la prueba fue exitosa:
        if (cameraStatusMsg) cameraStatusMsg.textContent = `Cámara '${selectedCameraLabel}' OK. Conectando al servidor...`;

        // Ocultar setup y mostrar UI principal ANTES de conectar WebSocket
        if (cameraSetupDiv) cameraSetupDiv.style.display = 'none';
        if (mainContainer) mainContainer.style.display = 'flex';
        if (videoElement) videoElement.style.display = 'block';

        // Intentar inicializar/reanudar AudioContext AHORA (interacción del usuario)
        // y comenzar la precarga de sonidos (usando userSoundURLs)
        initializeAudio();

        // Iniciar conexión WebSocket y enviar selección
        setupWebSocket(cameraIndexToSend);

        confirmCameraBtn.disabled = false; // Habilitar botón por si necesita reconectar
    });
}


// --- Funciones WebSocket (sin cambios funcionales mayores) ---
function setupWebSocket(cameraIndexToSend) {
    if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        console.log("WebSocket ya está conectado o conectándose.");
        // Podríamos reenviar el índice si es necesario por alguna razón, pero usualmente no.
        // socket.send(JSON.stringify({ type: 'set_camera_index', index: cameraIndexToSend }));
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
            connectionStatusElement.textContent = 'Enviando selección de cámara...';
            connectionStatusElement.style.color = 'yellow';
        }
        socket.send(JSON.stringify({ type: 'set_camera_index', index: cameraIndexToSend }));
        console.log(`Enviado índice de cámara: ${cameraIndexToSend}`);
        // Estado se actualizará en onmessage o cuando el audio cargue
    };

    socket.onerror = (error) => {
        console.error('WebSocket Error:', error);
        resetToCameraSelection('Error de conexión WebSocket. Intenta de nuevo.');
    };

    socket.onclose = (event) => {
        console.log('WebSocket desconectado:', event.code, event.reason);
        resetToCameraSelection(`Desconectado (${event.code}). Vuelve a seleccionar la cámara.`);
    };

    socket.onmessage = handleWebSocketMessage; // Sin cambios aquí
}

function handleWebSocketMessage(event) { // Sin cambios funcionales mayores aquí
    try {
        const data = JSON.parse(event.data);

        if (data.type === 'initial_state') {
            console.log("Received initial finger state:", data.fingers);
            data.fingers.forEach((isDown, index) => {
                updateFingerUI(index, isDown);
            });
             updateConnectionStatusBasedOnAudio(); // Actualizar estado ahora que estamos conectados

        } else if (data.type === 'finger_down') {
            // El índice recibido (data.finger_id) debe ser 0-9
            playSound(data.finger_id); // Llamar a nuestra función de audio
            updateFingerUI(data.finger_id, true);

        } else if (data.type === 'finger_up') {
            updateFingerUI(data.finger_id, false);

        } else if (data.type === 'video_frame') {
            if (videoElement && videoElement.style.display !== 'none') {
                videoElement.src = 'data:image/jpeg;base64,' + data.image;
            }
             // Actualizar estado si es la primera vez que llega video
             if (connectionStatusElement && connectionStatusElement.textContent.includes('Esperando video')) {
                 updateConnectionStatusBasedOnAudio();
             }

        } else if (data.type === 'status') {
             console.log("Status from backend:", data.message);
             // Podríamos mostrar mensajes informativos aquí si no son errores
             if (connectionStatusElement && connectionStatusElement.style.color !== 'red' && !data.message.toLowerCase().includes("error")) {
                 // No sobreescribir el estado de carga de audio a menos que sea relevante
             }

        } else if (data.type === 'error') {
             console.error("Error from backend:", data.message);
             if (connectionStatusElement) {
                 connectionStatusElement.textContent = `Error Backend: ${data.message}`;
                 connectionStatusElement.style.color = 'red';
             }
             if (data.message.toLowerCase().includes("camera") || data.message.toLowerCase().includes("opencv")) {
                 resetToCameraSelection(`Error de cámara en servidor: ${data.message}. Intenta otra cámara.`);
             }
        }

    } catch (e) {
        console.error("Error procesando mensaje WebSocket:", e, event.data);
    }
}


// --- Funciones Web Audio API (modificada para usar userSoundURLs) ---
function initializeAudio() {
     if (!audioContext) {
         try {
            console.log("Intentando crear AudioContext...");
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            console.log("AudioContext creado. Estado inicial:", audioContext.state);

            if (audioContext.state === 'suspended') {
                console.log("AudioContext suspendido. Necesita interacción del usuario (click/tap).");
                 updateConnectionStatusBasedOnAudio(); // Indicar que se necesita interacción
            } else {
                audioContextResumed = true;
                preloadSounds(); // Iniciar precarga si se pudo crear activo
            }

         } catch (e) {
             console.error("Error creando AudioContext:", e);
              if (connectionStatusElement) {
                 connectionStatusElement.textContent += " (Error Audio)";
                 connectionStatusElement.style.color = 'red';
              }
              // Deshabilitar carga de archivos si falla AudioContext
              fileInputs.forEach(input => input.disabled = true);
              alert("No se pudo inicializar el audio. La carga y reproducción de sonidos no funcionará.");
         }
     } else if (audioContext.state === 'suspended') {
         resumeAudioContext(); // Intentar reanudar si ya existe y está suspendido
     } else {
         // Si ya existe y está 'running', asegurarse que la precarga se haya iniciado
         if (soundsLoadedCount === 0 && totalSoundsToLoad > 0) {
             preloadSounds();
         }
     }
}

function resumeAudioContext() {
     if (!audioContext || audioContext.state !== 'suspended' || !audioContext.resume) return;

     console.log("Intentando reanudar AudioContext...");
     audioContext.resume().then(() => {
         console.log("AudioContext reanudado con éxito.");
         audioContextResumed = true;
         // Iniciar precarga AHORA si no se hizo antes
         if (soundsLoadedCount === 0 && totalSoundsToLoad > 0) {
             preloadSounds();
         } else {
             updateConnectionStatusBasedOnAudio(); // Solo actualizar estado si ya se cargaron
         }
     }).catch(e => {
         console.error("Error al reanudar AudioContext:", e);
         if (connectionStatusElement) connectionStatusElement.textContent += ' (Click para activar audio)';
     });
}

function preloadSounds() {
    // Asegurarse que el contexto esté listo y no se haya iniciado ya la carga
    if (!audioContext || !audioContextResumed || soundsLoadedCount > 0) {
        if(soundsLoadedCount >= totalSoundsToLoad && totalSoundsToLoad > 0) {
             console.log("Sonidos ya precargados.");
             updateConnectionStatusBasedOnAudio();
        } else if (!audioContextResumed) {
            console.warn("Intento de precarga pero AudioContext no está activo.");
        }
        return;
    }
    console.log("Precargando sonidos...");
    if (connectionStatusElement) updateConnectionStatusBasedOnAudio(); // Mostrar estado inicial de carga

    soundsLoadedCount = 0; // Reiniciar contador
    totalSoundsToLoad = Object.keys(userSoundURLs).length; // Basado en las URLs actuales
    let soundsSuccessfullyLoaded = 0;
    let soundsFailedToLoad = 0;

    if (totalSoundsToLoad === 0) {
        console.warn("No hay URLs de sonido definidas para precargar.");
        updateConnectionStatusBasedOnAudio();
        return;
    }

    Object.keys(userSoundURLs).forEach(keyIndexStr => {
        const keyIndex = parseInt(keyIndexStr);
        const url = userSoundURLs[keyIndex];

        // Evitar cargar Blobs directamente aquí, ya que los de localStorage son inválidos
        // Los Blobs válidos (de la sesión actual) se cargan en el listener 'change'
        if (url && !url.startsWith('blob:')) {
            loadSound(url, keyIndex, () => {
                soundsSuccessfullyLoaded++;
                if (soundsSuccessfullyLoaded + soundsFailedToLoad === totalSoundsToLoad) {
                    console.log(`Precarga finalizada. Éxitos: ${soundsSuccessfullyLoaded}, Fallos: ${soundsFailedToLoad}`);
                    updateConnectionStatusBasedOnAudio();
                }
            }, (error) => { // Callback de error
                 soundsFailedToLoad++;
                 if (soundsSuccessfullyLoaded + soundsFailedToLoad === totalSoundsToLoad) {
                    console.log(`Precarga finalizada. Éxitos: ${soundsSuccessfullyLoaded}, Fallos: ${soundsFailedToLoad}`);
                    updateConnectionStatusBasedOnAudio();
                 }
            });
        } else if (url && url.startsWith('blob:')) {
            // Sonidos Blob cargados desde localStorage no se pueden precargar aquí.
            // Se asume que el usuario los volverá a seleccionar si los quiere.
            // O, si se cargó en la sesión actual, ya está en soundBuffers.
             console.log(`Omitiendo precarga para Blob URL (tecla ${keyIndex+1}). Se cargará al seleccionar archivo o se usará si ya está en buffer.`);
              // Contamos como 'listo' si ya está en el buffer de una carga anterior en esta sesión
              if(soundBuffers.has(keyIndex)) {
                  soundsSuccessfullyLoaded++;
              } else {
                  // Si no está en el buffer, contamos como fallo para la precarga inicial
                  // (se cargará el default en su lugar si no se selecciona archivo)
                   soundsFailedToLoad++;
                    // Intentar cargar el default como fallback si el blob no está
                    const defaultURL = defaultSoundFiles[keyIndex];
                     if(defaultURL) {
                         console.warn(`Blob URL inválido para tecla ${keyIndex+1}, cargando default: ${defaultURL}`);
                         loadSound(defaultURL, keyIndex, () => { /* Manejado arriba */ }, () => { /* Manejado arriba */ });
                     }
              }

             // Verificar si la carga terminó después de manejar el blob
             if (soundsSuccessfullyLoaded + soundsFailedToLoad === totalSoundsToLoad) {
                 console.log(`Precarga finalizada. Éxitos: ${soundsSuccessfullyLoaded}, Fallos: ${soundsFailedToLoad}`);
                 updateConnectionStatusBasedOnAudio();
             }

        } else {
            console.warn(`URL no válida o ausente para índice ${keyIndex}`);
            soundsFailedToLoad++;
             if (soundsSuccessfullyLoaded + soundsFailedToLoad === totalSoundsToLoad) {
                 updateConnectionStatusBasedOnAudio();
             }
        }
    });
}

// Modificado para aceptar callback de error y forzar recarga
function loadSound(url, index, successCallback, errorCallback, forceReload = false) {
    if (!audioContext) {
         if(errorCallback) errorCallback(new Error("AudioContext no disponible"));
         return;
    }
    // No recargar si ya existe y no se fuerza
    if (!forceReload && soundBuffers.has(index)) {
        console.log(`Sonido ${index} ya en buffer, omitiendo recarga.`);
        soundsLoadedCount++; // Contar como cargado si ya estaba
        if (successCallback) successCallback();
        return;
    }

    fetch(url)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error ${response.status} for ${url}`);
            return response.arrayBuffer();
        })
        .then(arrayBuffer => audioContext.decodeAudioData(arrayBuffer))
        .then(audioBuffer => {
            soundBuffers.set(index, audioBuffer);
             // Solo incrementar soundsLoadedCount si no se está forzando una recarga de algo ya contado
             if(!forceReload || !soundBuffers.has(index)) { // Si es carga inicial o forzada y no existía antes (poco probable forzar algo que no existe)
                 soundsLoadedCount++;
             }
            console.log(`Sonido ${index} cargado (${url.split('/').pop()})`); // Mostrar solo nombre de archivo
            if (successCallback) successCallback();
        })
        .catch(error => {
            console.error(`Error cargando sonido ${index} (${url}):`, error);
            // Eliminar del buffer si falla la carga/recarga
            soundBuffers.delete(index);
            if (errorCallback) errorCallback(error);
        });
}

function playSound(index) {
    if (!audioContextResumed) {
        console.warn(`Intento de reproducir sonido ${index}, pero AudioContext no está activo.`);
        // No intentar reanudar aquí directamente, confiar en interacción del usuario
        return;
    }
    const buffer = soundBuffers.get(index);
    if (buffer) {
        try {
            const source = audioContext.createBufferSource();
            source.buffer = buffer;
            source.connect(audioContext.destination);
            source.start(0);
        } catch (e) {
            console.error(`Error reproduciendo sonido ${index}:`, e);
             // Podría pasar si el buffer es inválido por alguna razón
        }
    } else {
        console.warn(`Buffer no encontrado para índice: ${index}. ¿Se cargó correctamente?`);
        // Podríamos intentar cargar el default aquí como fallback extremo
        // const defaultURL = defaultSoundFiles[index];
        // if (defaultURL) loadSound(defaultURL, index, () => playSound(index)); // Cargar y reintentar
    }
}

// --- Funciones UI (Modal, Fullscreen, Indicadores, Reset - sin cambios funcionales mayores) ---
function openModal() { if (modal) modal.style.display = 'flex'; }
function closeModal() {
    if (modal) modal.style.display = 'none';
    // Ya no intentamos activar audio aquí, se hace al confirmar cámara o con click general
}

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        // Preferir entrar en fullscreen con el panel principal si existe
        if (mainPanel) {
            mainPanel.requestFullscreen().catch(err => {
                console.error(`Error al entrar en pantalla completa (mainPanel): ${err.message}`, err);
                // Fallback al documento si falla el panel
                document.documentElement.requestFullscreen().catch(errDoc => {
                     console.error(`Error al entrar en pantalla completa (document): ${errDoc.message}`, errDoc);
                });
            });
        } else {
             document.documentElement.requestFullscreen().catch(errDoc => {
                  console.error(`Error al entrar en pantalla completa (document): ${errDoc.message}`, errDoc);
             });
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

document.addEventListener('fullscreenchange', () => {
    const isFullscreen = !!document.fullscreenElement;
    if (mainPanel) mainPanel.classList.toggle('fullscreen', isFullscreen);
    if (circleButton) circleButton.textContent = isFullscreen ? '✕' : '⛶';
});

function updateFingerUI(index, isDown) { // Sin cambios
    const elementId = `finger-item-${index}`;
    const listItem = document.getElementById(elementId);
    if (listItem) {
        const indicator = listItem.querySelector('.finger-status-indicator');
        if (indicator) {
            indicator.style.backgroundColor = isDown ? 'lime' : 'transparent';
        }
    }
}

function updateConnectionStatusBasedOnAudio() { // Sin cambios
    if (!connectionStatusElement) return;
    let statusText = "Conectado";
    let statusColor = "lime";

    if (!socket || socket.readyState !== WebSocket.OPEN) {
        statusText = "Desconectado";
        statusColor = "red";
    } else if (!audioContext) {
        statusText += " | Audio no inicializado";
        statusColor = "red";
    } else if (!audioContextResumed) {
        statusText += " | Audio necesita activación (haz clic)";
        statusColor = "yellow";
    } else if (soundsLoadedCount < totalSoundsToLoad) {
        statusText += ` | Cargando sonidos (${soundsLoadedCount}/${totalSoundsToLoad})...`;
        statusColor = "orange";
         // Revisar si hay errores específicos de carga
         let failedLoads = 0;
         for(let i=0; i<totalSoundsToLoad; i++) {
             if(!soundBuffers.has(i) && !userSoundURLs[i]?.startsWith('blob:')) failedLoads++; // Contar fallos si no es blob y no está en buffer
         }
         if(failedLoads > 0) {
             statusText += ` (${failedLoads} fallaron)`;
             statusColor = 'orange'; // Mantener naranja o cambiar a rojo? Naranja parece bien.
         }

    } else { // Todos cargados (o fallaron y se contaron)
        statusText += " | Sonidos Listos";
         let failedLoads = 0;
         for(let i=0; i<totalSoundsToLoad; i++) {
             if(!soundBuffers.has(i)) failedLoads++;
         }
         if (failedLoads > 0) {
             statusText += ` (${failedLoads} fallaron)`;
             statusColor = 'orange';
         }
    }

    connectionStatusElement.textContent = statusText;
    connectionStatusElement.style.color = statusColor;
}

function resetToCameraSelection(message) { // Sin cambios
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.close();
    }
    socket = null;

    if (mainContainer) mainContainer.style.display = 'none';
    if (cameraSetupDiv) cameraSetupDiv.style.display = 'flex';
    if (requestCameraBtn) requestCameraBtn.style.display = 'inline-block';
    if (cameraSelectList) cameraSelectList.style.display = 'none';
    if (confirmCameraBtn) confirmCameraBtn.style.display = 'none';
    if (videoElement) {
        videoElement.style.display = 'none';
        videoElement.src = ""; // Limpiar imagen anterior
    }
    if (connectionStatusElement) connectionStatusElement.textContent = "Desconectado";

    if (cameraStatusMsg) {
        cameraStatusMsg.textContent = message || 'Desconectado. Selecciona una cámara para reconectar.';
        cameraStatusMsg.style.color = message && message.toLowerCase().includes('error') ? 'red' : 'inherit'; // Color rojo si es mensaje de error
    }

    // Resetear indicadores de dedos
    for (let i = 0; i < 10; i++) { updateFingerUI(i, false); }

    // Resetear estado de carga de audio para la próxima conexión
    soundsLoadedCount = 0;
    // No reiniciar audioContext ni audioContextResumed aquí,
    // podría seguir activo y es mejor no recrearlo innecesariamente.
    // soundBuffers tampoco se limpia, la precarga los sobrescribirá si es necesario.
}


// --- Event Listeners UI (Modal, Fullscreen, Botones Pie de página) ---
if (showModalBtn) showModalBtn.addEventListener('click', openModal);
if (closeModalBtn) closeModalBtn.addEventListener('click', closeModal);
if (confirmModalBtn) confirmModalBtn.addEventListener('click', closeModal);
if (circleButton) circleButton.addEventListener('click', toggleFullscreen); // Botón dedicado

// Click fuera del modal para cerrar Y PARA ACTIVAR AUDIO
window.addEventListener('click', (event) => {
    if (modal && event.target === modal) {
        closeModal();
    }
    // Intenta reanudar el audio en CUALQUIER click si está suspendido
    if (!audioContextResumed && audioContext && audioContext.state === 'suspended'){
         resumeAudioContext();
     }
});

// Escape para salir de pantalla completa
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && document.fullscreenElement) {
        // El navegador maneja Escape, pero podríamos querer actualizar nuestro botón
        if (circleButton) circleButton.textContent = '⛶';
         if(mainPanel) mainPanel.classList.remove('fullscreen');
    }
});

// Botón Guardar (AHORA FUNCIONAL para guardar URLs)
if (saveButton) {
     saveButton.addEventListener('click', saveUserSounds);
}

// Botón Tocar (AHORA activa PANTALLA COMPLETA)
if (playButton) {
    playButton.addEventListener('click', () => {
        console.log("Botón 'Tocar' presionado. Activando pantalla completa...");
        toggleFullscreen(); // Activar/desactivar pantalla completa
         // También intentar activar audio si no lo está
         if (!audioContextResumed) {
            console.log("Intentando activar/reanudar audio desde botón 'Tocar'");
            initializeAudio(); // Asegura que se intente crear o reanudar
         }
    });
}
// document.addEventListener("contextmenu", event => event.preventDefault());
// document.addEventListener("keydown", event => {
//     // Bloquear teclas de acceso rápido para ver el código fuente
//     if (event.ctrlKey && (event.key === "u" || event.key === "s" || event.key === "j")) {
//         event.preventDefault();
//     }

//     // Bloquear la tecla F12
//     if (event.key === "F12") {
//         event.preventDefault();
//     }
// });

