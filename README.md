# 🎬 MoviePy Video Generator

**Servicio gratuito para generar videos motivacionales con MoviePy - Compatible ARM64**

Reemplaza servicios pagos como JSON2Video ($45/mes) con una solución 100% gratuita y personalizable.

## ✨ Características

- 🆓 **Completamente gratuito** - Sin límites ni costos ocultos
- 🏗️ **ARM64 nativo** - Optimizado para Oracle Free Tier y Apple Silicon
- 🎨 **Videos HD** - Soporte para resoluciones hasta 1920x1080
- 🎵 **Música de fondo** - Descarga automática desde URLs
- 📝 **Texto personalizable** - Colores, fuentes y posiciones
- 🔄 **API REST** - Integración sencilla con n8n, Make.com, etc.
- ⚡ **Procesamiento asíncrono** - Generación en background
- 📊 **Monitoreo** - Estado y progreso en tiempo real

## 🚀 Inicio Rápido

### Con Docker (Recomendado)

```bash
# Clonar repositorio
git clone https://github.com/Nuevededos/moviepy-video-generator.git
cd moviepy-video-generator

# Construir imagen
docker build -t moviepy-generator .

# Ejecutar contenedor
docker run -p 8080:8080 -v $(pwd)/videos:/app/videos moviepy-generator
```

### Instalación Local

```bash
# Instalar dependencias del sistema (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg imagemagick python3-pip

# Instalar dependencias Python
pip install -r requirements.txt

# Ejecutar servidor
python main.py
```

## 📋 API Reference

### Generar Video

**POST** `/generate-video`

```json
{
  "title": "Video Motivacional - Mindfulness",
  "clips": [
    {
      "text": "Respira profundamente y encuentra tu paz interior",
      "duration": 8,
      "background_color": "#009c8c",
      "text_color": "#ffffff",
      "font_size": 60,
      "position": "center"
    },
    {
      "text": "Cada momento es una nueva oportunidad",
      "duration": 6,
      "background_color": "#c3e7e5",
      "text_color": "#000000",
      "font_size": 55,
      "position": "center"
    }
  ],
  "background_music_url": "https://example.com/relaxing-music.mp3",
  "resolution": "1920x1080",
  "fps": 30,
  "music_volume": 0.3
}
```

**Respuesta:**
```json
{
  "video_id": "video_abc123def456_1703123456",
  "video_url": "/videos/video_abc123def456_1703123456",
  "duration": 14.0,
  "status": "queued",
  "message": "Video generation started"
}
```

### Otros Endpoints

- **GET** `/health` - Health check
- **GET** `/status/{video_id}` - Estado del video
- **GET** `/videos/{video_id}` - Descargar video
- **GET** `/videos` - Listar todos los videos
- **DELETE** `/videos/{video_id}` - Eliminar video

## 🔧 Configuración

### Variables de Entorno

```bash
VIDEO_OUTPUT_DIR=/app/videos      # Directorio de salida
MAX_WORKERS=2                     # Trabajadores concurrentes
MAX_VIDEO_DURATION=300            # Duración máxima (segundos)
TEMP_DIR=/tmp                     # Directorio temporal
```

### Integración con n8n

1. **Nodo HTTP Request**
   - Method: POST
   - URL: `http://tu-servidor:8080/generate-video`
   - Body: JSON con estructura de clips

2. **Nodo Wait** 
   - Modo: Wait for Webhook
   - Timeout: 300 segundos

3. **Nodo HTTP Request (Estado)**
   - URL: `http://tu-servidor:8080/status/{{$json.video_id}}`
   - Loop hasta status = "completed"

4. **Nodo HTTP Request (Descarga)**
   - URL: `http://tu-servidor:8080/videos/{{$json.video_id}}`
   - Response: Binary Data

## 🎨 Ejemplos de Uso

### Video Básico
```json
{
  "title": "Mensaje Simple",
  "clips": [
    {
      "text": "Hola mundo",
      "duration": 3
    }
  ]
}
```

### Video Completo con Música
```json
{
  "title": "Video Terapéutico Completo",
  "clips": [
    {
      "text": "Bienvenido a tu espacio de calma",
      "duration": 5,
      "background_color": "#009c8c",
      "text_color": "#ffffff",
      "font_size": 70
    },
    {
      "text": "Cierra los ojos y respira profundamente",
      "duration": 8,
      "background_color": "#c3e7e5",
      "text_color": "#000000",
      "font_size": 60
    },
    {
      "text": "Siente como la tranquilidad llena tu ser",
      "duration": 10,
      "background_color": "#009c8c",
      "text_color": "#ffffff",
      "font_size": 55
    }
  ],
  "background_music_url": "https://www.soundjay.com/misc/sounds/meditation.mp3",
  "resolution": "1920x1080",
  "fps": 30,
  "music_volume": 0.25
}
```

## 🐳 Deploy con Coolify

1. **Agregar Source**
   - Repository: `https://github.com/Nuevededos/moviepy-video-generator`
   - Branch: `main`

2. **Configurar Servicio**
   - Port: `8080`
   - Health Check: `/health`

3. **Variables de Entorno**
   ```
   MAX_WORKERS=2
   MAX_VIDEO_DURATION=300
   ```

4. **Volúmenes**
   - `/app/videos` → Persistir videos generados

## 🛠️ Troubleshooting

### Error: "Font not found"
```bash
# Instalar fuentes adicionales
sudo apt-get install fonts-dejavu-core fonts-liberation
```

### Error: "FFmpeg not found"
```bash
# Instalar FFmpeg
sudo apt-get install ffmpeg
```

### Videos muy pesados
- Reducir `fps` a 24 o 25
- Usar resolución 1280x720 en lugar de 1920x1080
- Limitar duración de clips

### Memoria insuficiente
- Reducir `MAX_WORKERS` a 1
- Procesar videos más cortos
- Aumentar swap del sistema

## 📊 Rendimiento

**Oracle Free Tier (ARM64, 24GB RAM, 4 cores):**
- Video 1080p, 60s: ~2-3 minutos
- Video 720p, 60s: ~1-2 minutos
- Memoria por video: ~1-2GB
- Almacenamiento por video: ~50-150MB

## 🔄 Migración desde JSON2Video

### Workflow Original
```
n8n → Google Sheets → Claude → JSON2Video ($45/mes) → YouTube
```

### Workflow Nuevo
```
n8n → Google Sheets → Claude → MoviePy Service (FREE) → YouTube
```

### Cambios Necesarios en n8n
1. Reemplazar nodo JSON2Video con HTTP Request
2. Ajustar formato de payload (ver ejemplos arriba)
3. Agregar polling de estado si es necesario
4. Mantener resto del workflow igual

## 🤝 Contribución

1. Fork el repositorio
2. Crear branch: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## 📄 Licencia

MIT License - Uso libre sin restricciones.

## ✨ Créditos

Creado para optimizar workflows de videos motivacionales psicológicos.

**Ahorro estimado:** $540/año vs JSON2Video 🎉