from fastapi import FastAPI
import cv2
import numpy as np
import requests
from fastapi.responses import HTMLResponse
import base64

app = FastAPI()

def decode_hex_to_image(hex_string: str) -> np.ndarray:
    """Декодирует hex строку в изображение"""
    image_bytes = bytes.fromhex(hex_string)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

@app.get("/", response_class=HTMLResponse)
async def show_image():
    """Отображает страницу с изображением и кнопкой обновления"""
    return """
    <html>
        <head>
            <title>Camera Viewer</title>
            <style>
                .container { text-align: center; padding: 20px; }
                img { max-width: 800px; margin: 20px; }
                button { padding: 10px 20px; font-size: 16px; }
            </style>
        </head>
        <body>
            <div class="container">
                <button onclick="updateImage()">Обновить изображение</button>
                <br>
                <img id="camera-image" src="/get_image" alt="Camera Image">
            </div>
            <script>
                function updateImage() {
                    const img = document.getElementById('camera-image');
                    img.src = '/get_image?' + new Date().getTime();
                }
            </script>
        </body>
    </html>
    """

@app.get("/get_image")
async def get_image():
    """Получает и декодирует изображение с камеры"""
    try:
        response = requests.post("http://camera_processor:8000/capture")
        data = response.json()
        
        image = decode_hex_to_image(data["main_image"])
        
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return HTMLResponse(
            content=f'<img src="data:image/jpeg;base64,{image_base64}" />',
            media_type="text/html"
        )
    except Exception as e:
        return HTMLResponse(
            content=f"Error: {str(e)}",
            status_code=500
        )