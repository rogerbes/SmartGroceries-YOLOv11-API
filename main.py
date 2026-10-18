from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI(
    title="API de Detecção de Objetos - Groceries YOLOv11",
    description="Microserviço backend para detecção de itens de supermercado usando YOLOv11 e FastAPI.",
    version="1.0.0"
)

# Carrega o modelo treinado (ajuste o caminho para o seu best.pt gerado no treinamento)
MODEL_PATH = "runs/detect/groceries_yolo11_model-2/weights/best.pt"
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Aviso: Modelo treinado não encontrado em {MODEL_PATH}. Carregando modelo padrão para testes.")
    model = YOLO("yolo11n.pt")

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    # Valida se o arquivo enviado é uma imagem
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo enviado deve ser uma imagem.")
    
    try:
        # Lê o conteúdo da imagem enviada
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Realiza a inferência com o YOLOv11
        results = model(image)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extrai as coordenadas da bounding box, confiança e classe
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 2),
                    "box": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    }
                })
                
        return {
            "status": "success",
            "detections": detections
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento da imagem: {str(e)}")

@app.get("/")
def root():
    return {"message": "API de Visão Computacional YOLOv11 rodando com sucesso! Acesse /docs para o Swagger."}