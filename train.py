from ultralytics import YOLO

def main():
    # Carrega o modelo pré-treinado do YOLOv11 (versão nano para testes rápidos)
    model = YOLO("yolo11n.pt")

    # Inicia o treinamento (fine-tuning) com o dataset configurado
    # Certifique-se de que o arquivo data.yaml do seu dataset baixado do Roboflow está acessível
    results = model.train(
        data="dataset/data.yaml",  # Substitua pelo caminho real do data.yaml do seu dataset
        epochs=10,                  # Número de épocas
        imgsz=640,                  # Tamanho da imagem
        batch=16,                   # Tamanho do batch
        name="groceries_yolo11_model" # Nome da pasta de salvamento dos resultados
    )

    print("Treinamento concluído com sucesso!")
    
    # Validação do modelo
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map}")

if __name__ == "__main__":
    main()