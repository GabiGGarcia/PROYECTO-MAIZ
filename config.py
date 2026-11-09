"""
config.py - Parámetros compartidos por todos los scripts del proyecto.
Proyecto: Detección de enfermedades en hojas de maíz (PlantVillage) - TA3 IA USAT
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "maiz"          # aquí quedan train/ val/ test/

# En Google Colab se define MAIZ_SALIDA=/content/drive/MyDrive/maiz_ia_salida para que
# los modelos y resultados se guarden en Drive y no se pierdan si se cae la sesión.
SALIDA_DIR = Path(os.environ.get("MAIZ_SALIDA", BASE_DIR))
MODELOS_DIR = SALIDA_DIR / "modelos"           # pesos entrenados (.pt)
RESULTADOS_DIR = SALIDA_DIR / "resultados"     # gráficas, reportes, CSV

# Los tres algoritmos comparados (nombre corto usado en la línea de comandos)
ARQUITECTURAS = ["cnn", "mobilenet", "resnet"]
NOMBRES_MODELO = {"cnn": "CNN propia", "mobilenet": "MobileNetV2", "resnet": "ResNet50"}

for d in (MODELOS_DIR, RESULTADOS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Clases (nombre de carpeta -> nombre para mostrar)
# El orden alfabético de las carpetas define el índice de cada clase.
# ---------------------------------------------------------------------------
CLASES_ES = {
    "mancha_gris": "Mancha gris (Cercospora zeae-maydis)",
    "roya_comun": "Roya común (Puccinia sorghi)",
    "sana": "Hoja sana",
    "tizon_norte": "Tizón foliar del norte (Exserohilum turcicum)",
}

RECOMENDACIONES = {
    "mancha_gris": "Favorecida por alta humedad y rastrojo. Rotar cultivos, manejar "
                   "rastrojo y consultar a un técnico/INIA sobre fungicidas autorizados.",
    "roya_comun": "Pústulas color óxido. Usar híbridos tolerantes y vigilar el avance; "
                  "consultar a un técnico antes de aplicar fungicida.",
    "sana": "No se detectan síntomas. Continuar con el monitoreo periódico del campo.",
    "tizon_norte": "Lesiones alargadas gris-verdosas. Rotación, híbridos resistentes y "
                   "evaluación técnica para control químico si avanza.",
}

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto (se pueden cambiar por línea de comandos)
# ---------------------------------------------------------------------------
IMG_SIZE = 224          # usa 160 si tu laptop es lenta
BATCH_SIZE = 32
SEED = 42

# Normalización de ImageNet (necesaria para MobileNetV2 preentrenada)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# En Windows los DataLoader con varios procesos suelen dar problemas -> 0
NUM_WORKERS = 0 if os.name == "nt" else 2

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def obtener_dispositivo(preferencia: str = "auto"):
    """Devuelve el dispositivo de cómputo: cuda > directml (opcional) > cpu."""
    import torch
    if preferencia in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if preferencia in ("auto", "dml", "directml"):
        try:  # GPU integrada Intel/AMD en Windows (pip install torch-directml)
            import torch_directml
            if preferencia != "auto":
                return torch_directml.device()
        except ImportError:
            if preferencia != "auto":
                print("[AVISO] torch-directml no está instalado; se usará CPU.")
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    return torch.device("cpu")
