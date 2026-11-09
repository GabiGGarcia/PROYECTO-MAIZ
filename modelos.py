"""
modelos.py - Arquitecturas CNN usadas en el proyecto.

1) CNNMaiz      : CNN propia desde cero (línea base, inspirada en el ejemplo de arroz del curso).
2) MobileNetV2  : transfer learning con pesos de ImageNet (modelo ligero, viable en laptop).
3) ResNet50     : transfer learning con pesos de ImageNet (modelo profundo de referencia).
"""
import torch
import torch.nn as nn
from torchvision import models


def bloque_conv(c_in, c_out):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNNMaiz(nn.Module):
    """CNN de 5 bloques convolucionales (16-32-64-128-256 filtros) + clasificador denso.
    Se usa GlobalAveragePooling en vez de Flatten para reducir parámetros (~0.4 M)."""

    def __init__(self, num_clases: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            bloque_conv(3, 16),
            bloque_conv(16, 32),
            bloque_conv(32, 64),
            bloque_conv(64, 128),
            bloque_conv(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_clases),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def crear_mobilenet(num_clases: int = 4, preentrenado: bool = True):
    """MobileNetV2 (≈2.2 M parámetros con 4 clases) con la última capa reemplazada por 4 salidas."""
    pesos = models.MobileNet_V2_Weights.IMAGENET1K_V1 if preentrenado else None
    m = models.mobilenet_v2(weights=pesos)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.last_channel, num_clases),
    )
    return m


def crear_resnet(num_clases: int = 4, preentrenado: bool = True):
    """ResNet50 (He et al., 2016): 50 capas con conexiones residuales (~23.5 M parámetros
    con 4 clases). Es el modelo 'pesado' de referencia; se recomienda entrenarlo con GPU (Colab)."""
    pesos = models.ResNet50_Weights.IMAGENET1K_V2 if preentrenado else None
    m = models.resnet50(weights=pesos)
    m.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.fc.in_features, num_clases),
    )
    return m


def crear_modelo(arquitectura: str, num_clases: int = 4, preentrenado: bool = True):
    arquitectura = arquitectura.lower()
    if arquitectura == "cnn":
        return CNNMaiz(num_clases)
    if arquitectura == "mobilenet":
        return crear_mobilenet(num_clases, preentrenado)
    if arquitectura == "resnet":
        return crear_resnet(num_clases, preentrenado)
    raise ValueError(f"Arquitectura desconocida: {arquitectura} (usa 'cnn', 'mobilenet' o 'resnet')")


def base_y_cabeza(modelo, arquitectura: str):
    """Separa los parámetros preentrenados (base) de la capa nueva (cabeza) para usar
    learning rates distintos en el fine-tuning. La CNN propia no tiene base preentrenada."""
    if arquitectura == "mobilenet":
        cabeza = modelo.classifier
    elif arquitectura == "resnet":
        cabeza = modelo.fc
    else:
        return None, list(modelo.parameters())
    ids = {id(p) for p in cabeza.parameters()}
    base = [p for p in modelo.parameters() if id(p) not in ids]
    return base, list(cabeza.parameters())


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


def cargar_checkpoint(ruta, dispositivo="cpu"):
    """Carga un .pt guardado por 2_entrenar.py y devuelve (modelo, info)."""
    ckpt = torch.load(ruta, map_location="cpu")
    modelo = crear_modelo(ckpt["arquitectura"], len(ckpt["clases"]), preentrenado=False)
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.to(dispositivo).eval()
    return modelo, ckpt
