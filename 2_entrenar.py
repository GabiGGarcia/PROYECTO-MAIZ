"""
2_entrenar.py - Entrena una CNN para clasificar enfermedades de hojas de maíz.

Ejemplos:
    python 2_entrenar.py --modelo cnn                    # 1) CNN propia desde cero (línea base)
    python 2_entrenar.py --modelo mobilenet              # 2) MobileNetV2 (transfer learning)
    python 2_entrenar.py --modelo resnet                 # 3) ResNet50 (transfer learning, usar GPU)
    python 2_entrenar.py --modelo todos                  # los tres seguidos (ideal en Colab)
    python 2_entrenar.py --modelo mobilenet --rapido     # prueba de humo (1-2 min)
    python 2_entrenar.py --modelo mobilenet --img 160    # más rápido en CPU

Salidas (en MAIZ_SALIDA si está definida, si no en la carpeta del proyecto):
    modelos/<modelo>_best.pt              mejores pesos (según F1-macro en validación)
    resultados/<modelo>_historial.csv     métricas por época
    resultados/<modelo>_curvas.png        curvas de pérdida y accuracy
"""
import argparse
import csv
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from modelos import base_y_cabeza, contar_parametros, crear_modelo

EPOCAS_POR_DEFECTO = {"cnn": 30, "mobilenet": 15, "resnet": 12}


def fijar_semilla(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transformaciones(img_size):
    """Aumento de datos solo en entrenamiento; val/test solo redimensionan y normalizan."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])
    return train_tf, eval_tf


def crear_loaders(img_size, batch_size):
    train_tf, eval_tf = transformaciones(img_size)
    ds_train = datasets.ImageFolder(config.DATA_DIR / "train", transform=train_tf)
    ds_val = datasets.ImageFolder(config.DATA_DIR / "val", transform=eval_tf)
    gpu = torch.cuda.is_available()
    extra = {"persistent_workers": True} if config.NUM_WORKERS > 0 else {}
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=config.NUM_WORKERS, pin_memory=gpu, **extra)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=gpu, **extra)
    return ds_train, dl_train, dl_val


def pesos_de_clase(ds_train, dispositivo):
    """w_c = N / (K * n_c): compensa el desbalance (mancha gris tiene menos imágenes)."""
    conteo = np.bincount(ds_train.targets, minlength=len(ds_train.classes))
    w = conteo.sum() / (len(conteo) * conteo)
    print("Imágenes por clase (train):", dict(zip(ds_train.classes, conteo.tolist())))
    print("Pesos de clase:", dict(zip(ds_train.classes, np.round(w, 3).tolist())))
    return torch.tensor(w, dtype=torch.float32, device=dispositivo)


def crear_scaler(usar_amp):
    """GradScaler para precisión mixta (solo GPU NVIDIA). Compatible con varias versiones de torch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=usar_amp)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=usar_amp)


def ejecutar_epoca(modelo, loader, criterio, dispositivo, optimizador=None, scaler=None,
                   usar_amp=False, max_batches=None):
    """Una pasada por el loader. Si hay optimizador entrena; si no, evalúa."""
    entrenando = optimizador is not None
    modelo.train(entrenando)
    perdida_total, y_real, y_pred = 0.0, [], []
    with torch.set_grad_enabled(entrenando):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = x.to(dispositivo, non_blocking=True)
            y = y.to(dispositivo, non_blocking=True)
            # Precisión mixta (float16) en GPU: ~2x más rápido en la T4 de Colab
            with torch.autocast(device_type="cuda" if usar_amp else "cpu", dtype=torch.float16, enabled=usar_amp):
                salida = modelo(x)
                perdida = criterio(salida, y)
            if entrenando:
                optimizador.zero_grad(set_to_none=True)
                scaler.scale(perdida).backward()
                scaler.step(optimizador)
                scaler.update()
            perdida_total += perdida.item() * x.size(0)
            y_real += y.cpu().tolist()
            y_pred += salida.argmax(1).cpu().tolist()
    n = max(1, len(y_real))
    return (perdida_total / n,
            accuracy_score(y_real, y_pred),
            f1_score(y_real, y_pred, average="macro", zero_division=0))


def graficar(historial, ruta_png, titulo):
    ep = [h["epoca"] for h in historial]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, [h["train_loss"] for h in historial], label="Entrenamiento")
    a1.plot(ep, [h["val_loss"] for h in historial], label="Validación", color="red")
    a1.set_title("Pérdida (Cross-Entropy)"); a1.set_xlabel("Época"); a1.legend()
    a2.plot(ep, [h["train_acc"] for h in historial], label="Entrenamiento")
    a2.plot(ep, [h["val_acc"] for h in historial], label="Validación", color="red")
    a2.set_title("Accuracy"); a2.set_xlabel("Época"); a2.legend()
    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=120)
    plt.close()


def entrenar(arquitectura, args, dispositivo):
    fijar_semilla(config.SEED)
    epocas = args.epocas or EPOCAS_POR_DEFECTO[arquitectura]
    max_batches = 5 if args.rapido else None
    if args.rapido:
        epocas = 2
    usar_amp = dispositivo.type == "cuda"
    nombre = config.NOMBRES_MODELO[arquitectura]
    print("=" * 70)
    print(f"Modelo: {nombre} | Dispositivo: {dispositivo} | AMP: {usar_amp} | "
          f"Épocas máx: {epocas} | Imagen: {args.img}px | Batch: {args.batch}")

    ds_train, dl_train, dl_val = crear_loaders(args.img, args.batch)
    clases = ds_train.classes
    modelo = crear_modelo(arquitectura, len(clases)).to(dispositivo)
    print(f"Parámetros del modelo: {contar_parametros(modelo):,}")

    criterio = nn.CrossEntropyLoss(weight=pesos_de_clase(ds_train, dispositivo))

    # Transfer learning: LR 10x menor para la base preentrenada, normal para la cabeza nueva
    base, cabeza = base_y_cabeza(modelo, arquitectura)
    if base is None:
        grupos = [{"params": cabeza, "lr": args.lr}]
    else:
        grupos = [{"params": base, "lr": args.lr / 10}, {"params": cabeza, "lr": args.lr}]
    optimizador = torch.optim.AdamW(grupos, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, mode="max", factor=0.3, patience=2)
    scaler = crear_scaler(usar_amp)

    ruta_modelo = config.MODELOS_DIR / f"{arquitectura}_best.pt"
    historial, mejor_f1, sin_mejora = [], -1.0, 0
    t_inicio = time.time()

    for ep in range(1, epocas + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = ejecutar_epoca(modelo, dl_train, criterio, dispositivo,
                                                optimizador, scaler, usar_amp, max_batches)
        va_loss, va_acc, va_f1 = ejecutar_epoca(modelo, dl_val, criterio, dispositivo,
                                                None, None, usar_amp, max_batches)
        scheduler.step(va_f1)
        historial.append(dict(epoca=ep, train_loss=tr_loss, train_acc=tr_acc, train_f1=tr_f1,
                              val_loss=va_loss, val_acc=va_acc, val_f1=va_f1,
                              lr=optimizador.param_groups[-1]["lr"], seg=time.time() - t0))
        print(f"Época {ep:02d}/{epocas} | loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {va_loss:.4f} val_acc {va_acc:.4f} val_f1 {va_f1:.4f} | {time.time() - t0:.0f}s")

        if va_f1 > mejor_f1:  # guardar el mejor modelo (equivale a restore_best_weights)
            mejor_f1, sin_mejora = va_f1, 0
            torch.save({"arquitectura": arquitectura, "clases": clases, "img_size": args.img,
                        "state_dict": modelo.state_dict(), "epoca": ep, "val_f1": va_f1,
                        "val_acc": va_acc, "minutos_entrenamiento": 0.0}, ruta_modelo)
            print(f"   -> mejor modelo guardado (val_f1={va_f1:.4f})")
        else:
            sin_mejora += 1
            if sin_mejora >= args.paciencia:
                print(f"Early stopping: {args.paciencia} épocas sin mejorar.")
                break

    minutos = (time.time() - t_inicio) / 60
    # registrar el tiempo total de entrenamiento dentro del checkpoint (para la tabla comparativa)
    ckpt = torch.load(ruta_modelo, map_location="cpu")
    ckpt["minutos_entrenamiento"] = minutos
    ckpt["epocas_ejecutadas"] = len(historial)
    torch.save(ckpt, ruta_modelo)

    with open(config.RESULTADOS_DIR / f"{arquitectura}_historial.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=historial[0].keys())
        w.writeheader(); w.writerows(historial)
    graficar(historial, config.RESULTADOS_DIR / f"{arquitectura}_curvas.png",
             f"Entrenamiento {nombre} (mejor val F1 = {mejor_f1:.4f})")
    print(f"\n{nombre}: terminado en {minutos:.1f} min. Mejor val F1-macro: {mejor_f1:.4f}")
    print(f"Modelo guardado en: {ruta_modelo}")

    del modelo, optimizador
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=config.ARQUITECTURAS + ["todos"], default="mobilenet")
    ap.add_argument("--epocas", type=int, default=None, help="por defecto: cnn 30 / mobilenet 15 / resnet 12")
    ap.add_argument("--paciencia", type=int, default=5, help="early stopping")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--img", type=int, default=config.IMG_SIZE)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | dml")
    ap.add_argument("--rapido", action="store_true", help="prueba rápida: 2 épocas x 5 lotes")
    args = ap.parse_args()

    dispositivo = config.obtener_dispositivo(args.device)
    if dispositivo.type == "cuda":
        torch.backends.cudnn.benchmark = True  # elige los algoritmos de convolución más rápidos
        print("GPU:", torch.cuda.get_device_name(0))
    elif args.modelo in ("resnet", "todos"):
        print("[AVISO] ResNet50 en CPU es lento (~3-4x MobileNetV2). Se recomienda Google Colab con GPU.")

    arquitecturas = config.ARQUITECTURAS if args.modelo == "todos" else [args.modelo]
    for arq in arquitecturas:
        entrenar(arq, args, dispositivo)
    print("\nSiguiente paso: python 3_evaluar.py")


if __name__ == "__main__":
    main()
