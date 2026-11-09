"""
3_evaluar.py - Evalúa los modelos entrenados sobre el conjunto de PRUEBA (test).

Uso:
    python 3_evaluar.py                      # evalúa los 3 modelos (los que existan en modelos/)
    python 3_evaluar.py --modelo resnet      # solo uno

Salidas en resultados/:
    <modelo>_reporte.txt           classification_report (precision, recall, F1 por clase)
    <modelo>_matriz_confusion.png  matriz de confusión (absoluta y normalizada)
    <modelo>_errores.png           ejemplos mal clasificados (para la discusión)
    comparacion_modelos.csv        tabla comparativa (accuracy, F1, parámetros, tiempos)
    comparacion_modelos.png        gráfico de barras comparando los 3 modelos
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from modelos import cargar_checkpoint, contar_parametros


def latencia_cpu(modelo, x1, repeticiones=20):
    """ms por imagen en CPU con batch=1 (simula la app de escritorio en una laptop)."""
    m = modelo.to("cpu").eval()
    with torch.no_grad():
        for _ in range(3):  # calentamiento
            m(x1)
        t0 = time.time()
        for _ in range(repeticiones):
            m(x1)
    return (time.time() - t0) / repeticiones * 1000


def evaluar(nombre, dispositivo, batch):
    ruta = config.MODELOS_DIR / f"{nombre}_best.pt"
    if not ruta.exists():
        print(f"[omitido] No existe {ruta}")
        return None
    modelo, ckpt = cargar_checkpoint(ruta, dispositivo)
    clases, img = ckpt["clases"], ckpt["img_size"]

    tf = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(),
                             transforms.Normalize(config.MEAN, config.STD)])
    ds = datasets.ImageFolder(config.DATA_DIR / "test", transform=tf)
    assert ds.classes == clases, f"Clases distintas: {ds.classes} vs {clases}"
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=config.NUM_WORKERS)

    y_real, y_pred, probs = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in dl:
            p = torch.softmax(modelo(x.to(dispositivo)), dim=1).cpu()
            probs.append(p)
            y_pred += p.argmax(1).tolist()
            y_real += y.tolist()
    ms_img = (time.time() - t0) / len(ds) * 1000
    probs = torch.cat(probs).numpy()
    y_real, y_pred = np.array(y_real), np.array(y_pred)

    ms_cpu = latencia_cpu(modelo, ds[0][0].unsqueeze(0))

    # ---- Métricas -------------------------------------------------------
    acc = accuracy_score(y_real, y_pred)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_real, y_pred, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_real, y_pred, average="weighted", zero_division=0)
    reporte = classification_report(y_real, y_pred, target_names=clases, digits=4, zero_division=0)
    texto = (f"Modelo: {config.NOMBRES_MODELO.get(nombre, nombre)}  (mejor época {ckpt['epoca']}, img {img}px)\n"
             f"Imágenes de prueba: {len(ds)}\n\n{reporte}\n"
             f"Accuracy: {acc:.4f}\nPrecision macro: {p_m:.4f}\nRecall macro: {r_m:.4f}\n"
             f"F1 macro: {f_m:.4f}\nF1 ponderado: {f_w:.4f}\n"
             f"Tiempo de inferencia: {ms_img:.2f} ms/imagen en lote ({dispositivo}); "
             f"{ms_cpu:.1f} ms/imagen en CPU con batch 1\n"
             f"Parámetros: {contar_parametros(modelo):,}\n"
             f"Tiempo de entrenamiento: {ckpt.get('minutos_entrenamiento', 0):.1f} min "
             f"({ckpt.get('epocas_ejecutadas', '?')} épocas)\n")
    print("=" * 60); print(texto)
    (config.RESULTADOS_DIR / f"{nombre}_reporte.txt").write_text(texto, encoding="utf-8")

    # ---- Matriz de confusión -------------------------------------------
    mc = confusion_matrix(y_real, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ConfusionMatrixDisplay(mc, display_labels=clases).plot(ax=axes[0], cmap="Greens", colorbar=False)
    axes[0].set_title("Matriz de confusión (conteos)")
    mcn = mc.astype(float) / mc.sum(axis=1, keepdims=True)
    ConfusionMatrixDisplay(mcn, display_labels=clases).plot(ax=axes[1], cmap="Greens", colorbar=False,
                                                            values_format=".2f")
    axes[1].set_title("Matriz de confusión (normalizada por fila = recall)")
    for ax in axes:
        ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.suptitle(f"{config.NOMBRES_MODELO.get(nombre, nombre)} - Accuracy test = {acc:.4f} | F1 macro = {f_m:.4f}")
    plt.tight_layout()
    plt.savefig(config.RESULTADOS_DIR / f"{nombre}_matriz_confusion.png", dpi=120)
    plt.close()

    # ---- Ejemplos mal clasificados -------------------------------------
    errores = np.where(y_real != y_pred)[0][:12]
    if len(errores):
        cols = 4
        filas = int(np.ceil(len(errores) / cols))
        fig, axes = plt.subplots(filas, cols, figsize=(3.2 * cols, 3.4 * filas), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, idx in zip(axes.ravel(), errores):
            ax.imshow(Image.open(ds.samples[idx][0]).convert("RGB"))
            ax.set_title(f"Real: {clases[y_real[idx]]}\nPred: {clases[y_pred[idx]]} "
                         f"({probs[idx].max():.0%})", fontsize=9, color="darkred")
        plt.tight_layout()
        plt.savefig(config.RESULTADOS_DIR / f"{nombre}_errores.png", dpi=110)
        plt.close()

    return dict(modelo=config.NOMBRES_MODELO.get(nombre, nombre), accuracy=acc, precision_macro=p_m, recall_macro=r_m,
                f1_macro=f_m, f1_ponderado=f_w, parametros=contar_parametros(modelo),
                ms_por_imagen=ms_img, ms_cpu_1img=ms_cpu, epoca_mejor=ckpt["epoca"],
                epocas_ejecutadas=ckpt.get("epocas_ejecutadas", 0),
                min_entrenamiento=float(ckpt.get("minutos_entrenamiento", 0.0)), img_size=img,
                dispositivo_eval=str(dispositivo))


def graficar_comparacion(filas, ruta_png):
    """Barras agrupadas: accuracy, precision, recall y F1 (macro) por modelo."""
    metricas = [("accuracy", "Accuracy"), ("precision_macro", "Precision"),
                ("recall_macro", "Recall"), ("f1_macro", "F1-macro")]
    x = np.arange(len(metricas))
    ancho = 0.8 / len(filas)
    colores = ["#9aa5a0", "#3b7d4f", "#c9822b"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, f in enumerate(filas):
        vals = [f[k] for k, _ in metricas]
        barras = ax.bar(x + (i - (len(filas) - 1) / 2) * ancho, vals, ancho,
                        label=f["modelo"], color=colores[i % len(colores)])
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=7.5)
    minimo = min(f[k] for f in filas for k, _ in metricas)
    ax.set_ylim(max(0.0, minimo - 0.05), 1.0)
    ax.set_xticks(x); ax.set_xticklabels([m for _, m in metricas])
    ax.set_ylabel("Valor en conjunto de prueba")
    ax.set_title("Comparación de los tres modelos (test)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ruta_png, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", choices=config.ARQUITECTURAS + ["todos"], default="todos")
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    dispositivo = config.obtener_dispositivo(args.device)
    nombres = config.ARQUITECTURAS if args.modelo == "todos" else [args.modelo]
    filas = [r for r in (evaluar(n, dispositivo, args.batch) for n in nombres) if r]
    if not filas:
        raise SystemExit("No hay modelos entrenados. Ejecuta primero 2_entrenar.py")

    ruta_csv = config.RESULTADOS_DIR / "comparacion_modelos.csv"
    cab = list(filas[0].keys())
    lineas = [",".join(cab)] + [",".join(f"{f[k]:.4f}" if isinstance(f[k], float) else str(f[k])
                                         for k in cab) for f in filas]
    ruta_csv.write_text("\n".join(lineas), encoding="utf-8")
    if len(filas) > 1:
        graficar_comparacion(filas, config.RESULTADOS_DIR / "comparacion_modelos.png")
    print("\nCOMPARACIÓN DE MODELOS (conjunto de prueba)")
    print(f"{'modelo':13s}{'accuracy':>10s}{'F1 macro':>10s}{'params':>13s}{'ms CPU':>9s}{'min train':>11s}")
    for f in filas:
        print(f"{f['modelo']:13s}{f['accuracy']:10.4f}{f['f1_macro']:10.4f}"
              f"{f['parametros']:13,d}{f['ms_cpu_1img']:9.1f}{f['min_entrenamiento']:11.1f}")
    print(f"\nTabla guardada en {ruta_csv}")


if __name__ == "__main__":
    main()
