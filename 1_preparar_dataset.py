"""
1_preparar_dataset.py - Organiza las imágenes de maíz de PlantVillage en train/val/test.

Uso (Windows, desde la carpeta del proyecto):
    python 1_preparar_dataset.py --origen "C:/Users/TU_USUARIO/Downloads/plantvillage-dataset"

Qué hace:
  1. Busca dentro de --origen las carpetas de maíz (Corn_(maize)___...). Si no hay carpetas
     "Corn/maize", asume que --origen ya contiene solo carpetas de maíz (p. ej. dataset
     "Corn or Maize Leaf Disease": Blight, Common_Rust, Gray_Leaf_Spot, Healthy).
  2. Ignora las versiones 'grayscale' y 'segmented' (solo usa 'color').
  3. Elimina imágenes duplicadas (hash MD5) para evitar fuga de datos entre splits.
  4. Divide de forma estratificada 70 % train / 15 % val / 15 % test (semilla fija).
  5. Copia a data/maiz/{train,val,test}/{clase}/ y guarda la distribución y una muestra.
"""
import argparse
import hashlib
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split

import config

# palabra clave en el nombre de carpeta -> clase destino
PALABRAS_CLAVE = [
    (("cercospora", "gray_leaf", "gray leaf", "grey", "gris"), "mancha_gris"),
    (("rust", "roya"), "roya_comun"),
    (("blight", "tizon"), "tizon_norte"),
    (("healthy", "sana", "sano"), "sana"),
]


def clase_de_carpeta(nombre: str):
    n = nombre.lower()
    for claves, clase in PALABRAS_CLAVE:
        if any(c in n for c in claves):
            return clase
    return None


def es_imagen(p: Path):
    return p.suffix.lower() in config.EXTENSIONES


def listar_carpetas(origen: Path):
    """Recorre SOLO carpetas (os.walk con poda). No entra a 'grayscale'/'segmented' ni a las
    carpetas de otros cultivos (p. ej. 'Tomato___...'), así evita revisar ~160 000 imágenes."""
    import os
    carpetas = []
    for raiz, subdirs, _ in os.walk(origen):
        conservar = []
        for d in subdirs:
            n = d.lower()
            if n in ("grayscale", "segmented"):
                continue
            if "___" in d and not ("corn" in n or "maize" in n):
                continue  # carpeta de clase de otro cultivo: no se recorre
            conservar.append(d)
        subdirs[:] = conservar
        carpetas.append(Path(raiz))
    return carpetas


def buscar_carpetas(origen: Path):
    """Devuelve {clase: [lista de imágenes]} detectando carpetas de maíz."""
    carpetas = listar_carpetas(origen)
    hay_maiz = any(("corn" in d.name.lower() or "maize" in d.name.lower()) for d in carpetas)

    archivos = defaultdict(list)
    for d in carpetas:
        if hay_maiz and not ("corn" in d.name.lower() or "maize" in d.name.lower()):
            continue
        clase = clase_de_carpeta(d.name)
        if clase is None:
            continue
        imgs = [p for p in d.iterdir() if p.is_file() and es_imagen(p)]
        if imgs:
            print(f"  [{clase:12s}] {len(imgs):5d} imágenes  <- {d}")
            archivos[clase].extend(imgs)
    return archivos


def md5(p: Path):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


def quitar_duplicados(archivos):
    vistos, limpio, dup = set(), defaultdict(list), 0
    for clase, lista in archivos.items():
        for p in lista:
            h = md5(p)
            if h in vistos:
                dup += 1
                continue
            vistos.add(h)
            limpio[clase].append(p)
    print(f"Duplicados eliminados: {dup}")
    return limpio


def guardar_muestras(splits, destino_png):
    clases = sorted(splits["train"].keys())
    fig, axes = plt.subplots(len(clases), 5, figsize=(12, 2.6 * len(clases)))
    for i, c in enumerate(clases):
        for j, p in enumerate(splits["train"][c][:5]):
            ax = axes[i, j]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.axis("off")
            if j == 0:
                ax.set_title(c, loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(destino_png, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", required=True, help="Carpeta donde descomprimiste el dataset de Kaggle")
    ap.add_argument("--destino", default=str(config.DATA_DIR))
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    origen, destino = Path(args.origen), Path(args.destino)
    if not origen.exists():
        raise SystemExit(f"No existe la carpeta de origen: {origen}")

    print("Buscando carpetas de maíz...")
    archivos = buscar_carpetas(origen)
    if len(archivos) < 2:
        raise SystemExit("No se encontraron carpetas de maíz. Revisa la ruta --origen.")
    archivos = quitar_duplicados(archivos)

    # División estratificada
    rutas, etiquetas = [], []
    for c, lista in archivos.items():
        rutas += lista
        etiquetas += [c] * len(lista)
    r_train, r_tmp, y_train, y_tmp = train_test_split(
        rutas, etiquetas, test_size=args.val + args.test, stratify=etiquetas, random_state=args.seed)
    r_val, r_test, y_val, y_test = train_test_split(
        r_tmp, y_tmp, test_size=args.test / (args.val + args.test), stratify=y_tmp, random_state=args.seed)

    splits = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
    for nombre, rs, ys in (("train", r_train, y_train), ("val", r_val, y_val), ("test", r_test, y_test)):
        for r, y in zip(rs, ys):
            splits[nombre][y].append(r)

    if destino.exists():
        print(f"Borrando destino anterior: {destino}")
        shutil.rmtree(destino)
    for nombre, por_clase in splits.items():
        for c, lista in por_clase.items():
            carpeta = destino / nombre / c
            carpeta.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(lista):
                shutil.copy2(p, carpeta / f"{c}_{i:05d}{p.suffix.lower()}")

    # Tabla de distribución
    clases = sorted(archivos.keys())
    lineas = ["clase,train,val,test,total"]
    print(f"\n{'clase':14s}{'train':>7s}{'val':>7s}{'test':>7s}{'total':>7s}")
    for c in clases:
        n = [len(splits[s][c]) for s in ("train", "val", "test")]
        print(f"{c:14s}{n[0]:7d}{n[1]:7d}{n[2]:7d}{sum(n):7d}")
        lineas.append(f"{c},{n[0]},{n[1]},{n[2]},{sum(n)}")
    tot = [len(r_train), len(r_val), len(r_test)]
    print(f"{'TOTAL':14s}{tot[0]:7d}{tot[1]:7d}{tot[2]:7d}{sum(tot):7d}")
    lineas.append(f"TOTAL,{tot[0]},{tot[1]},{tot[2]},{sum(tot)}")
    (config.RESULTADOS_DIR / "distribucion_dataset.csv").write_text("\n".join(lineas), encoding="utf-8")

    # Gráfico de barras de la distribución
    cuenta = Counter(etiquetas)
    plt.figure(figsize=(7, 4))
    plt.bar(clases, [cuenta[c] for c in clases], color="#3b7d4f")
    for i, c in enumerate(clases):
        plt.text(i, cuenta[c], str(cuenta[c]), ha="center", va="bottom")
    plt.title("Distribución de imágenes por clase (maíz - PlantVillage)")
    plt.ylabel("N.° de imágenes")
    plt.tight_layout()
    plt.savefig(config.RESULTADOS_DIR / "distribucion_dataset.png", dpi=120)
    plt.close()

    guardar_muestras(splits, config.RESULTADOS_DIR / "muestras_dataset.png")
    print(f"\nListo. Dataset en: {destino}")
    print(f"Gráficas en: {config.RESULTADOS_DIR}")


if __name__ == "__main__":
    main()
