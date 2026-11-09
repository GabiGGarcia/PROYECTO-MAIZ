"""
4_app.py - Interfaz gráfica (Tkinter) para diagnosticar una hoja de maíz.

Uso:
    python 4_app.py                 # abre con MobileNetV2; el modelo se cambia desde la ventana
    python 4_app.py --modelo resnet # modelo inicial distinto (cnn | mobilenet | resnet)

Pasos para el usuario:
    1. Elegir el modelo en la lista desplegable (CNN propia, MobileNetV2 o ResNet50).
    2. "Cargar imagen" -> se muestra la hoja, la enfermedad detectada, la confianza,
       la probabilidad de cada clase y una recomendación básica.
    3. Al cambiar de modelo, la misma imagen se vuelve a evaluar automáticamente
       (útil para comparar los tres modelos en la exposición).
"""
import argparse
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from PIL import Image, ImageTk
from torchvision import transforms

import config
from modelos import cargar_checkpoint, contar_parametros

UMBRAL_CONFIANZA = 0.60  # por debajo de esto se avisa que el resultado es dudoso

# Nombres de los modelos tal como se muestran en la lista desplegable
MODELOS = {"cnn": "CNN propia", "mobilenet": "MobileNetV2", "resnet": "ResNet50"}

# Etiquetas formales para las barras de probabilidad (clave = nombre de carpeta)
ETIQUETAS = {
    "mancha_gris": "Mancha Gris",
    "roya_comun": "Roya Común",
    "sana": "Hoja Sana",
    "tizon_norte": "Tizón del Norte",
}

# Nombre completo (con agente causal) para el resultado principal
NOMBRES_COMPLETOS = {
    "mancha_gris": "Mancha gris (Cercospora zeae-maydis)",
    "roya_comun": "Roya común (Puccinia sorghi)",
    "sana": "Hoja sana",
    "tizon_norte": "Tizón foliar del norte (Exserohilum turcicum)",
}

FONDO = "#f3f7f1"
VERDE = "#24502f"


def etiqueta(clase):
    """Convierte 'tizon_norte' en 'Tizón del Norte' (o un título genérico si no está en la tabla)."""
    return ETIQUETAS.get(clase, clase.replace("_", " ").title())


class AppMaiz:
    def __init__(self, raiz, modelo_inicial):
        self.raiz = raiz
        self.dispositivo = torch.device("cpu")  # la inferencia de 1 imagen es rápida en CPU
        self.cache = {}            # modelos ya cargados: {clave: (modelo, ckpt, transform)}
        self.imagen_actual = None  # PIL.Image de la última foto cargada

        # Solo se ofrecen los modelos cuyo .pt existe en la carpeta modelos/
        self.disponibles = {k: v for k, v in MODELOS.items()
                            if (config.MODELOS_DIR / f"{k}_best.pt").exists()}
        if not self.disponibles:
            raise FileNotFoundError(f"No hay modelos .pt en {config.MODELOS_DIR}")
        if modelo_inicial not in self.disponibles:
            modelo_inicial = next(iter(self.disponibles))

        raiz.title("Detección de enfermedades en hojas de maíz")
        raiz.geometry("820x660")
        raiz.configure(bg=FONDO)

        tk.Label(raiz, text="Diagnóstico de hojas de maíz", font=("Segoe UI", 18, "bold"),
                 bg=FONDO, fg=VERDE).pack(pady=(14, 6))

        # --- Barra superior: selector de modelo + botón ---------------------
        barra = tk.Frame(raiz, bg=FONDO)
        barra.pack(pady=4)
        tk.Label(barra, text="Modelo:", font=("Segoe UI", 11), bg=FONDO).pack(side="left", padx=(0, 6))
        self.combo = ttk.Combobox(barra, state="readonly", width=16, font=("Segoe UI", 11),
                                  values=list(self.disponibles.values()))
        self.combo.set(self.disponibles[modelo_inicial])
        self.combo.bind("<<ComboboxSelected>>", self.cambiar_modelo)
        self.combo.pack(side="left", padx=(0, 16))
        tk.Button(barra, text="Cargar imagen", font=("Segoe UI", 11), bg="#3b7d4f", fg="white",
                  activebackground="#2e6340", padx=14, pady=4,
                  command=self.cargar_imagen).pack(side="left")

        self.lbl_info = tk.Label(raiz, text="", font=("Segoe UI", 9), bg=FONDO, fg="#555")
        self.lbl_info.pack(pady=(6, 8))

        # --- Cuerpo: imagen a la izquierda, resultados a la derecha ---------
        cuerpo = tk.Frame(raiz, bg=FONDO)
        cuerpo.pack(fill="both", expand=True, padx=16)
        self.lbl_imagen = tk.Label(cuerpo, bg="#dfe8dc", text="Sin imagen", fg="#777",
                                   width=40, height=18)
        self.lbl_imagen.pack(side="left", padx=8)

        panel = tk.Frame(cuerpo, bg=FONDO)
        panel.pack(side="left", fill="both", expand=True, padx=8)
        self.lbl_resultado = tk.Label(panel, text="Resultado: -", font=("Segoe UI", 14, "bold"),
                                      bg=FONDO, wraplength=370, justify="left")
        self.lbl_resultado.pack(anchor="w", pady=(4, 10))

        tk.Label(panel, text="Probabilidad por clase", font=("Segoe UI", 10, "bold"),
                 bg=FONDO, fg="#333").pack(anchor="w")
        self.marco_barras = tk.Frame(panel, bg=FONDO)
        self.marco_barras.pack(anchor="w", fill="x")
        self.barras = {}

        self.lbl_reco = tk.Label(panel, text="", font=("Segoe UI", 10), bg=FONDO,
                                 wraplength=370, justify="left", fg="#333")
        self.lbl_reco.pack(anchor="w", pady=12)

        tk.Label(raiz, text="Herramienta de apoyo: confirme el diagnóstico con un especialista.",
                 font=("Segoe UI", 8), bg=FONDO, fg="#777").pack(side="bottom", pady=6)

        self.clave = None
        self.activar_modelo(modelo_inicial)

    # ------------------------------------------------------------------ modelos
    def activar_modelo(self, clave):
        """Carga (o toma de la caché) el modelo elegido y actualiza la interfaz."""
        try:
            if clave not in self.cache:
                modelo, ckpt = cargar_checkpoint(config.MODELOS_DIR / f"{clave}_best.pt", self.dispositivo)
                img = ckpt["img_size"]
                tf = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(),
                                         transforms.Normalize(config.MEAN, config.STD)])
                self.cache[clave] = (modelo, ckpt, tf)
        except Exception as e:
            messagebox.showerror("Error al cargar el modelo",
                                 f"No se pudo cargar {MODELOS[clave]}:\n{e}\n\n"
                                 "Verifica que modelos.py esté actualizado (incluye ResNet50).")
            if self.clave:
                self.combo.set(MODELOS[self.clave])
            return

        self.clave = clave
        modelo, ckpt, _ = self.cache[clave]
        self.crear_barras(ckpt["clases"])
        self.lbl_info.config(text=f"{MODELOS[clave]}  |  F1 validación = {ckpt['val_f1']:.3f}  |  "
                                  f"{contar_parametros(modelo) / 1e6:.1f} M parámetros")
        if self.imagen_actual is not None:
            self.predecir()  # reevaluar la misma foto con el nuevo modelo

    def cambiar_modelo(self, _evento=None):
        nombre = self.combo.get()
        clave = next(k for k, v in self.disponibles.items() if v == nombre)
        if clave != self.clave:
            self.activar_modelo(clave)

    def crear_barras(self, clases):
        for w in self.marco_barras.winfo_children():
            w.destroy()
        self.barras = {}
        for c in clases:
            fila = tk.Frame(self.marco_barras, bg=FONDO)
            fila.pack(fill="x", pady=3)
            tk.Label(fila, text=etiqueta(c), width=15, anchor="w", bg=FONDO,
                     font=("Segoe UI", 10)).pack(side="left")
            barra = ttk.Progressbar(fila, length=180, maximum=100)
            barra.pack(side="left", padx=4)
            pct = tk.Label(fila, text="0%", width=6, bg=FONDO, font=("Segoe UI", 10))
            pct.pack(side="left")
            self.barras[c] = (barra, pct)

    # ------------------------------------------------------------------ imagen
    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")])
        if not ruta:
            return
        try:
            self.imagen_actual = Image.open(ruta).convert("RGB")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")
            return
        vista = self.imagen_actual.copy()
        vista.thumbnail((320, 320))
        foto = ImageTk.PhotoImage(vista)
        self.lbl_imagen.configure(image=foto, text="", width=320, height=320)
        self.lbl_imagen.image = foto
        self.predecir()

    def predecir(self):
        modelo, ckpt, tf = self.cache[self.clave]
        clases = ckpt["clases"]
        t0 = time.perf_counter()
        with torch.no_grad():
            x = tf(self.imagen_actual).unsqueeze(0).to(self.dispositivo)
            probs = torch.softmax(modelo(x), dim=1)[0].tolist()
        ms = (time.perf_counter() - t0) * 1000

        idx = max(range(len(probs)), key=lambda i: probs[i])
        clase, conf = clases[idx], probs[idx]
        color = "#2e7d32" if clase == "sana" else "#b23b2b"
        texto = f"Resultado: {NOMBRES_COMPLETOS.get(clase, etiqueta(clase))}\nConfianza: {conf:.1%}"
        if conf < UMBRAL_CONFIANZA:
            texto += "\n(Confianza baja: tome otra foto con mejor luz y fondo uniforme)"
            color = "#a06000"
        texto += f"\nModelo: {MODELOS[self.clave]}  ·  {ms:.0f} ms"
        self.lbl_resultado.config(text=texto, fg=color)

        for c, p in zip(clases, probs):
            barra, pct = self.barras[c]
            barra["value"] = p * 100
            pct.config(text=f"{p:.0%}")
        self.lbl_reco.config(text="Recomendación: " + config.RECOMENDACIONES.get(clase, ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="mobilenet", choices=list(MODELOS),
                    help="modelo con el que abre la app (luego se cambia desde la ventana)")
    args = ap.parse_args()
    try:
        raiz = tk.Tk()
        AppMaiz(raiz, args.modelo)
        raiz.mainloop()
    except FileNotFoundError as e:
        raise SystemExit(f"{e}\nEntrena primero con 2_entrenar.py o copia la carpeta modelos/ de Colab.")


if __name__ == "__main__":
    main()
