# Detección de enfermedades en hojas de maíz con CNN (PyTorch) — TA3 IA USAT

Clasifica una foto de hoja de maíz en 4 clases: **mancha_gris**, **roya_comun**, **tizon_norte** y **sana**.
Compara **tres algoritmos**:

| # | Modelo | Tipo | Parámetros (4 clases) |
|---|---|---|---|
| 1 | CNN propia | Desde cero (línea base) | ≈0,4 M |
| 2 | MobileNetV2 | Transfer learning, ligero | ≈2,2 M |
| 3 | ResNet50 | Transfer learning, profundo | ≈23,5 M |

```
maiz_ia/
├── config.py               rutas, clases, hiperparámetros, CPU/GPU, carpeta de salida
├── modelos.py              CNN propia, MobileNetV2, ResNet50
├── 1_preparar_dataset.py   filtra maíz, quita duplicados, split 70/15/15
├── 2_entrenar.py           entrenamiento (AMP en GPU) + early stopping + curvas
├── 3_evaluar.py            métricas en test, matrices de confusión, comparación de los 3
├── 4_app.py                interfaz gráfica Tkinter (se ejecuta en la laptop)
├── Colab_Maiz_CNN.ipynb    ★ flujo completo en Google Colab con GPU (recomendado)
├── Notebook_Maiz_CNN.ipynb flujo local (para exponer en la laptop)
└── requirements.txt
```

---

## Opción A (recomendada): entrenar en Google Colab con GPU gratis

Tu PC queda libre. Los tres modelos se entrenan en unos 25–40 min en total.

1. Sube **`proyecto_maiz_ia.zip`** (sin descomprimir) a la raíz de tu **Google Drive**.
2. Abre <https://colab.research.google.com> → *Archivo → Subir notebook* → elige `Colab_Maiz_CNN.ipynb`.
3. Menú *Entorno de ejecución → Cambiar tipo de entorno de ejecución* → **GPU T4** → Guardar.
4. Ejecuta las celdas en orden. El notebook:
   - verifica la GPU y conecta Drive;
   - descomprime el código y descarga PlantVillage directo de Kaggle (`kagglehub`, no hace falta bajarlo a tu PC);
   - prepara los datos, entrena los 3 modelos, evalúa y compara;
   - permite probar con tus propias fotos;
   - guarda todo en **`MyDrive/maiz_ia_salida/`** (`modelos/` y `resultados/`), así no se pierde si se cae la sesión.
5. Para la app en tu laptop: copia `MyDrive/maiz_ia_salida/modelos/` dentro de tu carpeta `maiz_ia/` y ejecuta:
   ```bat
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install pillow
   python 4_app.py --modelo mobilenet
   ```
   La inferencia de una imagen en CPU tarda milisegundos; no hace falta GPU.

**Tiempos aproximados en Colab T4** (≈2 700 imágenes de entrenamiento, 224 px):
| Modelo | Por época | Total con early stopping |
|---|---|---|
| CNN propia | 15–25 s | 4–8 min |
| MobileNetV2 | 20–30 s | 5–8 min |
| ResNet50 | 45–70 s | 10–15 min |

**Problemas frecuentes en Colab**
| Problema | Solución |
|---|---|
| `AssertionError: No hay GPU` | Activa GPU T4 (paso 3) y reinicia el entorno |
| `kagglehub` pide credenciales | Usa la celda alternativa con `kaggle.json` (Kaggle → Settings → API → Create New Token) |
| "No se encontró el zip en Drive" | Súbelo cuando la celda lo pida, o revisa que se llame `proyecto_maiz_ia.zip` |
| Se desconectó la sesión | Vuelve a ejecutar las celdas 0–4; los modelos ya entrenados siguen en Drive. Entrena solo los que falten |
| Límite de uso de GPU | Espera unas horas o usa otra cuenta de Google; también sirve un Notebook de Kaggle con GPU T4 |

---

## Opción B: ejecutar todo en la laptop (Windows, CPU)

```bat
cd C:\ruta\a\maiz_ia
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib numpy pillow jupyter pandas
```
Descarga de Kaggle **PlantVillage Dataset** (`abdallahalidev/plantvillage-dataset`) y descomprímelo. Solo se usan las 4 carpetas `color/Corn_(maize)___*` (≈3 852 imágenes).

```bat
python 1_preparar_dataset.py --origen "C:\Users\TU_USUARIO\Downloads\plantvillage-dataset"
python 2_entrenar.py --modelo mobilenet --rapido      :: prueba de 1-2 min
python 2_entrenar.py --modelo cnn
python 2_entrenar.py --modelo mobilenet --img 160
python 2_entrenar.py --modelo resnet --img 160        :: lento en CPU (1-3 h)
python 3_evaluar.py
python 4_app.py
```
**No uses** "New Plant Diseases Dataset (Augmented)": trae imágenes ya aumentadas y duplicadas, y eso infla las métricas.

---

## Qué produce cada paso 

| Archivo en `resultados/` | Sección del informe |
|---|---|
| `distribucion_dataset.png/.csv`, `muestras_dataset.png` | 3.2 Descripción del dataset |
| `cnn_curvas.png`, `mobilenet_curvas.png`, `resnet_curvas.png`, `*_historial.csv` | 3.4 Entrenamiento |
| `*_reporte.txt`, `*_matriz_confusion.png` | 3.5 Evaluación del modelo |
| `comparacion_modelos.csv`, `comparacion_modelos.png` | 3.5 y 4. Discusión (tabla y gráfico comparativo) |
| `*_errores.png` | 4. Discusión (análisis de errores) |
| Captura de pantalla de `4_app.py` | 3.6 Aplicación |

`comparacion_modelos.csv` incluye accuracy, precision, recall, F1 (macro y ponderado), parámetros, minutos de entrenamiento, épocas y **ms por imagen en CPU** (la latencia real en la laptop, aunque evalúes en Colab).

## Decisiones técnicas 
- **Tres modelos con complejidad creciente** (0,4 M → 2,2 M → 23,5 M parámetros): muestran el equilibrio entre precisión y costo computacional.
- **Split estratificado 70/15/15 con semilla fija** y **deduplicación MD5**: evita que la misma imagen quede en train y en test.
- **Aumento de datos** (recorte, giros, rotación, brillo/contraste) solo en train.
- **Pérdida ponderada por clase**: *mancha_gris* tiene ~513 imágenes frente a ~1 190 de *roya_comun*.
- **Early stopping por F1-macro en validación**: se guarda la mejor época (equivale a `restore_best_weights` de Keras).
- **Fine-tuning con learning rate diferencial**: 1e-4 en la base preentrenada y 1e-3 en la capa nueva (MobileNetV2 y ResNet50).
- **Precisión mixta (AMP)** en GPU: entrenamiento ~2× más rápido en Colab sin perder exactitud.
- **Métricas con scikit-learn**: accuracy, precision, recall, F1 y matriz de confusión.
