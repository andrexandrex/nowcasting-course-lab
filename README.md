# Laboratorio de Nowcasting de Precipitacion

Este material esta pensado para estudiantes que no necesariamente programan todos los dias. La idea es ejecutar un flujo pequeno y reproducible para entender un problema central del nowcasting de precipitacion:

> un pronostico puede tener buen error promedio, pero fallar justo donde mas importa: lluvia intensa, nucleos localizados y desplazamiento espacial.

El laboratorio usa 5 secuencias de radar IDEAM en formato `.npy`, cada una con forma `(25, 128, 128)`. Usaremos los primeros 13 cuadros como contexto y los ultimos 12 cuadros como futuro a pronosticar.

## Estructura del curso

```text
nowcasting_course_lab/
├── README.md
├── environment.yml
├── inference_requirements.txt
├── training_requirements.txt
├── config/
│   └── model_demo_paths.yaml
├── course_utils/
│   ├── data.py
│   ├── metrics.py
│   ├── model_inference.py
│   ├── palette.py
│   └── plotting.py
├── cascast_vendor/          # dependencias minimas de CasCast, ya incluidas en el repo
│   ├── configs/sevir_used/
│   ├── models/
│   ├── networks/
│   └── utils/
├── earthformer_training_lab/
│   ├── architectures/
│   │   └── earthformer_xy.py
│   ├── config/
│   │   └── earthformer_ideam_course.yaml
│   ├── data.py
│   ├── model.py
│   └── train.py
├── scripts/
│   ├── download_assets.py
│   ├── evaluate_predictions.py
│   ├── make_persistence_predictions.py
│   ├── run_inference_from_list.py
│   ├── sequences_example.txt
│   ├── run_model_inference.py
│   └── train_earthformer_course.py
├── notebooks/
│   ├── 01_walkthrough_nowcasting_practico.ipynb
│   ├── 02_tarea_nowcasting_metricas.ipynb
│   ├── 03_comparacion_modelos_post_inferencia.ipynb
│   ├── 04_tarea_mini_proyecto_nowcasting_avanzado.ipynb
│   ├── 05_tarea_piura_sophy_comparacion_interpretacion.ipynb
│   ├── 06_entrenamiento_earthformer_paso_a_paso.ipynb
│   └── nowcasting_metrics_lab.ipynb
├── data/
│   └── samples/
├── outputs/
│   └── predictions/
│       ├── persistence/
│       ├── earthformer/
│       ├── cascast/
│       └── model/
└── checkpoints/
```

## 1. Instalar Python de forma sencilla

La opcion recomendada es instalar Miniconda o Mambaforge. Si ya tienes Anaconda, Miniconda, Mambaforge o Python funcionando, puedes saltar esta parte.

Opcion recomendada:

1. Descarga Miniconda desde <https://docs.conda.io/en/latest/miniconda.html>.
2. Instala Miniconda con las opciones por defecto.
3. Abre una terminal nueva.

En Linux o macOS, la terminal suele llamarse `Terminal`. En Windows, puedes usar `Anaconda Prompt`.

## 2. Crear el ambiente del curso

Desde la carpeta `nowcasting_course_lab`, ejecuta:

```bash
conda env create -f environment.yml
conda activate nowcasting-course-lab
python -m ipykernel install --user --name nowcasting-course-lab --display-name "Nowcasting Course Lab"
```

Si el ambiente ya existe y quieres actualizarlo:

```bash
conda env update -f environment.yml --prune
conda activate nowcasting-course-lab
```

Este ambiente es liviano y sirve para visualizacion, persistencia y metricas. No instala PyTorch ni CUDA.

## 2b. Extras opcionales para inferencia

No necesitas crear otro ambiente. Usa el mismo `nowcasting-course-lab` e instala extras solo si vas a correr EarthFormer/CasCast.

> **IMPORTANTE — instala PyTorch con `conda`, no con `pip`.**
> La version de pip de PyTorch genera un error silencioso al importar
> (`undefined symbol: iJIT_NotifyEvent`) que el curso reporta como
> "PyTorch no esta instalado en este ambiente". Si ya corriste
> `pip install torch` antes, desinstalalo primero:
> ```bash
> pip uninstall -y torch torchvision
> ```

```bash
conda activate nowcasting-course-lab

# Opcion A: PyTorch con CUDA 12.x, recomendado si tienes GPU NVIDIA.
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1

# Opcion B: PyTorch CPU-only, mas lento pero funciona sin GPU.
# conda install -c pytorch pytorch torchvision cpuonly

# Paquetes extra para la inferencia (einops, timm).
pip install -r inference_requirements.txt
```

Si ya instalaste otro kernel por accidente, no pasa nada. Puedes seguir usando `nowcasting-course-lab` en Jupyter:

```bash
python -m ipykernel install --user --name nowcasting-course-lab --display-name "Nowcasting Course Lab"
```

Para verificar que PyTorch quedo bien instalado:

```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

### Forma recomendada: una lista de secuencias en un `.txt`

El script `scripts/run_inference_from_list.py` lee un archivo de texto con **una ruta `.npy`
por linea** (formato en `scripts/sequences_example.txt`). Cada archivo debe tener forma
`(25, 128, 128)` en mm/h. Las rutas pueden ser absolutas o relativas a la carpeta del curso.
Las predicciones se guardan por el *stem* de cada archivo en
`outputs/predictions/{persistence,earthformer,cascast}/`.

```bash
# Solo EarthFormer (rapido, funciona en CPU):
python scripts/run_inference_from_list.py scripts/sequences_example.txt --stage earthformer

# EarthFormer + CasCast (difusion). En GPU:
python scripts/run_inference_from_list.py scripts/sequences_example.txt --stage all --ddim-steps 20 --device auto

# Si solo tienes CPU, empieza con pocos pasos para probar:
python scripts/run_inference_from_list.py scripts/sequences_example.txt --stage all --device cpu --ddim-steps 2 --cpu-threads 8
```

Para tus propios casos, copia `scripts/sequences_example.txt`, pon tus rutas y pasalo al script.
CasCast usa por defecto 1 miembro de ensamble; subelo con `--ens-members 10` para una media mas
suave (mucho mas lento). El factor de escala latente sale de `config/model_demo_paths.yaml`.

### Forma alternativa: solo las 5 muestras del curso

`scripts/run_model_inference.py` corre la inferencia unicamente sobre las muestras de
`data/samples/`. `--device auto` usa NVIDIA si existe y CPU si no:

```bash
python scripts/run_model_inference.py --stage earthformer --smoke --device auto
python scripts/run_model_inference.py --stage all --sample all --ddim-steps 20 --device auto
```

Nota: EarthFormer pesa alrededor de 100 MB y deberia ser razonable en CPU. CasCast pesa varios GB y puede caber en RAM, pero la inferencia por difusion es lenta en CPU porque ejecuta muchos pasos de denoising. Para CPU, usa primero `--ddim-steps 2` o `--ddim-steps 5`.

Si ves un aviso como `NVIDIA driver on your system is too old`, el script usara CPU con `--device auto`. Para forzar CPU y evitar confusiones:

```bash
python scripts/run_model_inference.py --stage earthformer --smoke --device cpu
```

Si ves un error relacionado con `hf_cache_home` de `huggingface_hub`, el wrapper del curso ya incluye una compatibilidad para el `diffusers` antiguo que viene dentro de CasCast. Actualiza este repo/archivo y vuelve a ejecutar el comando; no deberia ser necesario bajar de version `huggingface_hub`.

Las predicciones se guardan en:

```text
outputs/predictions/earthformer/
outputs/predictions/cascast/
```

Despues de generar predicciones, ejecuta la evaluacion comparativa:

```bash
python scripts/evaluate_predictions.py --sample all
```

Esto crea:

```text
outputs/evaluation/per_lead_continuous_metrics.csv
outputs/evaluation/per_lead_event_metrics.csv
outputs/evaluation/per_file_summary.csv
outputs/evaluation/overall_summary.csv
outputs/evaluation/figures/
```

## 2c. Extras opcionales para entrenamiento EarthFormer

No necesitas crear otro ambiente. Usa el mismo `nowcasting-course-lab` e instala los extras de entrenamiento encima:

```bash
conda activate nowcasting-course-lab
pip install -r training_requirements.txt
```

Este extra se usa en:

```text
notebooks/06_entrenamiento_earthformer_paso_a_paso.ipynb
scripts/train_earthformer_course.py
```

El notebook 06 reconstruye el entrenamiento de EarthFormer de forma pedagogica: dataset, split, YAML, arquitectura, forward pass, perdida, loop de entrenamiento y checkpoint `.pth`.

Para una prueba rapida:

```bash
python scripts/download_assets.py
python scripts/train_earthformer_course.py --smoke --device auto
```

Para el ejemplo corto de 5 epocas:

```bash
python scripts/train_earthformer_course.py --epochs 5 --device auto
```

Los checkpoints se guardan en:

```text
outputs/training/earthformer_course/checkpoints/earthformer/
```

Nota: este entrenamiento con 5 muestras es didactico. Sirve para explicar el flujo y verificar que se puede guardar un `.pth`, pero no reemplaza un entrenamiento cientifico completo con miles de secuencias.

## 3. Descargar datos y checkpoints desde Hugging Face

> **Hazlo temprano.** El checkpoint de difusion pesa ~3.9 GB. Conviene lanzar la descarga en
> una terminal y, mientras baja, instalar en otra terminal las librerias de inferencia (seccion 2b).
> Solo necesitas `huggingface_hub`, que ya viene en el ambiente del curso.

Todo se descarga con un solo script, `scripts/download_assets.py`. Las fuentes son:

- Datos (dataset): <https://huggingface.co/datasets/andrexandrex322/ideam-nowcasting-samples>
- Checkpoints (modelo): <https://huggingface.co/andrexandrex322/ideam-nowcasting-earthformer-cascast>

```bash
conda activate nowcasting-course-lab

# Minimo para los notebooks 01 y 03: las 5 muestras del curso (-> data/samples/).
python scripts/download_assets.py

# Checkpoints EarthFormer + Autoencoder + CasCast (~4.7 GB).
python scripts/download_assets.py --checkpoints

# Datos de la tarea 02 (Piura y Sophy).
python scripts/download_assets.py --piura --sophy

# Dataset completo de IDEAM, incluido training_dataset/ (para el notebook 06).
python scripts/download_assets.py --ideam

# Todo de una vez (datos + checkpoints, pesado).
python scripts/download_assets.py --all
```

El script organiza las descargas asi:

```text
data/samples/                 # 5 secuencias del curso (siempre)
data/ideam_data/              # 5 casos + ideam_data/training_dataset/ con --ideam
data/piura_data/              # casos Piura con --piura
data/sophy_data/              # casos Sophy con --sophy
checkpoints/ef_ideam_final/   # ef_ckpt.pth, ae_ckpt.pth, diff_ckpt.pth con --checkpoints
```

Las 5 muestras del curso se validan automaticamente con forma `(25, 128, 128)`. Los checkpoints
se descargan a `checkpoints/ef_ideam_final/` con los nombres `ef_ckpt.pth`, `ae_ckpt.pth` y
`diff_ckpt.pth`, que es exactamente donde los busca `config/model_demo_paths.yaml`. No hace falta
configurar nada mas: la inferencia de los notebooks 01 y 03 los usa desde ahi.

## 4. Crear un pronostico base de persistencia

Antes de usar modelos profundos, usaremos un pronostico muy simple:

> persistencia = repetir el ultimo cuadro observado 12 veces.

Este pronostico base no es sofisticado, pero es excelente para aprender metricas porque permite comparar desplazamiento, suavizado y errores por intensidad.

Ejecuta:

```bash
python scripts/make_persistence_predictions.py
```

Los resultados quedaran en:

```text
outputs/predictions/persistence/
```

## 5. Abrir el notebook

Ejecuta:

```bash
jupyter lab
```

Luego abre:

```text
notebooks/01_walkthrough_nowcasting_practico.ipynb
```

Selecciona el kernel `Nowcasting Course Lab` si Jupyter lo solicita.

## 6. Orden recomendado de notebooks

1. `notebooks/01_walkthrough_nowcasting_practico.ipynb`
   - **Paso 1 del flujo: generar y mirar predicciones.**
   - Carga un caso, lo separa en contexto/futuro, construye persistencia y la visualiza.
   - Explica como generar EarthFormer y CasCast con `scripts/run_inference_from_list.py`.
   - No calcula metricas: solo genera y visualiza. Las metricas viven en el notebook 03.

2. `notebooks/02_tarea_nowcasting_metricas.ipynb`
   - Tarea guiada para estudiantes.
   - Tiene formulas, pistas y celdas `TODO` para completar.
   - No depende de `course_utils.metrics`: los estudiantes implementan sus propias funciones y luego construyen un pipeline por archivo.

3. `notebooks/nowcasting_metrics_lab.ipynb`
   - Version resuelta o notebook de referencia para instructor.
   - Usa las mismas utilidades compartidas que los otros notebooks.

4. `notebooks/03_comparacion_modelos_post_inferencia.ipynb`
   - **Paso 2 del flujo: medir y comparar.**
   - Se usa despues de generar predicciones (notebook 01 o `run_inference_from_list.py`).
   - Define las metricas con formulas (MAE, RMSE, sesgo, Pearson, CSI, POD, FAR, F1) y las
     calcula **una sola vez** con `evaluate_predictions`.
   - Compara persistencia, EarthFormer y CasCast: tablas por archivo/modelo, curvas por tiempo
     de pronostico, fidelidad RMSE vs CSI y paneles visuales.

5. `notebooks/04_tarea_mini_proyecto_nowcasting_avanzado.ipynb`
   - Mini-proyecto de 10 horas.
   - Incluye auditoria del dataset, comparacion de modelos, pooling espacial, metricas de objetos, extremos, baselines opcionales e incertidumbre.
   - Material de profundizacion opcional, no evaluado en el modulo corto.

6. `notebooks/05_tarea_piura_sophy_comparacion_interpretacion.ipynb`
   - Segunda tarea evaluada.
   - Usa unicamente datos Sophy/Piura.
   - Continua la tarea `02` con comparacion de modelos, resumen por archivo e interpretacion de extremos simples.

7. `notebooks/06_entrenamiento_earthformer_paso_a_paso.ipynb`
   - Notebook docente para explicar como se construye y entrena EarthFormer.
   - Usa una copia local de la arquitectura real `EarthFormer_xy`.
   - Muestra dataset, YAML, batch, forward pass, perdida ponderada, loop de entrenamiento y guardado `.pth`.

## 7. Que haras en el laboratorio

El laboratorio guia paso a paso:

1. Cargar una secuencia de precipitacion.
2. Separar `inputs = sequence[:13]` y `target = sequence[13:25]`.
3. Cargar una prediccion. Si no hay predicciones de modelo, se usa persistencia.
4. Visualizar entrada, objetivo, prediccion y error absoluto.
5. Calcular metricas continuas por tiempo de pronostico:
   - MAE
   - RMSE
   - Bias
   - Correlacion de Pearson
6. Calcular metricas de eventos de lluvia usando umbrales:
   - 0.5 mm/h: lluvia ligera
   - 2.0 mm/h: lluvia moderada
   - 5.0 mm/h: lluvia fuerte
   - 10.0 mm/h: lluvia intensa
7. Interpretar por que RMSE no basta para evaluar lluvia intensa.

## 8. Utilidades compartidas

Los notebooks usan `course_utils/` para mantener consistencia:

- `palette.py`: paleta discreta de lluvia en mm/h y paleta de error.
- `data.py`: carga de muestras, limpieza de NaN, split 13->12 y predicciones.
- `metrics.py`: MAE, RMSE, bias, correlacion, CSI, POD, FAR, F1 para notebooks resueltos o instructor.
- `plotting.py`: figuras compartidas para eventos, metricas y paneles target/prediccion.
- `model_inference.py`: demo avanzado opcional con checkpoints y GPU.
- `evaluation.py`: evaluacion comparativa de predicciones guardadas.

Esto evita copiar y pegar codigo entre notebooks.

## 9. Opcion avanzada: checkpoints del modelo

Los checkpoints de EarthFormer, Autoencoder y CasCast estan publicados aqui:

<https://huggingface.co/andrexandrex322/ideam-nowcasting-earthformer-cascast/tree/main>

Son archivos grandes (el de difusion pesa ~3.9 GB), por eso no se requieren para la parte basica
de metricas con persistencia. Para correr EarthFormer/CasCast, descargalos con:

```bash
python scripts/download_assets.py --checkpoints
```

Quedaran en `checkpoints/ef_ideam_final/` con los nombres `ef_ckpt.pth`, `ae_ckpt.pth` y
`diff_ckpt.pth`, justo donde los espera `config/model_demo_paths.yaml`. Con eso, la inferencia ya
los encuentra sin configurar nada mas.

> **El repo es autocontenido.** La carpeta `cascast_vendor/` ya incluye todas las
> dependencias de codigo necesarias para correr EarthFormer. No necesitas clonar
> ningun repositorio externo de CasCast. Solo necesitas: este repo + los checkpoints
> descargados con el comando de arriba.

La configuracion del demo avanzado esta en:

```text
config/model_demo_paths.yaml
```

Ahi se definen las rutas de los checkpoints (`ef_ckpt.pth`, `ae_ckpt.pth`, `diff_ckpt.pth`) y el
`scale_factor` latente que usa CasCast. Ajusta esas rutas si guardas los checkpoints en otro lugar.

La inferencia se corre siempre desde los scripts (no desde el notebook), para que sea
reproducible:

```bash
python scripts/run_inference_from_list.py scripts/sequences_example.txt --stage earthformer --device auto
python scripts/evaluate_predictions.py --sample all
```

## 10. Entregables sugeridos

Cada estudiante entrega:

1. Notebook ejecutado: `nowcasting_metrics_lab.ipynb`.
2. Una figura con entrada, objetivo, prediccion y error.
3. Una tabla de MAE, RMSE, bias y correlacion por tiempo de pronostico.
4. Una tabla de CSI, POD, FAR y F1 por umbral.
5. Una reflexion corta de 300 a 500 palabras.

## 11. Rubrica sugerida

| Componente | Puntos |
|---|---:|
| Carga y visualizacion de datos | 20 |
| Metricas continuas implementadas correctamente | 20 |
| Metricas de eventos implementadas correctamente | 25 |
| Interpretacion por tiempo y umbral | 25 |
| Claridad y reproducibilidad | 10 |
| Total | 100 |
