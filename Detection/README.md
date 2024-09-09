# PMM FALP

## To-DO

- [x] Comprender Windowing y VOI LUT
- [x] Visualizar muestras de los tipos de hallazgos
- [x] Revisar que todas las imágenes tengan VOI LUT o Windowing
- [x] Convertir a PNG usando los datos de la imagen
- [ ] ~~Modelo YOLO de detección para 1 tipo de hallazgo~~


## Archivos

- `Image_Exploration.ipynb` Notebook principal. Exploración de imágenes y las diferentes categorías de hallazgos, junto con los metadatos necesarios para hacer el proceso de **Windowing**.
- `Image_Classification.ipynb` Creación de sub set de imágenes según características de visualización
- `Utils.py` Clase que contiene diferentes funciones para visualizar imágenes.

#### Antiguos

- `Dicom_view.ipynb` Notebook inicial para visualizar imagen DICOM.
- `kaggle_preprocessing.ipynb` Notebook donde se probaron diferentes cosas para visualización de imagen con diferentes windowings.
- `Preprocessing.ipynb` Exploración de csv finding_annotations y también función para copiar imágenes y organizarlas