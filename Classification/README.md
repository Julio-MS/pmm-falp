# PMM - FALP

## Archivos

[`model_1.0.0.35.ipynb`](model_1.0.0.35.ipynb) 
- Adaptación del modelo propuesto en [mammography-models
](https://github.com/escuccim/mammography-models/tree/master?tab=readme-ov-file).
- Se actualxiza la arquitectura de TensorFlow 1.X a Keras 3.5. 
- Se usa transfer learning cargando los pesos originales, congelando las capas convolucionales y entrenando las capas fully conected.
[model_1.0.0.35_evaluation.ipynb](model_1.0.0.35_evaluation.ipynb)

[`model_1.0.0.35_evaluation.ipynb`](model_1.0.0.35_evaluation.ipynb), 
[`model_1.0.0.35_evaluation_2.ipynb`](model_1.0.0.35_evaluation_2.ipynb), 
[`model_1.0.0.35_PCA.ipynb`](model_1.0.0.35_PCA.ipynb)

- Evaluación del desempeño del modelo: Métricas, Matriz de Confusión, ROC Curve, PCA en el espacio latente.