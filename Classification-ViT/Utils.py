import evaluate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def confusion_matrix(samples_, predictions_):
    
    # Cargar la métrica de 'confusion_matrix'
    confusion_matrix_ = evaluate.load("confusion_matrix")
    
    # Calcular la matriz de confusión
    results = confusion_matrix_.compute(predictions=predictions_, references=samples_['label'])

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


def show_predictions(rows, cols, samples_, predictions_, id2label_):
    # Hacer una selección aleatoria de 'rows * cols' muestras
    indices = np.random.choice(len(samples_), size=rows * cols, replace=False)  # Selección aleatoria de índices

    # Visualización de las imágenes seleccionadas aleatoriamente junto con sus predicciones
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for i, idx in enumerate(indices):
        img = samples_[idx]['image']  # Obtener la imagen correspondiente al índice seleccionado
        prediction = predictions_[idx]  # Obtener la predicción correspondiente al índice

        # Crear la etiqueta para mostrar la etiqueta verdadera y la predicción
        label = f"label: {id2label_[samples_[idx]['label']]}\npredicted: {id2label_[prediction]}"

        # Visualizar la imagen con la predicción
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label)
        plt.axis('off')

    # Mostrar el gráfico
    plt.show()