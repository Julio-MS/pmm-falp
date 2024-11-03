import evaluate
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm  # Barra de progreso para visualizar el procesamiento
import torch

def confusion_matrix(samples_, predictions_, class_names):
    
    # Cargar la métrica de 'confusion_matrix'
    confusion_matrix_ = evaluate.load("confusion_matrix")
    
    # Calcular la matriz de confusión
    results = confusion_matrix_.compute(predictions=predictions_, references=samples_['label'])

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
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

def compute_metrics(y_true, y_pred):

    # Cargar métricas de evaluación
    accuracy = evaluate.load('accuracy')
    precision = evaluate.load('precision')
    recall = evaluate.load('recall')
    f1 = evaluate.load('f1')

    # Precisión por clase
    precision_per_class = precision.compute(predictions=y_pred, references=y_true, average=None)
    print("Precision por clase:", precision_per_class, '\n')

    # Accuracy
    accuracy_score = accuracy.compute(predictions=y_pred, references=y_true)['accuracy']

    # Otras métricas (macro-average)
    precision_score = precision.compute(predictions=y_pred, references=y_true, average='macro')['precision']
    recall_score = recall.compute(predictions=y_pred, references=y_true, average='macro')['recall']
    f1_score = f1.compute(predictions=y_pred, references=y_true, average='macro')['f1']

    return {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score,
        'f1': f1_score
    }

def leyenda_manual(scatter, class_names):
    handles = []
    for i, class_name in enumerate(class_names):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=class_name,
                                  markerfacecolor=scatter.cmap(scatter.norm(i)), markersize=10))
    return handles

def plot_pca(labels, latent_features, class_names):
    # Aplicar PCA considerando todas las componentes
    pca = PCA()  # Si no especificas n_components, calculará todas las componentes
    pca_result = pca.fit_transform(latent_features)

    # Obtener la proporción de varianza explicada por cada componente
    explained_variance = pca.explained_variance_ratio_

    # Imprimir la varianza explicada por las primeras componentes
    print("Varianza explicada por las primeras 10 componentes:")
    print(explained_variance[:10])

    # Graficar la varianza explicada acumulada
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(explained_variance), marker='o', linestyle='--', color='b')
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza explicada acumulada')
    plt.title('Varianza explicada por las componentes principales')
    plt.grid(True)
    plt.show()

    pca2 = PCA(n_components=2)
    latent_2d = pca2.fit_transform(latent_features)

    """Grafica las características latentes proyectadas con PCA."""
    plt.figure(figsize=(8, 6))
    # Asumimos que tienes 3 categorías (ajusta los colores según sea necesario)
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    # Crear una leyenda manualmente
    handles = leyenda_manual(scatter, class_names)

    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.title('Proyección PCA de las Representaciones Latentes')
    # Añadir la leyenda
    plt.legend(handles=handles, title="Clases", bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.colorbar(scatter, label='Etiqueta de la clase')
    plt.show()

def t_sne(labels, latent_features, class_names):
    # Aplicar t-SNE a las representaciones latentes (por ejemplo, el espacio después del encoder)
    tsne = TSNE(n_components=2, perplexity=30, max_iter=300)
    tsne_result = tsne.fit_transform(latent_features)  # Asume que 'latent_representations' es el espacio latente

    # Crear el gráfico t-SNE
    plt.figure(figsize=(8, 6))

    # Colores personalizados para cada clase
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    handles = leyenda_manual(scatter, class_names)

    # Añadir la leyenda
    plt.legend(handles=handles, title="Clases", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Etiquetas y título
    plt.xlabel('Dimensión t-SNE 1')
    plt.ylabel('Dimensión t-SNE 2')
    plt.title('t-SNE de las representaciones latentes')

    # Mostrar gráfico
    plt.show()

def extract_latent_features(dataloader, vit_without_classifier, device):
    """Extrae las representaciones latentes utilizando MPS o CPU."""
    vit_without_classifier.eval()  # Modo evaluación
    latent_features = []
    labels = []

    # Iterar sobre los batches en el DataLoader
    for batch in tqdm(dataloader, desc="Extrayendo representaciones latentes"):
        # Mover los datos al dispositivo correcto
        inputs = batch['pixel_values'].to(device)  # Imagen en MPS o CPU
        batch_labels = batch['labels'].to(device)

        with torch.no_grad():
            # Extraer las representaciones latentes
            output = vit_without_classifier(inputs).last_hidden_state
            latent_repr = output.mean(dim=1)  # Promediar sobre la secuencia
            latent_features.append(latent_repr.cpu().numpy())  # Mover a CPU para evitar saturación
            labels.extend(batch_labels.cpu().numpy())  # Guardar etiquetas en CPU

    # Convertir listas to arrays NumPy
    latent_features = np.vstack(latent_features)
    labels = np.array(labels)

    return latent_features, labels