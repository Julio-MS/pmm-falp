import os
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd

class Utils:
    def __init__(self):
        self.root = '/Users/julio/Documentos-Local/data/VinDr-Mammo/images'
        self.findings = pd.read_csv('/Users/julio/Documentos-Local/data/VinDr-Mammo/finding_annotations.csv')

    def get_path(self, image_id):
        row = self.findings[self.findings['image_id'] == image_id]
        if len(row) > 0:
            row = row.iloc[0].squeeze()  # Seleccionar la primera fila y convertirla a Serie

        study_id = row['study_id']
        path = os.path.join(self.root, study_id, image_id + '.dicom')
        return path

    def show_image(self, image_id, cmap, bbox=True):

        row = self.findings.loc[(self.findings['image_id'] == image_id)]
        row.reset_index(inplace=True, drop=True)

        image_path = self.get_path(image_id)
        dicom = pydicom.dcmread(image_path)

        arr = apply_voi_lut(dicom.pixel_array, dicom)
        plt.imshow(arr, cmap=cmap)
        # Obtener el eje actual
        ax = plt.gca()

        colors = ["red", "blue", "green"]

        if len(row) == 0:
            print("No findings")

        for index, item in row.iterrows():
            print(
                f"{index + 1} - {colors[index]} - finding_categories: {row.loc[index]['finding_categories']}, finding_birads: {row.loc[index]['finding_birads']}")
            # Coordenadas del punto superior izquierdo (x1, y1) y punto inferior derecho (x2, y2)
            x1, y1 = row['xmin'].values[index], row['ymin'].values[index]
            x2, y2 = row['xmax'].values[index], row['ymax'].values[index]

            # Calcular el ancho y alto del rectángulo
            width = x2 - x1
            height = y2 - y1

            # Dibujar el rectángulo sobre la imagen
            rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor=colors[index], facecolor='none')
            if bbox:
                ax.add_patch(rect)

        # Mostrar la imagen con el rectángulo
        plt.axis('off')
        plt.show()
