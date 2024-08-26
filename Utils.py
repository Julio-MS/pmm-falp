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

    def show_image(self, image_id, cmap='gray', bbox=True, metadata=True, finding=None, ax=None):

        if finding:
            row = self.findings.loc[(self.findings['image_id'] == image_id) &
                                    (self.findings['finding_categories'] == finding)]
            row.reset_index(inplace=True, drop=True)
        else:
            row = self.findings.loc[(self.findings['image_id'] == image_id)]
            row.reset_index(inplace=True, drop=True)

        image_path = self.get_path(image_id)
        dicom = pydicom.dcmread(image_path)

        arr = apply_voi_lut(dicom.pixel_array, dicom)

        if ax is None:
            # Crear la figura y el eje
            fig, ax = plt.subplots()

        ax.imshow(arr, cmap=cmap)
        # Obtener el eje actual

        colors = ["red", "blue", "green"]

        if len(row) == 0:
            print("No findings")

        for index, item in row.iterrows():
            if metadata:
                print(
                    f"{index + 1} - {colors[index]} - finding_categories: {row.loc[index]['finding_categories']}, "
                    f"\nfinding_birads: {row.loc[index]['finding_birads']}, "
                    f"breast_birads: {row.loc[index]['breast_birads']}, "
                    f"Photometric Interpretation: {dicom.PhotometricInterpretation}"
                )
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

        if ax is None:
            # Mostrar la imagen con el rectángulo
            plt.axis('off')
            plt.show()

        return ax

    def show_sample(self, finding_categories):

        sample = self.findings.loc[self.findings['finding_categories'] == finding_categories]
        sample.reset_index(drop=True, inplace=True)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for index, row in sample.iterrows():
            if index < 6:
                i = index // 3
                j = index % 3

                ax = axes[i, j]

                self.show_image(row['image_id'], finding=finding_categories, metadata=False, ax=ax)
                #ax.axis('off')
                ax.set_title(row['image_id'])

            else:
                break

        for i in range(2):
            for j in range(3):
                axes[i, j].axis('off')

        to_delite = ["'", "[", "]"]
        title = finding_categories

        for char in to_delite:
            title = title.replace(char, '')

        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
