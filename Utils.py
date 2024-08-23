import os
import pydicom
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pandas as pd

class Utils:
    def __init__(self, image_id, bbox):
        self.image_id = image_id
        self.bbox = bbox
        self.findings = pd.read_csv('/Users/julio/Documentos-Local/data/DrVinn-Mammo/metadata/finding_annotations.csv')

    def show_image(self):
        raw_path = '/Users/julio/Documentos-Local/data/DrVinn-Mammo/raw'
        image_id = 'd88d4b9103281220b9ac1a122364973b'  # Seleccionar imágen a revisar

        path = os.path.join(raw_path, image_id + '.dicom')
        dicom = pydicom.dcmread(path)

        row = self.findings.loc[(self.findings['image_id'] == self.image_id) & (self.findings['finding_categories'] == "['Mass']")]

        if not self.bbox:
            arr = apply_voi_lut(dicom.pixel_array, dicom)
            plt.axis('off')
            plt.imshow(arr, 'gray')

