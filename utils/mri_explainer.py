# utils/mri_explainer.py
import matplotlib.pyplot as plt
import numpy as np
import os

MRI_OUTPUT_DIR = os.path.join("outputs", "mri_explanations")
os.makedirs(MRI_OUTPUT_DIR, exist_ok=True)

class MRIAlzheimerExplainer:
    """
    Simplified clinical-style explainer: generates textual interpretation and a small figure.
    """

    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.brain_areas = {
            'hippocampus': 'Memory and learning; early atrophy indicates risk.',
            'entorhinal_cortex': 'Memory & navigation; often affected early.',
            'ventricles': 'Fluid spaces that enlarge with atrophy.',
            'temporal_lobe': 'Memory and language processes.'
        }

    def explain_mri_findings(self, image_tensor, image_name, predicted_class):
        text_map = {
            0: "Pattern within normal range for age; no significant atrophy detected.",
            1: "Subtle changes consistent with very mild cognitive decline.",
            2: "Moderate atrophy consistent with mild dementia.",
            3: "Generalized atrophy consistent with moderate dementia."
        }
        explanation = text_map.get(predicted_class, "Pattern could not be classified reliably.")

        # Create a small figure summarizing explanation
        fig, ax = plt.subplots(figsize=(6,4))
        ax.axis("off")
        ax.text(0, 0.9, f"Diagnosis: {self.class_names[predicted_class]}", fontsize=12, weight='bold')
        ax.text(0, 0.65, f"Interpretation: {explanation}", fontsize=10)
        ax.text(0, 0.4, "Relevant regions:", fontsize=11, weight='bold')
        y = 0.32
        for k, v in self.brain_areas.items():
            ax.text(0.02, y, f"• {k}: {v}", fontsize=9)
            y -= 0.06

        save_path = os.path.join(MRI_OUTPUT_DIR, f"mri_explanation_{self.class_names[predicted_class].replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return save_path
