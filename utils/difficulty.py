# utils/difficulty.py
import torch

class DifficultyAnalyzer:
    """
    Simple confidence-based difficulty categorizer.
    """
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names

    def predict_confidence(self, image_tensor):
        self.model.eval()
        with torch.no_grad():
            out = self.model(image_tensor.unsqueeze(0).to(self.device))
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, 1)
        return float(conf.item()), int(pred.item())

    def categorize(self, confidence, correct=None):
        """
        Return 'easy', 'medium', or 'hard'.
        If 'correct' is provided (True/False), it influences the label.
        """
        if confidence > 0.8 and (correct is None or correct):
            return "easy"
        if confidence < 0.6 or (correct is not None and not correct):
            return "hard"
        return "medium"
