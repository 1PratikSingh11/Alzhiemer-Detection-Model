# utils/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class GradCAMAnalyzer:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        # last conv block of ResNet18
        try:
            target = self.model.resnet18.layer4[-1]
        except Exception:
            # fallback: try generic attribute
            target = None
            for m in self.model.modules():
                if isinstance(m, torch.nn.Conv2d):
                    target = m
        if target is None:
            raise RuntimeError("Could not find target layer for Grad-CAM in the model.")

        def forward_hook(mod, inp, outp):
            self.activations = outp.detach()

        def backward_hook(mod, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handles.append(target.register_forward_hook(forward_hook))
        self.handles.append(target.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, image_tensor, target_class):
        self.model.eval()
        x = image_tensor.unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        out = self.model(x)
        # choose target class
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1.0
        self.model.zero_grad()
        out.backward(gradient=one_hot, retain_graph=True)

        # pooled gradients
        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)  # N,C,1,1
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)  # C,H,W -> H,W
        cam = F.relu(cam)
        cam_np = cam.cpu().numpy()
        cam_np -= cam_np.min()
        cam_np /= cam_np.max() if cam_np.max() != 0 else 1.0
        return cam_np, target_class

    def visualize_gradcam(self, image_tensor, image_path, target_class, save_path=None):
        heat, _ = self.generate_heatmap(image_tensor, target_class)
        img_np = self._denormalize(image_tensor.cpu())
        heat_u8 = np.uint8(255 * heat)
        heat_resize = np.array(Image.fromarray(heat_u8).resize(img_np.shape[1::-1], Image.BILINEAR))
        cmap = plt.cm.jet(heat_resize / 255.0)[:, :, :3]
        overlay = 0.5 * cmap + 0.5 * img_np
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1,3, figsize=(15,5))
        axes[0].imshow(img_np); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(heat_resize, cmap="jet"); axes[1].set_title("Grad-CAM"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title(f"Overlay - {self.class_names[target_class]}"); axes[2].axis("off")
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def _denormalize(self, tensor):
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        t = tensor * std + mean
        t = torch.clamp(t, 0, 1)
        return t.permute(1,2,0).numpy()
