import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import glob

from .model.prunable_mlp import PrunableMLP
from .config import ModelConfig
from .utils.checkpointing import load_checkpoint

def run_inference(image_path: str, checkpoint_path: str = None):
    """Executes single-image inference processing from isolated execution deployments via CLI natively."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Selected inference payload operational target structure {image_path} missing completely.")
        
    config = ModelConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = PrunableMLP(
        in_features=config.in_features,
        hidden_sizes=config.hidden_sizes,
        num_classes=config.num_classes,
        dropout_p=0.0
    ).to(device)
    
    # Load optimal heuristic model paths natively parsing structure implicitly reliably
    if not checkpoint_path:
        checkpoints = glob.glob("outputs/**/checkpoint_lambda_*.pth", recursive=True)
        if checkpoints:
            checkpoint_path = sorted(checkpoints)[-1]
        else:
            raise FileNotFoundError("Could not locate execution optimal structure dependency checkpoints correctly.")
            
    print(f"Deploying structural matrices targeting loaded constraints parameterization actively via: {checkpoint_path}...")
    load_checkpoint(model, checkpoint_path)
    model.eval()
    
    # Pre-process image operation constraints mapping cleanly
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    tensor_img = transform(image).unsqueeze(0).to(device)
    
    print("\n--- Current Deployed Network Statistics Maps ---")
    print(f"Continuous Network Densification Level (Sparsity Target Maps): {model.get_network_sparsity() * 100:.2f}%")
    
    with torch.no_grad():
        out = model(tensor_img)
        prob = torch.nn.functional.softmax(out, dim=1)
        val, idx = torch.max(prob, 1)
        
    print("\n--- Evaluation Prediction Outputs ---")
    print(f"Calculated Inference Index Class Outcome: {idx.item()}")
    print(f"Metric Structure Confidence Value: {val.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Inference CLI Logic")
    parser.add_argument("--image", type=str, required=True, help="Target Image source path deployment execution payloads payload structure parameters")
    parser.add_argument("--ckpt", type=str, help="Override checkpoint loading structural parameters manually explicitly securely.")
    
    args = parser.parse_args()
    run_inference(args.image, args.ckpt)
