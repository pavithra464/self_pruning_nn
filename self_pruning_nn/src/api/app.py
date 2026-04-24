from fastapi import FastAPI, HTTPException, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

from ..model.prunable_mlp import PrunableMLP
from ..config import ModelConfig
from ..utils.checkpointing import load_checkpoint

app = FastAPI(
    title="Self-Pruning NN Inference API",
    description="Exposes inference and architectural metadata tracking for structurally parameterized self-pruning Neural Networks."
)

# Global instances for memory-persistent serving
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
transform = None

@app.on_event("startup")
def startup_event():
    """Initializes the model dynamically on remote infrastructure cycle deployment start."""
    global model, transform
    
    # Ideally loaded exclusively relying configurations parameters tracked explicitly externally
    config = ModelConfig()
    
    model = PrunableMLP(
        in_features=config.in_features,
        hidden_sizes=config.hidden_sizes,
        num_classes=config.num_classes,
        dropout_p=0.0 # Absolute operational dropout removal standard evaluation parameter logic requirement
    ).to(device)
    
    # Heuristic deployment loading logic targeting most modern pipeline completion cycles
    import glob
    checkpoints = glob.glob("outputs/**/checkpoint_lambda_*.pth", recursive=True)
    if checkpoints:
        best_chk = sorted(checkpoints)[-1]
        load_checkpoint(model, best_chk)
        print(f"Deployment Ready - Successfully Loaded Optimal Path Checkpoint Component Structure: {best_chk}")
    else:
        print("Warning Runtime Expectation - Functional Weights Missing. Deploying Null Heuristic Models Structure Effectively Temporarily.")
        
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


@app.get("/health")
def health_check():
    """Minimal endpoint confirming execution continuity operational parameters exist cleanly active."""
    return {"status": "ok", "active_device_map": str(device)}


@app.get("/model-info")
def model_info():
    """Returns static dimension scaling capacity configurations statically tracking active operational matrices dynamically deployed currently."""
    if not model:
        raise HTTPException(status_code=503, detail="Operational Execution Unavailable - Architecture Missing Setup Initializations")
        
    return {
        "architecture_topology": "Self-Pruning MLP Network",
        "in_features": model.in_features,
        "prunable_layers_count": len(model.get_prunable_layers()),
        "output_capacity_metrics": 10
    }


@app.get("/sparsity")
def get_sparsity(threshold: float = 0.01):
    """Dynamically queries currently executing live parameter boundary density levels calculating explicitly via active component operations calculations cleanly."""
    if not model:
        raise HTTPException(status_code=503, detail="Operational Execution Unavailable - Architecture Missing Setup Initializations")
        
    sparsity = model.get_network_sparsity(threshold=threshold)
    return {
        "threshold_value_cap": threshold,
        "structural_sparsity_percentage_value": round(sparsity * 100, 2)
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Primary pipeline operations evaluating singular continuous functional payload matrices actively via live deployment."""
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, detail="Invalid operational payload parameter constraints formatting limitations execution")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        top_prob, top_class = torch.max(probabilities, 1)

    return {
        "class_index_map": top_class.item(),
        "probabilistic_confidence_score": float(top_prob.item()),
    }
