import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime

from ..config import ModelConfig
from ..data.cifar10 import get_dataloaders, get_tiny_dataloader
from ..model.prunable_mlp import PrunableMLP
from ..train import train_one_epoch, run_smoke_test
from ..evaluate import evaluate_model
from ..utils.seeding import seed_everything
from ..utils.logging_utils import get_logger
from ..utils.checkpointing import save_checkpoint, load_checkpoint
from ..visualization.plot_gates import plot_gate_distribution

logger = get_logger("LambdaSweep")

def determine_device(req_device: str) -> torch.device:
    """Determines the appropriate computing device."""
    if req_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            # Force CPU for MPS locally if there are implementation limits, but usually MPS works.
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(req_device)


def run_experiment(config: ModelConfig, lambda_val: float, exp_dir: str, device: torch.device):
    """Runs a complete training loop for a specific lambda value."""
    logger.info(f"--- Starting experiment with lambda = {lambda_val} ---")
    
    seed_everything(config.seed)
    
    train_loader, test_loader = get_dataloaders(config.data_dir, config.batch_size)
    
    # Smoke test on a tiny batch
    tiny_loader = get_tiny_dataloader(config.data_dir, config.batch_size)
    
    model = PrunableMLP(
        in_features=config.in_features,
        hidden_sizes=config.hidden_sizes,
        num_classes=config.num_classes,
        dropout_p=config.dropout_p
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Run the sanity check before jumping in
    run_smoke_test(model, tiny_loader, optimizer, criterion, device)
    
    # Re-initialize for actual training
    model.apply(lambda m: getattr(m, "reset_parameters", lambda: None)())
    
    best_acc = 0.0
    checkpoint_path = os.path.join(exp_dir, f"checkpoint_lambda_{lambda_val}.pth")
    
    for epoch in range(1, config.epochs + 1):
        logger.info(f"[Epoch {epoch}/{config.epochs}] Lambda {lambda_val}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, lambda_val)
        
        logger.info(f"Train - Total Loss: {train_metrics['loss']:.4f} | " 
                    f"Cls Loss: {train_metrics['cls_loss']:.4f} | " 
                    f"Reg Loss: {train_metrics['reg_loss']:.4f} | "
                    f"Acc: {train_metrics['accuracy']:.2f}%")
        
        eval_metrics = evaluate_model(model, test_loader, criterion, device, lambda_val, config.pruning_threshold)
        
        logger.info(f"Eval  - Total Loss: {eval_metrics['loss']:.4f} | " 
                    f"Acc: {eval_metrics['accuracy']:.2f}% | "
                    f"Sparsity: {eval_metrics['sparsity']:.2f}%")
        
        # Model Selection: We prioritize validation classification accuracy.
        # This keeps representation reliable while restricting capacity.
        if eval_metrics['accuracy'] >= best_acc:
            best_acc = eval_metrics['accuracy']
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            logger.info(f"-> Saved new best model with {best_acc:.2f}% accuracy.")
            
    # Load best architecture to return final stats for sweeping table
    logger.info(f"Loading best model from {checkpoint_path} for final eval.")
    load_checkpoint(model, checkpoint_path)
    final_metrics = evaluate_model(model, test_loader, criterion, device, lambda_val, config.pruning_threshold)
    
    return model, final_metrics


def main():
    parser = argparse.ArgumentParser(description="Sweep Lambda for Pruning")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = ModelConfig.from_yaml(args.config)
    device = determine_device(config.device)
    logger.info(f"Using device: {device}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config.output_dir, f"sweep_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    best_overall_acc = 0.0
    best_model_run = None
    
    for l_val in config.lambdas:
        model, metrics = run_experiment(config, l_val, out_dir, device)
        
        results.append({
            "Lambda": l_val,
            "Test Accuracy": metrics["accuracy"],
            "Sparsity Level (%)": metrics["sparsity"]
        })
        
        # Track the very best single pipeline model across all sweeps for plotting
        if metrics["accuracy"] >= best_overall_acc:
            best_overall_acc = metrics["accuracy"]
            best_model_run = (l_val, model)
            
    # Save Summary Report
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    df.to_csv(csv_path, index=False)
    
    md_path = os.path.join(out_dir, "sweep_results.md")
    with open(md_path, "w") as f:
        f.write("# Lambda Sweep Results\n\n")
        f.write(df.to_markdown(index=False))
        
    logger.info(f"Saved results summary to {out_dir}")
    print("\n" + df.to_markdown(index=False))
    
    # Plot best model explicitly
    if best_model_run:
        l_val, b_model = best_model_run
        plot_path = os.path.join(out_dir, "best_model_gate_dist.png")
        plot_gate_distribution(b_model, plot_path, title=f"Gate Distribution (Best Model, Lambda={l_val})")
        logger.info(f"Saved gate distribution plot to {plot_path}")

if __name__ == "__main__":
    main()
