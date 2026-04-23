import torch
import torch.nn as nn
import torch.optim as optim
from model import TinyVGG
from train import Trainer
from config import train_config, config_TinyVGG

class FineTuner(Trainer):
    def __init__(self, train_config, model_config, pretrained_model_path, save_freq=2):
        super().__init__(train_config, model_config, save_freq=save_freq)
        
        # Load the pretrained model
        self._load_pretrained_model(pretrained_model_path)
        
        # Freeze all layers except the last linear layer
        self._freeze_layers()
        
        # Reset the optimizer to only update the unfrozen parameters
        self._reset_optimizer()
        
    def _load_pretrained_model(self, pretrained_model_path):
        """Load weights from a pretrained model"""
        checkpoint = torch.load(pretrained_model_path, map_location=self.device)
        # Handle both direct state dict and dictionary format
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded pretrained model from {pretrained_model_path}")
        
    def _freeze_layers(self):
        """Freeze all layers except the last linear layer (classifier)"""
        # Freeze the feature extractor
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        # Freeze the flatten layer in classifier but not the linear layer
        for name, param in self.model.classifier.named_parameters():
            if "1." not in name:  # Only the linear layer (index 1 in sequential) should be trainable
                param.requires_grad = False
                
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
        
    def _reset_optimizer(self):
        """Reset the optimizer to only update the unfrozen parameters"""
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.learning_rate
        )

def main():
    # Path to the pretrained model
    pretrained_model_path = "models/pretrained_tinyvgg.pth"
    
    # Create fine-tuner instance with configs
    finetuner = FineTuner(train_config, config_TinyVGG, pretrained_model_path, save_freq=2)
    
    # Run fine-tuning
    finetuner.train()
    
    # Clean up resources
    finetuner.cleanup()
    
    # Save the fine-tuned model
    torch.save(finetuner.model.state_dict(), f"models/finetuned_tinyvgg_{finetuner.timestamp}.pth")
    print(f"Saved fine-tuned model to models/finetuned_tinyvgg_{finetuner.timestamp}.pth")

if __name__ == "__main__":
    main()
