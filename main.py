
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Data Preparation and Preprocessing
class NoisyMNIST:
    def __init__(self, train=True, noise_level=0.5):
        self.mnist = datasets.MNIST(
            root='./data', 
            train=train, 
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        self.noise_level = noise_level
    
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        clean_img, label = self.mnist[idx]
        
        # Add Gaussian noise
        noise = torch.randn_like(clean_img) * self.noise_level
        noisy_img = clean_img + noise
        
        # Clip to valid range
        noisy_img = torch.clamp(noisy_img, -1.0, 1.0)
        
        return noisy_img, clean_img, noise

# 2. Define the Noise Prediction Network
class NoisePredictor(nn.Module):
    def __init__(self):
        super(NoisePredictor, self).__init__()
        
        self.encoder = nn.Sequential(
            # Input: 1x28x28
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x14x14
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x7x7
        )
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            # Upsample to 14x14
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            
            # Upsample to 28x28
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            
            # Final output
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# 3. Training Function
def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-3):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    val_psnrs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for noisy_imgs, clean_imgs, true_noise in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            noisy_imgs = noisy_imgs.to(device)
            true_noise = true_noise.to(device)
            
            optimizer.zero_grad()
            pred_noise = model(noisy_imgs)
            loss = criterion(pred_noise, true_noise)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        with torch.no_grad():
            for noisy_imgs, clean_imgs, true_noise in val_loader:
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                true_noise = true_noise.to(device)
                
                pred_noise = model(noisy_imgs)
                loss = criterion(pred_noise, true_noise)
                val_loss += loss.item()
                
                # Calculate PSNR
                denoised_imgs = noisy_imgs - pred_noise
                denoised_imgs = torch.clamp(denoised_imgs, -1.0, 1.0)
                
                # Convert to numpy for PSNR calculation
                clean_np = clean_imgs.cpu().numpy()
                denoised_np = denoised_imgs.cpu().numpy()
                
                batch_psnr = 0.0
                for i in range(len(clean_np)):
                    # Denormalize from [-1, 1] to [0, 1]
                    clean_img = (clean_np[i].squeeze() + 1) / 2
                    denoised_img = (denoised_np[i].squeeze() + 1) / 2
                    batch_psnr += psnr(clean_img, denoised_img, data_range=1.0)
                
                val_psnr += batch_psnr / len(clean_np)
        
        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_losses.append(val_loss)
        val_psnrs.append(val_psnr)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val PSNR: {val_psnr:.2f} dB')
        print('-' * 50)
    
    return train_losses, val_losses, val_psnrs

# 4. Visualization Function
def visualize_results(model, test_loader, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Get a batch of test data
        noisy_imgs, clean_imgs, true_noise = next(iter(test_loader))
        noisy_imgs = noisy_imgs.to(device)
        
        # Predict noise and denoise
        pred_noise = model(noisy_imgs)
        denoised_imgs = noisy_imgs - pred_noise
        denoised_imgs = torch.clamp(denoised_imgs, -1.0, 1.0)
        
        # Convert to numpy for visualization
        noisy_imgs_np = noisy_imgs.cpu().numpy()[:num_samples]
        clean_imgs_np = clean_imgs.numpy()[:num_samples]
        denoised_imgs_np = denoised_imgs.cpu().numpy()[:num_samples]
        
        # Create visualization
        fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
        
        for i in range(num_samples):
            # Denormalize from [-1, 1] to [0, 1]
            noisy_img = (noisy_imgs_np[i].squeeze() + 1) / 2
            clean_img = (clean_imgs_np[i].squeeze() + 1) / 2
            denoised_img = (denoised_imgs_np[i].squeeze() + 1) / 2
            
            # Calculate PSNR for this sample
            sample_psnr = psnr(clean_img, denoised_img, data_range=1.0)
            
            # Plot images
            axes[0, i].imshow(noisy_img, cmap='gray')
            axes[0, i].set_title(f'Noisy Input')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(clean_img, cmap='gray')
            axes[1, i].set_title('Clean Ground Truth')
            axes[1, i].axis('off')
            
            axes[2, i].imshow(denoised_img, cmap='gray')
            axes[2, i].set_title(f'Denoised\nPSNR: {sample_psnr:.2f} dB')
            axes[2, i].axis('off')
        
        plt.tight_layout()
        plt.show()

# 5. Main Execution
def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 2
    noise_level = 0.5
    learning_rate = 1e-3
    
    print("Loading and preparing datasets...")
    
    # Create datasets
    train_dataset = NoisyMNIST(train=True, noise_level=noise_level)
    val_dataset = NoisyMNIST(train=False, noise_level=noise_level)
    
    # Split validation for test
    val_size = len(val_dataset)
    test_size = val_size // 2
    val_size = val_size - test_size
    
    val_dataset, test_dataset = torch.utils.data.random_split(
        val_dataset, [val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print("Initializing model...")
    model = NoisePredictor()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, val_psnrs = train_model(
        model, train_loader, val_loader, num_epochs, learning_rate
    )
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_psnrs)
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.title('Validation PSNR')
    
    plt.tight_layout()
    plt.show()
    
    # Visualize results on test set
    print("Visualizing results...")
    visualize_results(model, test_loader)
    
    # Calculate final metrics on test set
    print("Calculating final metrics...")
    model.eval()
    test_psnr_before = 0.0
    test_psnr_after = 0.0
    
    with torch.no_grad():
        for noisy_imgs, clean_imgs, true_noise in test_loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            pred_noise = model(noisy_imgs)
            denoised_imgs = noisy_imgs - pred_noise
            denoised_imgs = torch.clamp(denoised_imgs, -1.0, 1.0)
            
            # Convert to numpy for PSNR calculation
            clean_np = clean_imgs.numpy()
            noisy_np = noisy_imgs.cpu().numpy()
            denoised_np = denoised_imgs.cpu().numpy()
            
            for i in range(len(clean_np)):
                clean_img = (clean_np[i].squeeze() + 1) / 2
                noisy_img = (noisy_np[i].squeeze() + 1) / 2
                denoised_img = (denoised_np[i].squeeze() + 1) / 2
                
                test_psnr_before += psnr(clean_img, noisy_img, data_range=1.0)
                test_psnr_after += psnr(clean_img, denoised_img, data_range=1.0)
    
    test_psnr_before /= len(test_dataset)
    test_psnr_after /= len(test_dataset)
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"PSNR before denoising: {test_psnr_before:.2f} dB")
    print(f"PSNR after denoising: {test_psnr_after:.2f} dB")
    print(f"PSNR improvement: {test_psnr_after - test_psnr_before:.2f} dB")
    print("="*50)
    
    # Save model
    torch.save(model.state_dict(), 'denoising_model.pth')
    print("Model saved as 'denoising_model.pth'")

if __name__ == "__main__":
    main()