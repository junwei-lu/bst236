import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from random_svd import randomized_svd

# Load the image
img = Image.open('tiger.jpg')
img_array = np.array(img)

# Fixed rank for all comparisons
rank = 20

# Check if the image is RGB or grayscale
if len(img_array.shape) == 3:
    # Only process channel 0 (typically red in RGB)
    height, width, channels = img_array.shape
    print(f"Image dimensions: {height}x{width} with {channels} channels")
    print(f"Only processing channel 0")
    
    # Extract channel 0
    channel = img_array[:, :, 0]
    
    # Create a figure to display the results in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Sketch ratios for random SVD
    sketch_ratios = [0.2, 0.4, 0.6, 0.8]
    
    # Original channel
    axes[0].imshow(channel, cmap='gray')
    axes[0].set_title(f"Original Channel 0")
    axes[0].axis('off')
    
    # Deterministic SVD (full SVD and then truncate)
    U, Sigma, Vt = np.linalg.svd(channel, full_matrices=False)
    
    # Reconstruct with rank 20
    reconstructed_det = U[:, :rank] @ np.diag(Sigma[:rank]) @ Vt[:rank, :]
    
    # Clip values to valid pixel range
    reconstructed_det = np.clip(reconstructed_det, 0, 255).astype(np.uint8)
    
    # Display the deterministic SVD reconstructed image
    axes[1].imshow(reconstructed_det, cmap='gray')
    axes[1].set_title(f"Deterministic SVD\nrank={rank}")
    axes[1].axis('off')
    
    # Apply random SVD with different sketch ratios
    for i, s in enumerate(sketch_ratios):
        if i >= 4:  # Only use the first 4 sketch ratios to fit in our 2x3 grid
            break
            
        # Calculate the number of samples based on the sketch ratio
        s_samples = int(min(height, width) * s)
        
        # Apply randomized SVD
        U, Sigma, Vt = randomized_svd(channel, rank, s_samples)
        
        # Reconstruct the image using only the top singular values
        reconstructed = U[:, :rank] @ np.diag(Sigma[:rank]) @ Vt[:rank, :]
        
        # Clip values to valid pixel range
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        # Display the reconstructed image
        axes[i+2].imshow(reconstructed, cmap='gray')
        axes[i+2].set_title(f"Random SVD\nrank={rank}, sketch={s}")
        axes[i+2].axis('off')
        
        # Save the reconstructed image (channel 0 only)
        reconstructed_img = Image.fromarray(reconstructed)
        reconstructed_img.save(f"tiger_ch0_random_svd_rank{rank}_sketch{s}.jpg")
    
    # Save the deterministic SVD reconstructed image (channel 0 only)
    reconstructed_img = Image.fromarray(reconstructed_det)
    reconstructed_img.save(f"tiger_ch0_deterministic_svd_rank{rank}.jpg")
    
    plt.tight_layout()
    plt.savefig("tiger_ch0_svd_comparison.png", dpi=300)
    plt.show()
else:
    # Grayscale image processing
    height, width = img_array.shape
    print(f"Image dimensions: {height}x{width} (grayscale)")
    
    # Create a figure to display the results in a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # Flatten to make indexing easier
    
    # Original image
    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Deterministic SVD (full SVD and then truncate)
    U, Sigma, Vt = np.linalg.svd(img_array, full_matrices=False)
    
    # Reconstruct with rank 20
    reconstructed_det = U[:, :rank] @ np.diag(Sigma[:rank]) @ Vt[:rank, :]
    
    # Clip values to valid pixel range
    reconstructed_det = np.clip(reconstructed_det, 0, 255).astype(np.uint8)
    
    # Display the deterministic SVD reconstructed image
    axes[1].imshow(reconstructed_det, cmap='gray')
    axes[1].set_title(f"Deterministic SVD\nrank={rank}")
    axes[1].axis('off')
    
    # Save the deterministic SVD reconstructed image
    reconstructed_img = Image.fromarray(reconstructed_det)
    reconstructed_img.save(f"tiger_deterministic_svd_rank{rank}.jpg")
    
    # Sketch ratios for random SVD
    sketch_ratios = [0.2, 0.4, 0.6, 0.8]
    
    # Apply random SVD with different sketch ratios
    for i, s in enumerate(sketch_ratios):
        if i >= 4:  # Only use the first 4 sketch ratios to fit in our 2x3 grid
            break
            
        # Calculate the number of samples based on the sketch ratio
        s_samples = int(min(height, width) * s)
        
        # Apply randomized SVD
        U, Sigma, Vt = randomized_svd(img_array, rank, s_samples)
        
        # Reconstruct the image using only the top singular values
        reconstructed = U[:, :rank] @ np.diag(Sigma[:rank]) @ Vt[:rank, :]
        
        # Clip values to valid pixel range
        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        
        # Display the reconstructed image
        axes[i+2].imshow(reconstructed, cmap='gray')
        axes[i+2].set_title(f"Random SVD\nrank={rank}, sketch={s}")
        axes[i+2].axis('off')
        
        # Save the reconstructed image
        reconstructed_img = Image.fromarray(reconstructed)
        reconstructed_img.save(f"tiger_random_svd_rank{rank}_sketch{s}.jpg")
    
    plt.tight_layout()
    plt.savefig("tiger_svd_comparison.png", dpi=300)
    plt.show()

print("SVD comparison complete. Images saved.") 