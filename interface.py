import torch
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
from torchvision.utils import save_image
from pathlib import Path


import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, in_dim=100, dim=64):
        super(Generator, self).__init__()
        
        def dconv_bn_relu(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2, padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU())
        
        # Fully connected layer to expand noise to a larger size
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, dim * 8 * 15 * 15, bias=False),
            nn.BatchNorm1d(dim * 8 * 15 * 15),
            nn.ReLU())

        # Deconvolutional layers for upsampling to 240x240
        self.l2_5 = nn.Sequential(
            dconv_bn_relu(dim * 8, dim * 4),   # 15x15 -> 30x30
            dconv_bn_relu(dim * 4, dim * 2),   # 30x30 -> 60x60
            dconv_bn_relu(dim * 2, dim),       # 60x60 -> 120x120
            nn.ConvTranspose2d(dim, 1, 5, 2, padding=2, output_padding=1),  # 120x120 -> 240x240
            nn.Sigmoid())  # Output pixel values in range [0, 1]

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 15, 15)
        y = self.l2_5(y)
        return y

def get_G(z_dim, epoch):
    # Initialize the generator model
    G = Generator(z_dim)
    G = torch.nn.DataParallel(G)  # Wrap the model for potential multi-device compatibility

    # Define the path based on the epoch
    path_G = ""
    if epoch == 75:
        path_G = "./ep75_improved_BraTS23_G.pt"
    elif epoch == 25:
        path_G = "./improved_BraTS23_G.pt"
    elif epoch == 50:
        path_G = "./new_improved_BraTS23_G.pt"
    elif epoch == 100:
        path_G = "./ep100_improved_loss_BraTS23_G.pt"
    else:
        raise ValueError("Unsupported epoch. Use 25, 50, 75 or 100.")

    # Automatically handle devices (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model weights and map to the correct device
    ckp_G = torch.load(path_G, map_location=device)
    G.load_state_dict(ckp_G['state_dict'], strict=True)

    # Move the model to the correct device (defaults to CPU if no GPU is available)
    G = G.to(device)
    
    print(f"Loaded Pretrained Model (Specific GAN {epoch}epoch) on {device.type.upper()}")
    return G


def helper():
    G_ep25 = get_G(2,100,25)
    G_ep50 = get_G(2,100,50)
    G_ep75 = get_G(2,100,75)
    G_ep75.eval()
    G_ep25.eval()
    G_ep50.eval()
    noise = torch.randn(1, 100)
    with torch.no_grad():
        generated_image_ep25 = G_ep25(noise)
        generated_image_ep50 = G_ep50(noise)
        generated_image_ep75 = G_ep75(noise)
        print(generated_image_ep25.shape)
    generated_image_ep25 = generated_image_ep25.squeeze(0).cpu().numpy()
    generated_image_ep50 = generated_image_ep50.squeeze(0).cpu().numpy()
    generated_image_ep75 = generated_image_ep75.squeeze(0).cpu().numpy()
    generated_image_ep25 = np.squeeze(generated_image_ep25)  # Remove the channel dimension for grayscale
    generated_image_ep50 = np.squeeze(generated_image_ep50)
    generated_image_ep75 = np.squeeze(generated_image_ep75)
            # Plot the generated image
    plt.imshow(generated_image_ep25, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.savefig('generated_image_ep25.png')
    plt.imshow(generated_image_ep50, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.savefig('generated_image_ep50.png')
    plt.imshow(generated_image_ep75, cmap='gray')
    plt.axis('off')  # Turn off axis labels
    plt.savefig('generated_image_ep75.png')
    

# if __name__ == "__main__":
#     helper()
#     # print(G)

# Streamlit App
def main():
    st.title("GAN Image Generator")
    st.markdown("Select the number of epochs to load the pretrained model and generate PNG images.")

    # Input options for the user
    epoch = st.selectbox("Select Epochs:", [25, 50, 75,100], index=0)
    n_images = st.slider("Number of Images to Generate:", min_value=1, max_value=10, value=5)

    # Button to generate images
    if st.button("Generate Images"):
        st.info("Loading the model and generating images...")

        try:
            # Load the model
            z_dim = 100  # Define latent dimension
            G = get_G(z_dim, epoch)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Generate images
            G.eval()
            z = torch.randn(n_images, z_dim).to(device)  # Random latent vectors
            with torch.no_grad():
                generated_images = G(z).cpu()  # Generate images
            
            # Save and display images
            output_dir = Path("generated_images")
            output_dir.mkdir(exist_ok=True)
            st.success("Images generated successfully!")
            
            for i, img in enumerate(generated_images):
                file_path = output_dir / f"image_{i + 1}.png"
                save_image(img, file_path)
                st.image(str(file_path), caption=f"Generated Image {i + 1}", use_column_width=True)
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
