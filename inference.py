import os
import torch
import argparse
from tqdm import tqdm
from glob import glob
from PIL import Image
from torchvision import transforms as T
from ESRGAN.generator import Generator
import matplotlib.pyplot as plt
import numpy as np

def main(root_dir: str, output_dir: str, resize: bool):
    """
    Main function to perform super-resolution on images in a directory.
    
    Parameters:
    - root_dir: str : Directory containing input images.
    - output_dir: str : Directory to save super-resolved images.
    - resize: bool : Whether to resize super-resolved images to the original image size.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(script_dir, "Weights/Best_generator.pth")
    
    model = Generator().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    img_dir = sorted(glob(os.path.join(root_dir, "*")))
    
    for img_path in tqdm(img_dir, desc="[INFO] Super-Resolution: "):
        img = Image.open(img_path).convert("RGB")
        
        img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(img_tensor)
        
        sr_tensor = (sr_tensor + 1) / 2
        
        sr_tensor = torch.clamp(sr_tensor, 0, 1)
        
        sr_numpy = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        
        sr_img = Image.fromarray((sr_numpy * 255).astype(np.uint8))
        
        base_name = os.path.basename(img_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_SR{ext}")
                
        if resize:
            sr_img = sr_img.resize((img.width, img.height), Image.BICUBIC)
            sr_img.save(output_path)
        else:
            sr_img.save(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Super-resolution on images using a pre-trained model.")
    
    parser.add_argument('--root_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save super-resolved images.")
    parser.add_argument('--resize', action='store_true', help="Whether to resize super-resolved images to the original image size.")
    
    args = parser.parse_args()
    
    main(args.root_dir, args.output_dir, args.resize)