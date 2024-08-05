from generator import Generator
from discriminator import Discriminator
from loss import VGGLoss
from dataset import load_dataset
import os
from torch import optim as opt
import torch
from tqdm import tqdm
import argparse
from logger import LOGWRITER
import math

def calculate_psnr(sr_img, hr_img):
    mse = torch.mean((sr_img - hr_img) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def pretraining(generator, train_dl, opt_g, epochs, device, logger, save_dir):
    mse_criterion = torch.nn.MSELoss()
    
    logger.write("[INFO] Pretraining Generator with MSE Loss.")
    
    for epoch in range(epochs):

        generator.train()
        total_train_loss_g = 0

        for hr_imgs, lr_imgs in tqdm(train_dl, desc=f"Pretraining Epoch {epoch+1}/{epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            opt_g.zero_grad()
            sr_imgs = generator(lr_imgs)

            g_loss_content = mse_criterion(sr_imgs, hr_imgs)
            
            g_loss_content.backward()
            
            opt_g.step()
            total_train_loss_g += g_loss_content.item()
        
        avg_train_loss_g = total_train_loss_g / len(train_dl)
        logger.log_results(epoch=epoch+1, tr_loss_g=avg_train_loss_g)

    logger.write("[INFO] Pretraining Generator with MSE Loss finished")
    save_path_g = os.path.join(save_dir, 'Pretrained_generator.pth')
    torch.save(generator.state_dict(), save_path_g)

    return generator

def train_esrgan(train_dl, valid_dl, save_dir, epochs, warmup, logger, device='cuda'):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    optimizer_g = opt.Adam(generator.parameters(), lr=2e-4, betas=(0.9, 0.999))
    optimizer_d = opt.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    scheduler_g = opt.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=int(epochs * 0.9), eta_min=6e-6)
    scheduler_d = opt.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=int(epochs * 0.9), eta_min=6e-6)
    
    mse_criterion = torch.nn.MSELoss()
    bce_criterion = torch.nn.BCEWithLogitsLoss()
    vgg_criterion = VGGLoss().to(device)
    
    best_loss = float('inf')
    
    pretraining(generator, train_dl, optimizer_g, warmup, device, logger, save_dir)

    for epoch in range(epochs):

        generator.train()
        discriminator.train()
        total_train_loss_g = 0
        total_train_loss_d = 0

        # Training loop
        for hr_imgs, lr_imgs in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            sr_imgs = generator(lr_imgs)
            real_out = discriminator(hr_imgs)
            fake_out = discriminator(sr_imgs.detach())
            
            d_loss_real = bce_criterion(real_out - torch.mean(fake_out), torch.ones_like(real_out))
            d_loss_fake = bce_criterion(fake_out - torch.mean(real_out), torch.zeros_like(fake_out))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            
            optimizer_d.step()
            total_train_loss_d += d_loss.item()

            # Train Generator
            optimizer_g.zero_grad()
            sr_imgs = generator(lr_imgs)
            fake_out = discriminator(sr_imgs)
            
            g_loss_adv = bce_criterion(fake_out - torch.mean(real_out.detach()), torch.ones_like(fake_out))
            g_loss_content = mse_criterion(sr_imgs, hr_imgs)
            g_loss_perceptual = vgg_criterion(sr_imgs, hr_imgs)
            g_loss = 5e-3 * g_loss_adv + g_loss_perceptual + 1e-2 * g_loss_content  # Weighted loss
            g_loss.backward()
            
            optimizer_g.step()
            total_train_loss_g += g_loss.item()

        generator.eval()
        discriminator.eval()
        total_val_loss_g = 0
        total_val_loss_d = 0
        total_psnr = 0

        # Validation Loop
        with torch.no_grad():
            for hr_imgs, lr_imgs in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = generator(lr_imgs)

                real_out = discriminator(hr_imgs)
                fake_out = discriminator(sr_imgs)
                d_loss_real = bce_criterion(real_out - torch.mean(fake_out), torch.ones_like(real_out))
                d_loss_fake = bce_criterion(fake_out - torch.mean(real_out), torch.zeros_like(fake_out))
                d_loss = (d_loss_real + d_loss_fake) / 2
                total_val_loss_d += d_loss.item()

                g_loss_adv = bce_criterion(fake_out - torch.mean(real_out.detach()), torch.ones_like(fake_out))
                g_loss_content = mse_criterion(sr_imgs, hr_imgs)
                g_loss_perceptual = vgg_criterion(sr_imgs, hr_imgs)
                g_loss = 5e-3 * g_loss_adv + g_loss_perceptual + 1e-2 * g_loss_content  # Weighted loss
                total_val_loss_g += g_loss.item()

                # Calculate PSNR
                total_psnr += calculate_psnr(sr_imgs, hr_imgs)

        avg_train_loss_g = total_train_loss_g / len(train_dl)
        avg_train_loss_d = total_train_loss_d / len(train_dl)
        avg_val_loss_g = total_val_loss_g / len(valid_dl)
        avg_val_loss_d = total_val_loss_d / len(valid_dl)
        avg_psnr = total_psnr / len(valid_dl)

        if avg_val_loss_g < best_loss:
            best_loss = avg_val_loss_g
            save_path_g = os.path.join(save_dir, f'Best_generator_{epoch+1}.pth')
            save_path_d = os.path.join(save_dir, f'Best_discriminator_{epoch+1}.pth')
            torch.save(generator.state_dict(), save_path_g)
            torch.save(discriminator.state_dict(), save_path_d)

        logger.log_results(epoch=epoch+1, tr_loss_g=avg_train_loss_g, tr_loss_d=avg_train_loss_d, val_loss_g=avg_val_loss_g, val_loss_d=avg_val_loss_d, psnr=avg_psnr)
        scheduler_g.step()
        scheduler_d.step()

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="root directory for imgs, requires train and val split")
    parser.add_argument("--save", type=str, required=True, help="save directory for weights and log files") 
    parser.add_argument("--epochs", type=int, required=True, help="total epochs")
    parser.add_argument("--warmup", type=int, required=True, help="total epochs for pretraining phase of epochs")
    parser.add_argument("--upscale", type=int, required=False, default=4, help="upscale factor to train ESRGAN on")
    
    args = parser.parse_args() 
    
    tr_dl = load_dataset(root_dir=args.root, mode="train", batch_size=16, scale_factor=args.upscale)
    val_dl = load_dataset(root_dir=args.root, mode="val", batch_size=16, scale_factor=args.upscale)
    
    weight_path = os.path.join(args.save, "saved_weights")
    os.makedirs(weight_path, exist_ok=True)
    
    log_path = os.path.join(args.save, "logs")
    os.makedirs(log_path, exist_ok=True)
    
    logger = LOGWRITER(output_directory=log_path, total_epochs=args.epochs)
    
    train_esrgan(train_dl=tr_dl, valid_dl=val_dl, save_dir=weight_path, epochs=args.epochs, warmup=args.warmup, logger=logger)