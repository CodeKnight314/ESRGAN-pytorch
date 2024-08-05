from generator import Generator
from discriminator import Discriminator
from loss import VGGLoss, EdgeAwareLoss
from dataset import load_dataset
import os
from torch import optim as opt
import torch
from tqdm import tqdm
import argparse
from logger import LOGWRITER

def train_esrgan(train_dl, valid_dl, save_dir, epochs, warmup, logger, device='cuda'):
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    optimizer_g = opt.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_d = opt.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    scheduler_g = opt.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=int(epochs * 0.8), eta_min=1e-6)
    scheduler_d = opt.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=int(epochs * 0.8), eta_min=1e-6)
    
    l1_criterion = torch.nn.L1Loss()
    mse_criterion = torch.nn.MSELoss()
    vgg_criterion = VGGLoss().to(device)
    gradient_criterion = EdgeAwareLoss().to(device)
    
    best_loss = float('inf')

    for epoch in range(epochs):

        generator.train()
        discriminator.train()
        total_train_loss_g = 0
        total_train_loss_d = 0

        # Training loop
        for lr_imgs, hr_imgs in tqdm(train_dl, desc=f"Training Epoch {epoch+1}/{epochs}"):
            lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

            # Train Discriminator
            optimizer_d.zero_grad()
            sr_imgs = generator(lr_imgs)
            real_out = discriminator(hr_imgs)
            fake_out = discriminator(sr_imgs.detach())
            d_loss_real = mse_criterion(real_out - torch.mean(fake_out), torch.ones_like(real_out))
            d_loss_fake = mse_criterion(fake_out - torch.mean(real_out), torch.zeros_like(fake_out))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()
            total_train_loss_d += d_loss.item()

            # Train Generator
            optimizer_g.zero_grad()
            sr_imgs = generator(lr_imgs)
            fake_out = discriminator(sr_imgs)
            g_loss_adv = mse_criterion(fake_out - torch.mean(real_out), torch.ones_like(fake_out))
            g_loss_content = l1_criterion(sr_imgs, hr_imgs)
            g_loss_edge = gradient_criterion(sr_imgs, hr_imgs)
            g_loss_perceptual = vgg_criterion(sr_imgs, hr_imgs, device)
            g_loss = 1e-3 * g_loss_adv + 1e-2 * g_loss_perceptual + 1e-2 * g_loss_edge + g_loss_content  # Weighted loss
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_g.step()
            total_train_loss_g += g_loss.item()

        generator.eval()
        discriminator.eval()
        total_val_loss_g = 0
        total_val_loss_d = 0

        # Validation Loop
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(valid_dl, desc=f"Validating Epoch {epoch+1}/{epochs}"):
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                sr_imgs = generator(lr_imgs)

                real_out = discriminator(hr_imgs)
                fake_out = discriminator(sr_imgs)
                d_loss_real = mse_criterion(real_out - torch.mean(fake_out), torch.ones_like(real_out))
                d_loss_fake = mse_criterion(fake_out - torch.mean(real_out), torch.zeros_like(fake_out))
                d_loss = (d_loss_real + d_loss_fake) / 2
                total_val_loss_d += d_loss.item()

                g_loss_adv = mse_criterion(fake_out - torch.mean(real_out), torch.ones_like(fake_out))
                g_loss_content = l1_criterion(sr_imgs, hr_imgs)
                g_loss_perceptual = vgg_criterion(sr_imgs, hr_imgs, device)
                g_loss_edge = gradient_criterion(sr_imgs, hr_imgs)
                g_loss = 1e-3 * g_loss_adv + 1e-2 * g_loss_perceptual + 1e-2 * g_loss_edge + g_loss_content  # Weighted loss
                total_val_loss_g += g_loss.item()

        avg_train_loss_g = total_train_loss_g / len(train_dl)
        avg_train_loss_d = total_train_loss_d / len(train_dl)
        avg_val_loss_g = total_val_loss_g / len(valid_dl)
        avg_val_loss_d = total_val_loss_d / len(valid_dl)

        if avg_val_loss_g < best_loss:
            best_loss = avg_val_loss_g
            save_path_g = os.path.join(save_dir, f'Best_generator_{epoch+1}.pth')
            save_path_d = os.path.join(save_dir, f'Best_discriminator_{epoch+1}.pth')
            torch.save(generator.state_dict(), save_path_g)
            torch.save(discriminator.state_dict(), save_path_d)

        logger.log_results(epoch=epoch+1, tr_loss_g=avg_train_loss_g, tr_loss_d=avg_train_loss_d, val_loss_g=avg_val_loss_g, val_loss_d=avg_val_loss_d)

        if epoch > warmup:
            scheduler_g.step()
            scheduler_d.step()
            
if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="root directory for imgs, requires train and val split")
    parser.add_argument("--save", type=str, required=True, help="save directory for weights and log files") 
    parser.add_argument("--epochs", type=int, required=True, help="total epochs")
    parser.add_argument("--warmup", type=int, required=True, help="total epochs for warm up phase of epochs")
    parser.add_argument("--upscale", type=int, required=False, default=2, help="upscale factor to train ESRGAN on")
    
    args = parser.parse_args() 
    
    tr_dl = load_dataset(root_dir=args.root, mode="train", batch_size=32, scale_factor=args.upscale)
    val_dl = load_dataset(root_dir=args.root, mode="val", batch_size=32, scale_factor=args.upscale)
    
    weight_path = os.makedirs(os.path.join(args.save, "saved_weights"), exist_ok=True)
    log_path = os.makedirs(os.path.join(args.save, "logs"), exist_ok=True)
    
    logger = LOGWRITER(output_directory=log_path, total_epochs=args.epochs)
    
    train_esrgan(train_dl=tr_dl, valid_dl=val_dl, save_dir=weight_path, epochs=args.epochs, warmup=args.warmup, logger=logger)
    
    
    
    
