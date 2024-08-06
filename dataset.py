import random
from glob import glob
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import os

class ImageDataset(Dataset):
    """
    Args:
        root_dir (str): The directory containing the images.
        patch_size (int): The patch size for a patch_size by patch_size patch.
        scale_factor (float): The factor by which to downscale the images for degradation.
        mode (str): The mode indicating whether the dataset is for training or validation.
    """
    def __init__(self, root_dir, patch_size, scale_factor, mode: str):
        self.clean_dir = sorted(glob(os.path.join(root_dir, mode) + "/*"))
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        self.hr_transformer = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.lr_transformer = T.ToTensor()

        self.to_pil = T.ToPILImage()

    def __len__(self):
        return len(self.clean_dir)

    def __getitem__(self, index):
        clean_img = Image.open(self.clean_dir[index]).convert("RGB")

        img_w, img_h = clean_img.size

        if self.patch_size > img_w or self.patch_size > img_h:
            raise ValueError("[INFO] Patch size is larger than the image dimensions")

        start_x = random.randint(0, img_w - self.patch_size)
        start_y = random.randint(0, img_h - self.patch_size)

        clean_img_patch = clean_img.crop((start_x, start_y, start_x + self.patch_size, start_y + self.patch_size))

        lr_img_patch = clean_img_patch.resize(
            (int(self.patch_size / self.scale_factor), int(self.patch_size / self.scale_factor)),
            Image.BICUBIC
        )
        
        clean_img_patch = self.hr_transformer(clean_img_patch)
        lr_img_patch = self.lr_transformer(lr_img_patch)

        if random.random() > 0.5:
            clean_img_patch = T.functional.hflip(clean_img_patch)
            lr_img_patch = T.functional.hflip(lr_img_patch)

        if random.random() > 0.5:
            clean_img_patch = T.functional.vflip(clean_img_patch)
            lr_img_patch = T.functional.vflip(lr_img_patch)

        return clean_img_patch, lr_img_patch

def load_dataset(root_dir: str, mode: str, batch_size: int = 32, patch_size: int = 128, scale_factor: int = 4):
    ds = ImageDataset(root_dir=root_dir, mode=mode, patch_size=patch_size, scale_factor=scale_factor)
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, num_workers=8)