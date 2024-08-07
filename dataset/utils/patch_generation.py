import argparse
import cv2
import os
import shutil
from tqdm import tqdm

def generate_patches(image_path, patch_size, stride, save_dir, count):
    image = cv2.imread(image_path)

    height, width, _ = image.shape

    patch_count = count
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_filename = os.path.join(save_dir, f'{patch_count:04d}.png')
            cv2.imwrite(patch_filename, patch)
            patch_count += 1

    return patch_count

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate patches from images in subfolders')
    parser.add_argument('--src_folder', type=str, help='Path to the main folder')
    parser.add_argument('--dst_folder', type=str, help='Directory to save the patches')
    parser.add_argument('--patch_size', type=int, default=64, help='Size of each patch (default: 64)')
    parser.add_argument('--stride', type=int, default=32, help='Stride between patches (default: 32)')
    args = parser.parse_args()

    subfolders = [f for f in os.listdir(args.src_folder) if os.path.isdir(os.path.join(args.src_folder, f))]

    total_patches = 0

    for subfolder in subfolders:
        subfolder_path = os.path.join(args.src_folder, subfolder)
        save_subfolder_path = os.path.join(args.dst_folder, subfolder)
        os.makedirs(save_subfolder_path, exist_ok=True)

        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not f.startswith('._')]

        patch_count = 1

        progress_bar = tqdm(image_files, desc=f'Processing {subfolder}', leave=False)
        for image_file in progress_bar:
            image_path = os.path.join(subfolder_path, image_file)
            patch_count = generate_patches(image_path, args.patch_size, args.stride, save_subfolder_path, patch_count)

        total_patches += patch_count - 1

    print(f'Saved {total_patches:04d} patches from {len(subfolders)} subfolders to {args.dst_folder}.')
