import os
import argparse
import random
import shutil
from glob import glob

def split_dataset(src_folder, dst_folder, train_ratio, val_ratio, test_ratio, seed=None):
    if seed is not None:
        random.seed(seed)
    
    os.makedirs(dst_folder, exist_ok=True)

    os.makedirs(os.path.join(dst_folder, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, 'val'), exist_ok=True)
    os.makedirs(os.path.join(dst_folder, 'test'), exist_ok=True)

    files = glob(os.path.join(src_folder, '*.png'))

    random.shuffle(files)

    total_files = len(files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    test_count = total_files - train_count - val_count

    print(f"Total files: {total_files}")
    print(f"Training files: {train_count}")
    print(f"Validation files: {val_count}")
    print(f"Test files: {test_count}")

    for i, file_path in enumerate(files):
        if i < train_count:
            shutil.copy(file_path, os.path.join(dst_folder, 'train'))
        elif i < train_count + val_count:
            shutil.copy(file_path, os.path.join(dst_folder, 'val'))
        else:
            shutil.copy(file_path, os.path.join(dst_folder, 'test'))

def main():
    parser = argparse.ArgumentParser(description="Split a dataset of PNG images into train, val, and test sets.")
    parser.add_argument('--src_folder', type=str, required=True, help="Source folder containing PNG images.")
    parser.add_argument('--dst_folder', type=str, required=True, help="Destination folder to save the split datasets.")
    parser.add_argument('--train_ratio', type=float, default=0.7, help="Ratio of training set. Default is 0.7")
    parser.add_argument('--val_ratio', type=float, default=0.2, help="Ratio of validation set. Default is 0.2")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="Ratio of test set. Default is 0.1")
    parser.add_argument('--seed', type=int, help="Random seed for reproducibility. Default is None", default=None)

    args = parser.parse_args()

    if args.train_ratio + args.val_ratio + args.test_ratio != 1.0:
        raise ValueError("The sum of train_ratio, val_ratio, and test_ratio must be 1.0")

    split_dataset(args.src_folder, args.dst_folder, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

if __name__ == "__main__":
    main()
