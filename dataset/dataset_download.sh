#!/bin/bash

kaggle datasets download -d daehoyang/flickr2k
kaggle datasets download -d sharansmenon/div2k

tar -zxvf flickr2k.zip
tar -zxvf div2k.zip

mkdir -p Images

cp DIV2K_train_HR/*.png Images/
cp DIV2K_valid_HR/*.png Images/
cp Flickr2K/*.png Images/

rm -r DIV2K_train_HR DIV2K_valid_HR Flickr2K

python utils/split_dataset.py --src_folder Images/ --dst_folder SR/ --train_ratio 0.85 --val_ratio 0.10 --test_ratio 0.05

python utils/patch_generation.py --src_folder SR/ --dst_folder SR_patches/ --patch_size 512 --stride 256