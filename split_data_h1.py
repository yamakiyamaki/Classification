import csv
import os
import numpy as np
from PIL import Image
import shutil

train_file = 'DATASETS/Emotion6/csv_emotion6/ground_truth_train.csv'
test_file = 'DATASETS/Emotion6/csv_emotion6/ground_truth_test.csv'
output_path = 'DATASETS/Emotion6/split_data/'
"""
with open(train_file) as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    image_path = 'DATASETS/Emotion6/Images/'
    dirs = os.listdir(image_path)
    folder_dir = [f for f in dirs if os.path.isdir(os.path.join(image_path, f))]
for i in range(len(l)-1):
    image_name = l[i+1][0].replace('.jpg', '')
    for j in range(len(folder_dir)):
        if (image_name.split('/')[0]) in folder_dir[j]:
            infe_image_path = image_path + l[i+1][0]
            if not os.path.exists(os.path.join(output_path, 'train_images', folder_dir[j])):
                os.makedirs(os.path.join(output_path, 'train_images', folder_dir[j]))
            shutil.copy(infe_image_path, os.path.join(output_path, 'train_images', folder_dir[j]))
"""

with open(test_file) as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    image_path = 'DATASETS/Emotion6/Images/'
    dirs = os.listdir(image_path)
    folder_dir = [f for f in dirs if os.path.isdir(os.path.join(image_path, f))]
for i in range(len(l)-1):
    image_name = l[i+1][0].replace('.jpg', '')
    for j in range(len(folder_dir)):
        if (image_name.split('/')[0]) in folder_dir[j]:
            infe_image_path = image_path + l[i+1][0]
            dirpath, filename = os.path.split(infe_image_path)
            
            if not os.path.exists(os.path.join(output_path, 'test')):
                os.makedirs(os.path.join(output_path, 'test'))
            # 移動先のファイルが既に存在する場合は、代わりの名前を見つける。
            name, ext = os.path.splitext(filename)
            if os.path.exists(os.path.join(output_path, 'test', filename)):
                newname = '{}({}){}'.format(name, j+1, ext)
                newpath = os.path.join(output_path, 'test')
                print('new:{} -> {}'.format(infe_image_path, os.path.join(newpath, newname)))
                shutil.copy(infe_image_path, os.path.join(newpath, newname))
            else:
                print('base:{} -> {}'.format(infe_image_path, os.path.join(output_path, 'test', filename)))
                shutil.copy(infe_image_path, os.path.join(output_path, 'test'))