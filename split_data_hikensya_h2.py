import os
import csv
import shutil

train_file = 'DATASETS/Emotion6/csv_emotion6/ground_truth_train.csv'
test_file = 'DATASETS/Emotion6/csv_emotion6/ground_truth_test.csv'
output_path = 'DATASETS/Emotion6/split_data/hikensya/'
newpath = ''
newname = ''

# myClass = {"1":"anger", "2":"disgust", "3":"fear", "4":"joy", "5":"sadness", "6":"surprise", "7":"neutral"}

def duplicate_rename2(file_path):
    if os.path.exists(file_path):
        name, ext = os.path.splitext(file_path)
        i = 1
        while True:
            # 数値を3桁などにしたい場合は({:0=3})とする
            new_name = "{} ({}){}".format(name, i, ext)
            if not os.path.exists(new_name):
                return new_name
            i += 1
    else:
        return file_path

with open(train_file) as f:
    reader = csv.reader(f)
    l = [row for row in reader]
    image_path = 'DATASETS/Emotion6/split_data/train_images/'
    dirs = os.listdir(image_path)
    folder_dir = [f for f in dirs if os.path.isdir(os.path.join(image_path, f))]

for i in range(len(l)-1):
    image_name = l[i+1][0].replace('.jpg', '')
    for j in range(len(folder_dir)):
        if (image_name.split('/')[0]) in folder_dir[j]:
            infe_image_path = image_path + l[i+1][0]
            if not os.path.exists(os.path.join(output_path, folder_dir[j])):
                os.makedirs(os.path.join(output_path, folder_dir[j]))

count = 0; count2 = 0
neutral_path = os.path.join(output_path, "neutral")
if not os.path.exists(neutral_path):
    os.makedirs(neutral_path)

for i in range(len(l)-1):
    image_name = l[i+1][0].replace('.jpg', '')
    for j in range(len(folder_dir)):
        if (image_name.split('/')[0]) in folder_dir[j]:
            infe_image_path = image_path + l[i+1][0]
            dirpath, filename = os.path.split(infe_image_path)
            
            if l[i+1][1] == "7":
                if os.path.exists(os.path.join(neutral_path, filename)):
                    newname = '{}({}){}'.format(name, j+1, ext)
                    print('new:{} -> {}'.format(infe_image_path, os.path.join(neutral_path, newname)))
                    shutil.copy(infe_image_path, os.path.join(neutral_path, newname))
                else:
                    print('base:{} -> {}'.format(infe_image_path, os.path.join(neutral_path, filename)))
                    shutil.copy(infe_image_path, neutral_path)
                count+=1
            else:
                x = int(l[i+1][1])
                # 移動先のファイルが既に存在する場合は、代わりの名前を見つける
                name, ext = os.path.splitext(filename)
                if os.path.exists(os.path.join(output_path, folder_dir[x-1], filename)):
                    newname = '{}({}){}'.format(name, j, ext)
                    newpath = os.path.join(output_path, folder_dir[x-1])
                    print('new:{} -> {}'.format(infe_image_path, os.path.join(newpath, newname)))
                    shutil.copy(infe_image_path, os.path.join(newpath, newname))
                else:
                    print('base:{} -> {}'.format(infe_image_path, os.path.join(output_path, folder_dir[x-1], filename)))
                    shutil.copy(infe_image_path, os.path.join(output_path, folder_dir[x-1]))
            count2+=1
        
print(count)
print(count2)