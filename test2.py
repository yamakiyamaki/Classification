import tensorflow as tf
from PIL import Image
import csv
import os
import cv2
import numpy as np

if __name__ == "__main__":
    
    submission_list = [['image', 'label']]
    image_size = 224
    time_number = '190344'
    
    # 同じモデルを読み込んで、重みやオプティマイザーを含むモデル全体を再作成
    #new_model = tf.keras.models.load_model('saved_model/my_model')
    new_model = tf.keras.models.load_model('./models/emotion6_{}/{}.h5'.format(time_number, time_number))
    
    test_file = "DATASETS/Emotion6/csv_emotion6/ground_truth_test.csv"
    with open(test_file) as f:
        reader = csv.reader(f)
        l = [row for row in reader] # 0番目は['Image', 'Label']
        image_path = 'DATASETS/Emotion6/Images/'
        dirs = os.listdir(image_path)
        folder_dir = [f for f in dirs if os.path.isdir(os.path.join(image_path, f))]
    for i in range(len(l)-1):
        image_name = l[i+1][0].replace('.jpg', '')

        # 推論用画像があるディレクトリを探索
        for j in range(len(folder_dir)):
            if (image_name.split('/')[0]) in folder_dir[j]:
                infe_image_path = image_path + '/' + l[i+1][0]
                image = cv2.imread(infe_image_path)
                image = cv2.resize(image, (image_size, image_size))
                image = image.reshape(1, image_size, image_size, 3)
                data = np.asarray(image) #画像データを配列へ変更
                # print(data.shape)
                predictions = new_model.predict(data)
                results = np.argmax(predictions[0]) + 1
                # print(results)
                submission_list.append([l[i+1][0], results])
    
        if(i % 50 == 0):
            print('test_file_number:{}'.format(i))

    # 提出用csv書き込み
    with open('./CSV/submission_{}_{}.csv'.format(time_number), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(submission_list)