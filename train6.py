import os
import glob
import datetime
import numpy as np

from PIL import Image
from tensorflow.keras.layers import Input
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import models, layers
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import DenseNet201, EfficientNetB7, InceptionResNetV2, InceptionV3, MobileNet, EfficientNetB0
from tensorflow.keras.applications import MobileNetV2, NASNetMobile, ResNet101, ResNet50, ResNet50V2, VGG16, VGG19, Xception

# -------------------------------------------------------------------------------------
#                        初期設定部
# -------------------------------------------------------------------------------------

# 事前学習済みモデル名
modelName = EfficientNetB0

# GrayScaleのときに1、COLORのときに3にする
COLOR_CHANNEL = 3

# 入力画像サイズ(画像サイズは正方形とする)
INPUT_IMAGE_SIZE = 224

# 訓練時のバッチサイズとエポック数
BATCH_SIZE = 16
EPOCH_NUM = 200

# 使用する訓練画像の各クラスのフォルダ名

folder = ["anger",
            "disgust",
            "fear",
            "joy",
            "sadness",
            "surprise",
            "neutral"] #分類するクラス


# CLASS数を取得する
CLASS_NUM = len(folder)
print("class_num:" + str(CLASS_NUM))

# -------------------------------------------------------------------------------------
#                        関数部
# -------------------------------------------------------------------------------------

def show(folder_name):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(folder_name, 'acc.png'))
    plt.clf()
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(folder_name,'loss.png'))
    plt.clf()
    plt.close()

# -------------------------------------------------------------------------------------
#                        訓練画像入力部
# -------------------------------------------------------------------------------------

#画像データの配列を用意
X = []

#ラベルのデータを用意
Y = []

train_dir = "DATASETS/Emotion6/split_data/hikensya/"

#画像データとラベルデータに分割
for index, classlabel in enumerate(folder):
    photo_dir =  train_dir + classlabel
    files = glob.glob(photo_dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE))
        data = np.asarray(image) # リストをNumpy配列に変換 # https://punhundon-lifeshift.com/array_asarray
        X.append(data)
        Y.append(index)
X = np.array(X) # https://punhundon-lifeshift.com/array_asarray
Y = np.array(Y)

#画像データを0~1の値へ変換
X = X.astype('float32')
X = X / 255.0

#正解ラベルの形式を変換
Y = to_categorical(Y, CLASS_NUM)

# 学習用データと検証用データに分割する
train_images, valid_images, train_labels, valid_labels = train_test_split(X, Y, test_size = 0.30)

# -------------------------------------------------------------------------------------
#                      モデルアーキテクチャ定義部
# -------------------------------------------------------------------------------------

input_tensor = Input(shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3))

base_model = modelName(
    include_top=False,
    weights='imagenet',
    input_shape=(INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3),
    pooling='max')

# パラメータ凍結
for layer in base_model.layers:
    layer.trainable = False

# モデルの構築
x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.6)(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.6)(x)
x = layers.Dense(CLASS_NUM, activation='softmax')(x)
model = models.Model(base_model.input, x)

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

# モデル構成の確認
# model.summary()

# EaelyStoppingの設定(学習が進まなくなったら学習を止める)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')

# 訓練
#history = model.fit(train_images, train_labels, validation_data = (valid_images, valid_labels),
#                    batch_size=BATCH_SIZE, epochs=EPOCH_NUM, callbacks=[early_stopping])
# trainの実行
history = model.fit(train_images, train_labels, validation_split = 0.30,
                    batch_size=BATCH_SIZE, epochs=EPOCH_NUM, callbacks=[early_stopping])


# -------------------------------------------------------------------------------------
#                              訓練実行&結果確認部
# -------------------------------------------------------------------------------------

score = model.evaluate(valid_images, valid_labels, verbose=0)
print(len(valid_images))
print('Loss:', score[0])
print('Accuracy:', score[1])

time_now = str(datetime.datetime.now().strftime("%d%H%M"))

folder = 'emotion6' + '_' + time_now
folder_name = os.path.join('models', folder)
os.mkdir(folder_name)
show(folder_name)

"""
y_pred_test = model.predict(valid_images)
cm = confusion_matrix(valid_labels, y_pred_test)
sns.heatmap(cm)
plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))
"""

# --------------------------------------------------------------------------------------
# モデルの保存
# --------------------------------------------------------------------------------------

model.save(os.path.join(folder_name, time_now + '.h5'))
model.save_weights(os.path.join(folder_name, 'model_emotion6.hdf5'))
