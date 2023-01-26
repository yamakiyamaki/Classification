import os
import glob
import keras
import datetime
import numpy as np
import seaborn as sns
import tensorflow as tf

from PIL import Image
import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras import models, layers
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201, EfficientNetB4, InceptionResNetV2, InceptionV3, MobileNet
from tensorflow.keras.applications import MobileNetV2, NASNetMobile, ResNet101, ResNet50, ResNet50V2, VGG16, VGG19, Xception, EfficientNetB0

# -------------------------------------------------------------------------------------
#                        初期設定部
# -------------------------------------------------------------------------------------

# 事前学習済みモデル名
modelName = EfficientNetB4

# GrayScaleのときに1、COLORのときに3にする
COLOR_CHANNEL = 3

# 入力画像サイズ(画像サイズは正方形とする)
INPUT_IMAGE_SIZE = (224, 224)

# 訓練時のバッチサイズとエポック数
BATCH_SIZE = 8
EPOCH_NUM = 50

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

datagen = keras.preprocessing.image.ImageDataGenerator(shear_range=0.1, zoom_range=0.1, horizontal_flip=True,
                                                       preprocessing_function=preprocess_input, validation_split=0.3)
train = datagen.flow_from_directory(train_dir, batch_size=BATCH_SIZE, target_size=INPUT_IMAGE_SIZE, class_mode='categorical', subset='training')
valid = datagen.flow_from_directory(train_dir, batch_size=BATCH_SIZE, target_size=INPUT_IMAGE_SIZE, class_mode='categorical', subset='validation')
print(train)
print(valid)

# -------------------------------------------------------------------------------------
#                      モデルアーキテクチャ定義部
# -------------------------------------------------------------------------------------

input_tensor = Input(shape=(INPUT_IMAGE_SIZE+ (3,)))

base_model = modelName(
    include_top=False,
    weights='imagenet',
    input_shape=(INPUT_IMAGE_SIZE+(3,)),
    pooling='average')

"""
K.clear_session()
tf.random.set_seed(0)
"""

# パラメータ凍結
for layer in base_model.layers:
    layer.trainable = False

# モデルの構築
x = preprocess_input(input_tensor)
x = base_model.output
"""
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.6)(x)
x = layers.Flatten()(x)
"""
x = keras.layers.GlobalAveragePooling2D()(x)
"""
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.6)(x)
"""
x = BatchNormalization()(x)
x = layers.Dense(CLASS_NUM, activation='softmax')(x)
model = models.Model(base_model.input, x)

# コンパイル
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adamax(learning_rate=0.0005),
              metrics=['accuracy'])

# EaelyStoppingの設定
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

# 訓練
history = model.fit(train, validation_data=valid,
                    batch_size=BATCH_SIZE, epochs=EPOCH_NUM, callbacks=[early_stopping],
                    class_weight={0:20.0,
                                  1:2.5,
                                  2:2.0,
                                  3:1.0,
                                  4:2.1,
                                  5:6.3,
                                  6:2.0})

# -------------------------------------------------------------------------------------
#                              訓練実行&結果確認部
# -------------------------------------------------------------------------------------

score = model.evaluate(valid, verbose=0)
print(len(valid))
print('Loss:', score[0])
print('Accuracy:', score[1])

time_now = str(datetime.datetime.now().strftime("%d%H%M"))

folder = 'emotion6' + '_' + time_now
folder_name = os.path.join('models', folder)
os.mkdir(folder_name)
show(folder_name)

# -------------------------------------------------------------------------------------
#                              混同行列とHeatmapの出力
# -------------------------------------------------------------------------------------

y_pred_test = model.predict(valid) 
valid_labels_list = valid.argmax(axis=1) # hot-one形式の二次元配列を一次元配列に変換
pred_labels_list = y_pred_test.argmax(axis=1)
# print("vList", valid_labels_list)
# print("pList", pred_labels_list)

cm = confusion_matrix(valid_labels_list, pred_labels_list)
sns.heatmap(cm, annot=True, square= True, cbar=True, cmap='Blues', fmt='d')
plt.xlabel("Predicted Class", fontsize=13)
plt.ylabel("True Class", fontsize=13)
plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))

# --------------------------------------------------------------------------------------
#                                   モデルの保存
# --------------------------------------------------------------------------------------

model.save(os.path.join(folder_name, time_now + '.h5'))
model.save_weights(os.path.join(folder_name, 'model_emotion6.hdf5'))