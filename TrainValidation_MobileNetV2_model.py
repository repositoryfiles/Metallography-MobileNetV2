import glob
import os

import numpy as np
from keras.applications import MobileNetV2
from keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from PIL import Image, ImageOps
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.applications.mobilenet_v2 import preprocess_input

# 作業フォルダ
data_path = "c:/data/"  # 画像フォルダ
output_path = "c:/python/"  # h5ファイルなどの出力フォルダ

IMAGE_SIZE = 224
BATCH_SIZE = 16
NB_EPOCH = 100
VALID_SIZE = 0.2

# モデルの構築
def build_model():

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        alpha=0.35,  # alpha=0.35, 0.50, 0.75, 1.0, 1.3 or 1.4
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(nb_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # ベースモデル部分は再学習しない（転移学習）
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(learning_rate=5e-4, momentum=0.9),
        metrics=["accuracy"],
    )

    return model


labels = os.listdir(data_path)
nb_classes = len(labels)

labels.sort()
print(labels)

with open(output_path + "/labels.txt", "w") as o:
    for i, label in enumerate(labels):
        print(i, label, file=o)

X = []
y = []

for index, name in enumerate(labels):
    data_dir = data_path + "/" + name
    data_files = glob.glob(data_dir + "/*.*")
    for i, data_file in enumerate(data_files):

        image = Image.open(data_file)
        image = image.convert("RGB")

        im = ImageOps.fit(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS,  # Pillow 9.1.0で追加
        )

        data = np.asarray(im)
        X.append(data)
        y.append(index)

X = np.array(X)
y = np.array(y)

X = X.astype("float32")
print("Xmax and Xmin before preprocess_input")
print(X.max())
print(X.min())

x = preprocess_input(X)

print("Xmax and Xmin after preprocess_input")
print(X.max())
print(X.min())

data_num = len(X)
steps_per_epoch = data_num * (1 - VALID_SIZE) / BATCH_SIZE
print(steps_per_epoch)

valid_scores = []
models = []

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=VALID_SIZE, random_state=1
)

y_train = to_categorical(y_train, nb_classes)
y_valid = to_categorical(y_valid, nb_classes)

# インスタンスの呼び出し
model = build_model()

train_datagen = ImageDataGenerator(
    # rescale=1.0 / 255,  # 画素値の正規化（0～255を0～1に）
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    zoom_range=0.2,
    #rotation_range=20,
    #channel_shift_range=20,
    #brightness_range=[0.80, 1.0],
    # horizontal_flip=True,
    # vertical_flip=True,
)

train_datagen.fit(X_train)

# valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_datagen = ImageDataGenerator()
valid_datagen.fit(X_valid)

ResultFileName = output_path + "/" + "model"

# 学習経過の記録
csv_log = CSVLogger(ResultFileName + ".csv")

# モデルの学習が進まなくなったら学習終了
early_stopping = EarlyStopping(monitor="val_loss", patience=5, verbose=1)

# 学習率を減らす
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=1
)

modelCheckpoint = ModelCheckpoint(
    filepath=ResultFileName + ".h5",
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode="min",
    period=1,
)

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    epochs=NB_EPOCH,
    steps_per_epoch=steps_per_epoch,
    validation_steps=1,
    verbose=1,
    validation_data=valid_datagen.flow(X_valid, y_valid),
    callbacks=[csv_log, early_stopping, reduce_lr, modelCheckpoint],
)

# 結果のグラフ表示
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))
plt.plot(epochs, acc, "bo", label="Training")
plt.plot(epochs, val_acc, "b", label="Validation")
plt.title("Training and validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training")
plt.plot(epochs, val_loss, "b", label="Validation")
plt.title("Training and validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
