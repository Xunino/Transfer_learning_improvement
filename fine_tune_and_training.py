# Using Mobilenet_V2

# -----------------------------------------------------------------------

# Liên kết danh sách ảnh trong bộ dữ liệu, sau đó khởi tạo
# danh sách dữ liệu và nhãn
from imutils import paths
import config
imagePaths = list(paths.list_images(config.BASE_PATH))
# print("\n".join(imagePaths))

# Khởi tạo biến để lưu label và data
labels = []
dataset = []

import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# Load từng ảnh trong dữ liệu (data)
print("[INFO] Loading image form disk....")
for imagePath in imagePaths:
    # Lấy nhãn
    label = imagePath.split(os.path.sep)[1]
    # print(label)

    # Tải ảnh với đầu vào là 224x224 và xử lý ảnh sang dạng mảng
    image = load_img(imagePath, target_size=config.INPUT_DIMS)
    image = img_to_array(image)
    image = preprocess_input(image)

    # Cập nhật dữ liệu và nhãn
    labels.append(label)
    dataset.append(image)

# -----------------------------------------------------------------------

# Chuyển đổi dữ liệu và nhãn sang kiểu dữ liệu Numpy Array
print("[INFO] Converting data format...")
import numpy as np
dataset = np.array(dataset, dtype="float32")
labels = np.array(labels)
# print(labels)

# -----------------------------------------------------------------------

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
# Sử dụng one-hot coding để tăng hiệu xuất phân loại cho class
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
# print(labels)

# -----------------------------------------------------------------------

from sklearn.model_selection import train_test_split
# Chia tập train và tập validation
(Xtrain, Xtest, Ytrain, Ytest) = train_test_split(dataset, labels, test_size=0.2,
                                                  stratify=labels, random_state=42)
# print(Xtrain.shape)
# print(Ytrain)
# print(Xtest.shape)
# print(Ytest)

# -----------------------------------------------------------------------

# Tạo bộ dữ liệu lớn hơn bằng cách sử dụng ImageGenerater
from tensorflow.keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest"
                         )

# -----------------------------------------------------------------------

# Using transfer learning - Mobilenet_V2 model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model


baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# print(baseModel.summary())


# Tạo phần ourput model để thay thế cho phần output của model gốc
head_Model = baseModel.output
head_Model = AveragePooling2D(pool_size=(7, 7))(head_Model)
head_Model = Flatten()(head_Model)
head_Model = Dense(64, activation="relu")(head_Model)
head_Model = Dropout(0.2)(head_Model)
head_Model = Dense(2, activation="softmax")(head_Model)

# Đặt vào phần FC model và phần cuối của model gốc
model = Model(inputs=baseModel.input, outputs=head_Model)
# print(model.summary())

# Đóng băng các layers để chúng ko bị update trong quá trình training
for layer in baseModel.layers:
    layer.trainable = False

# -----------------------------------------------------------------------

# Biên soạn model
from tensorflow.keras.optimizers import Adam
print("[INFO] Compiling model....")
model.compile(loss="categorical_crossentropy", optimizer=Adam(config.LR), metrics=["acc"])

# Train model
print("[INFO] Training head model....")
BS = 24
epochs = 5
Hist = model.fit(aug.flow(Xtrain, Ytrain, batch_size=BS),
                      steps_per_epoch=len(Xtrain) // BS,
                      validation_data=(Xtest, Ytest),
                      validation_steps=len(Xtest) // BS,
                      epochs=epochs)

# -----------------------------------------------------------------------

# Dự đoán trên tập test
print("[INFO] Evaluating network....")
predIdxs = model.predict(Xtest, batch_size=BS)

# Tìm index của mỗi ảnh trong tập test và dự đoán class cho ảnh đó
predIdxs = np.argmax(predIdxs, axis=1)
# print(predIdxs)

# Show classification report
from sklearn.metrics import classification_report
print(classification_report(Ytest.argmax(axis=1), predIdxs, target_names=lb.classes_))

# -----------------------------------------------------------------------

# Save the model
import pickle

print("[INFO] Saving the model....")
model.save(config.MODEL_PATH)

# Save label sử dụng label encoder

f = open(config.LABEL_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

# -----------------------------------------------------------------------

# plot the training loss and accuracy
import matplotlib.pyplot as plt
N = epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), Hist.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), Hist.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), Hist.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), Hist.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")






