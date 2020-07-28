# Load model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import config
import pickle
import cv2
import imutils
import numpy as np
from nms import nms
# from imutils.object_detection import non_max_suppression

# load model
def load_(model_path, labels_path):
    model = load_model(model_path)
    # load labels
    lb = pickle.loads(open(labels_path, "rb").read())
    return model, lb
model, lb = load_(config.MODEL_PATH, config.LABEL_PATH)


# Sử dụng Selective Search
def selective_search(image):
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    return rects


image = cv2.imread("images\\raccoon_01.jpg")
image = imutils.resize(image, width=500)

rects = selective_search(image)

proposals = []
boxes = []
for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
    # print((x, y, w, h))
    # Roi in images
    roi = image[y:y + h, x:x + h]
    # print(roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)

    # Xử lý ROI
    roi = img_to_array(roi)
    roi = preprocess_input(roi)

    # Cập nhật
    proposals.append(roi)
    boxes.append((x, y, w + x, h + y))

# Chuyển đổi proposals và bounding boxes sang Numpy Array

proposals = np.array(proposals, dtype="float32")
boxes = np.array(boxes, dtype="int32")

# Classify of each ROIs khi sử dụng tinh chỉnh (fine-tune)
proba = model.predict(proposals)


# Tìm tất cả các chỉ mục của dự đoán phù hợp với raccoon class
# print(np.argmax(proba, axis=1))
# print(lb.classes_)
labels = lb.classes_[np.argmax(proba, axis=1)]
idxs = np.where(labels == "raccoons")[0]
# print(idxs)


# Sử dụng index để trích xuất tất cả các bounding boxes và mối quan hệ với class label
# có quan hệ với "raccoon" class
boxes = boxes[idxs]
# print(boxes)
proba = proba[idxs][:, 1]
# print("Probability before:", proba)


# Lọc những những dự đoán nhỏ
idxs = np.where(proba > config.MIN_PROB)
boxes = boxes[idxs]
proba = proba[idxs]
# print("Probability after: ", proba)


# Result without NMS
clone = image.copy()

for (box, prob) in zip(boxes, proba):
    # Draw box
    (startX, startY, endX, endY) = box

    cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startX + 10
    text = "Raccoon: {:.2f}-%".format(prob * 100)
    cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

cv2.imshow("Before NMS", clone)
# cv2.waitKey(0)



# Result with NMS

clone_1 = image.copy()
boxess = nms(boxes, proba, overlapThresh=0.3)

for i, box in enumerate(boxess):
    # Draw box
    (startX, startY, endX, endY) = box

    cv2.rectangle(clone_1, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startX + 10
    text = "Raccoon: {:.2f}-%".format(proba[i] * 100)
    cv2.putText(clone_1, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

cv2.imshow("After NMS", clone_1)
cv2.waitKey(0)