import numpy as np

def nms(boxes, probs=None, overlapThresh=0.3):

    # Kiểm tra có box hay không
    if len(boxes) == 0:
        return []

    # Chuyển đổi giá trị của box từ integer sang float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Khởi tạo một list để chứa các box được lựa chọn
    pick = []

    # lấy tọa độ của box
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    idxs = y2

    # tính toán vùng của các bounding boxs và sự giao nhau của các vùng chỉ định để sắp xếp
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    if probs is not None:
        idxs = probs

    idxs = np.argsort(idxs)

    # Lặp những indexes vẫn còn trong danh sách
    while len(idxs) > 0:
        # Liên kết chỉ mục cuối cùng trong danh sách và thêm giá trị của chỉ mục tới danh sách chỉ mục được chọn
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Tìm tọa độ lớn nhất và tọa độ nhỏ nhất của bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Tính toán chiều dài và chiều rộng của bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)


        # Tính toán tỷ lệ giao nhau
        overlap = (w * h) / area[idxs[:last]]

        # Giữ lại những bounding box có tỷ lệ giao lớn nhất
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")
