def compute_iou(boxA, boxB):
    # Định nghĩa tạo độ (x, y) của hình chữ nhật/hình vuông
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Tính toán vùng giao nhau của hình chữ nhật/hình vuông
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Tính diện tích của 2 boxes
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính IoU (Intersection over Union) bằng cách  =  intersection_area / (boxAArea + BoxBArea - intersection_area)
    iou = intersection_area / float(boxAArea + boxBArea - intersection_area)

    return iou