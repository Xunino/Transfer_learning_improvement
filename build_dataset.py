from iou import compute_iou
from bs4 import BeautifulSoup
from imutils import paths
import os
import cv2
import config


# Lặp qua chỉ dẫn Positive và Negative
for dirPath in (config.POSITIVE_PATH, config.NEGATIVE_PATH):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

# Liên kết tất cả đường dẫn của ảnh từ đường dẫn ảnh đầu vào
imagesPaths = list(paths.list_images(config.ORIG_IMAGES))
# print("\n".join(imagesPaths))

# Khởi tạo tổng số ảnh Positive và Negative để lưu
totalPositive = 0
totalNegative = 0

# Lặp qua tất cả đường dẫn của ảnh
for (i, imagePath) in enumerate(imagesPaths):
    # Hiện thị quá trình xử lý
    print("[INFO] proccessing image {}/{}...".format(i + 1, len(imagesPaths)))

    # Trích xuất tên file từ đường dẫn file và thu được đường dẫn từ annotation file
    filename = imagePath.split(os.path.sep)[-1]
    # print(filename)
    # Lấy từ đầu dến dấu "."
    filename = filename.split(".")[0]
    # filename = filename[:filename.rfind(".")]
    print(filename)
    annoPath = os.path.sep.join([config.ORIG_ANNO, "{}.xml".format(filename)])
    # print(annoPath)


    # Load the annotation file, build the soup, and khởi tạo danh sách của ground-truth bounding boxes
    contents = open(annoPath).read()
    soup = BeautifulSoup(contents, "html.parser")
    gtBoxes = []

    # Trích xuất kích thước của tấm ảnh
    w = int(soup.find("width").string)
    h = int(soup.find("height").string)
    # print("(w, h) - ({}, {})".format(w, h))

    # Lặp qua tất cả phần từ "object"
    for o in soup.find_all("object"):
        # Trích xuất nhãn và tọa độ của bounding boxes
        label = o.find("name").string
        # print(label)
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)
        # print("(xMin, xMin) - ({}, {})\n(xMax, yMax) - ({}, {})".format(xMin, yMin, xMax, yMax))

        # Cập nhật các box
        gtBoxes.append((xMin, yMin, xMax, yMax))
    # print(gtBoxes)


    # Load ảnh và perform Selective Search
    # Load ảnh form disk
    image = cv2.imread(imagePath)

    # Chạy Selective Search và khởi tạo list của box có khả năng
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()
    proposedRects = []

    # Lặp qua các hình chữ nhật đê được tạo bởi phương pháp Selective Search
    for (x, y, w, h) in rects:
        # convert our bounding boxes from (x, y, w, h) to (startX,
        # startY, startX, endY)
        proposedRects.append((x, y, x + w, y + h))


    # Khởi tạo biến đếm để đếm số ROIs Positive và Negative sau đó lưu trữ lại
    positiveRois = 0
    negativeRois = 0

    # Lặp qua số lượng tốt đa các đề xuất khu vực
    for proposedRect in proposedRects[:config.MAX_PROPOSALS]:
        # Lấy tọa độ của bounding box
        (propStartX, propStartY, propEndX, propEndY) = proposedRect

        # Lặp qua các box mốc
        for gtBox in gtBoxes:
            # Tính toán giao nhau giữa box mốc và bounding box dự đoán
            iou = compute_iou(gtBox, proposedRect)

            # Lấy tọa độ của mox mốc
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

            # Khởi tạo ROI và đường dẫn dầu ra
            roi = None
            outputPath = None

            # Kiểm tra xem, nếu giao IoU lớn hơn 70% và không vượt quá giới hạn cho phép (99%)
            if iou > 0.7 and positiveRois <= config.MAX_POSITIVE:
                # Trích xuất ROI và lấy được đường dẫn từ những trường hợp có thể
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalPositive)
                outputPath = os.path.sep.join([config.POSITIVE_PATH, filename])
                # print(outputPath)

                # Tăng biến đếm
                positiveRois +=1
                totalPositive +=1

            # Kiểm tra đúng sai
            fullOverlap = propStartX >= gtStartX
            fullOverlap = fullOverlap and propStartY >= gtStartY
            fullOverlap = fullOverlap and propEndX >= gtEndX
            fullOverlap = fullOverlap and propEndY >= gtEndY

            # Kiểm tra xem, nếu tại đó không giao nhau và IOU nhỏ hơn 5%
            if not fullOverlap and iou < 0.05 and negativeRois <= config.MAX_NEGATIVE:
                # Trích xuất ROI và sau đó lấy đường dẫn đầu ra từ trường hợp Negative
                roi = image[propStartY:propEndY, propStartX:propEndX]
                filename = "{}.png".format(totalNegative)
                outputPath = os.path.sep.join([config.NEGATIVE_PATH, filename])

                # Tăng biến đếm
                negativeRois +=1
                totalNegative +=1

            # Kiểm tra khả dụng của ROI và dường đẫn đầu ra
            # print(roi)
            if roi is not None and outputPath is not None:
                # Thay đổi kích cỡ ROI để kích thước đầu vào cùa mạng CNN, Nó sẽ được tinh chỉnh, sau đó lưu lại ROI từ disk
                roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(outputPath, roi)