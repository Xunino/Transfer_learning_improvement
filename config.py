import os

# Đường đẫn tới dataset
ORIG_PATH = "raccoons"
ORIG_IMAGES = os.path.sep.join([ORIG_PATH, "images"])
# print(ORIG_IMAGES)
ORIG_ANNO = os.path.sep.join([ORIG_PATH, "annotations"])
# print(ORIG_ANNO)


# Tạo file dataset để tạo bộ dữ liệu
BASE_PATH = "dataset"
POSITIVE_PATH = os.path.sep.join([BASE_PATH, "raccoons"])
# print(POSITIVE_PATH)
NEGATIVE_PATH = os.path.sep.join([BASE_PATH, "no_raccoons"])
# print(NEGATIVE_PATH)


# Định nghĩa Maximum Number of Region Proposal khi sử dụng Selective Search
# Và Performing inference
MAX_PROPOSALS = 1000
MAX_PROPOSALS_INFER = 200


# Định nghĩa số lượng vùng tối đa của Positive and Negative images được tạo từ những tấm ảnh
MAX_POSITIVE = 20
MAX_NEGATIVE = 10


# Định nghĩa kích thước ảnh đầu vào
INPUT_DIMS = (224, 224)


# Định nghĩa file lưu model và label dạng số nhị phân
MODEL_PATH = "raccoon_detector.h5"
LABEL_PATH = "label_encoder.pickle"


# Định nghĩa tỷ lệ thấp nhất để dự đoán vùng có khả năng
MIN_PROB = 0.99

# Learing_rate
LR = 0.001