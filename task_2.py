import cv2
import numpy as np
import matplotlib.pyplot as plt

# Hàm để hiển thị ảnh trên cùng một biểu đồ
def show_images(images, titles):
    plt.figure(figsize=(20, 10))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Hàm xử lý từng ảnh
def process_image(image):
    if image is None:
        print("Không thể tải ảnh. Vui lòng kiểm tra lại đường dẫn đến file.")
        return None

    # Bước 1: Ảnh âm tính
    negative_image = cv2.bitwise_not(image)

    # Bước 2: Tăng độ tương phản
    alpha = 1.5  # Hệ số độ tương phản (có thể điều chỉnh)
    beta = 0  # Giá trị điều chỉnh độ sáng
    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Bước 3: Biến đổi logarit
    norm_image = image / 255.0
    c = 255 / np.log(1 + np.max(norm_image))
    log_image = c * np.log(1 + norm_image)  # Áp dụng phép biến đổi logarit
    log_image = np.array(log_image * 255, dtype=np.uint8)  # Chuyển kết quả về kiểu uint8 để hiển thị ảnh

    # Bước 4: Cân bằng Histogram
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])  # Cân bằng Histogram trên kênh độ sáng
    hist_eq_image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)  # Chuyển lại ảnh về không gian màu BGR

    return [negative_image, contrast_image, log_image, hist_eq_image]

# Đọc và xử lý 2 ảnh
image1 = cv2.imread('images.jpg')
image2 = cv2.imread('images (1).jpg')
image3 = cv2.imread('nguoi.jpg')

# Xử lý từng ảnh
processed_images1 = process_image(image1)
processed_images2 = process_image(image2)
processed_images3 = process_image(image3)

# Hiển thị các ảnh kết quả nếu đã xử lý thành công
if processed_images1:
    show_images([image1] + processed_images1,
                ['Original Image 1', 'Negative Image 1', 'Contrast Image 1', 'Log Transform 1', 'Histogram Equalization 1'])

if processed_images2:
    show_images([image2] + processed_images2,
                ['Original Image 2', 'Negative Image 2', 'Contrast Image 2', 'Log Transform 2', 'Histogram Equalization 2'])

if processed_images3:
    show_images([image3] + processed_images3,
                ['Original Image 3', 'Negative Image 3', 'Contrast Image 3', 'Log Transform 3', 'Histogram Equalization 3'])