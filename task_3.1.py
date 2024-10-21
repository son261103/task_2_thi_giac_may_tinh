import cv2
import numpy as np
from matplotlib import pyplot as plt

# Danh sách các đường dẫn ảnh
image_paths = ['nguoi.jpg', 'images.jpg', 'images (1).jpg']

for image_path in image_paths:
    # Đọc ảnh
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Cannot load image {image_path}")
        continue

    # 1. Áp dụng toán tử Sobel
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # 2. Áp dụng Laplacian of Gaussian (LoG)
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
    log_edges = cv2.Laplacian(blurred_img, cv2.CV_64F)

    # Hiển thị kết quả
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(log_edges, cmap='gray')
    plt.title('Laplacian of Gaussian (LoG)')
    plt.axis('off')

    plt.show()
