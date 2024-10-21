import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt6.QtGui import QPixmap, QImage
from matplotlib import pyplot as plt


class EdgeDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Edge Detection with Sobel and LoG')


        layout = QVBoxLayout()


        self.image_label = QLabel("No image loaded.")
        layout.addWidget(self.image_label)


        self.btn_load = QPushButton('Load Image')
        self.btn_load.clicked.connect(self.load_image)
        layout.addWidget(self.btn_load)


        self.btn_process = QPushButton('Detect Edges')
        self.btn_process.clicked.connect(self.detect_edges)
        layout.addWidget(self.btn_process)


        self.setLayout(layout)
        self.img = None  # Để lưu ảnh sau khi tải lên

    def load_image(self):

        image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image', '', 'Image files (*.jpg *.jpeg *.png)')
        if image_path:
            self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh dạng grayscale
            self.show_image(self.img)  # Hiển thị ảnh đã tải lên

    def show_image(self, img):

        height, width = img.shape
        q_img = QImage(img.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

    def detect_edges(self):
        if self.img is None:
            return  # Không có ảnh để xử lý

        # 1. Áp dụng toán tử Sobel
        sobel_x = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # 2. Áp dụng Laplacian of Gaussian (LoG)
        blurred_img = cv2.GaussianBlur(self.img, (3, 3), 0)
        log_edges = cv2.Laplacian(blurred_img, cv2.CV_64F)

        # Hiển thị kết quả của Sobel và LoG
        self.show_results(sobel_combined, log_edges)

    def show_results(self, sobel, log):
        # Sử dụng matplotlib để hiển thị ảnh kết quả
        plt.subplot(1, 2, 1)
        plt.imshow(sobel, cmap='gray')
        plt.title('Sobel Edge Detection')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(log, cmap='gray')
        plt.title('Laplacian of Gaussian (LoG)')
        plt.axis('off')

        plt.show()


# Khởi tạo ứng dụng Qt
app = QApplication(sys.argv)
window = EdgeDetectionApp()
window.show()
sys.exit(app.exec())
