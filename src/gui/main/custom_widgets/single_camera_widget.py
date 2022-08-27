import enum

import cv2
import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget


import logging

logger = logging.getLogger(__name__)


class SingleCameraViewWidget(QWidget):
    started = pyqtSignal()

    def __init__(self, camera_id: str):
        logger.info(f"Creating camera widget with Camera: {camera_id}")

        super().__init__()
        self._camera_id = camera_id
        self._image_view_label_widget = QLabel()

        layout = QHBoxLayout()
        layout.addWidget(self._image_view_label_widget)

        self.setLayout(layout)

    @property
    def camera_id(self):
        return self._camera_id

    def update_displayed_image(self, image: np.ndarray):
        image_to_display = cv2.flip(image, 1)
        image_to_display = cv2.cvtColor(image_to_display, cv2.COLOR_BGR2RGB)

        qimage = QImage(
            image_to_display,
            image_to_display.shape[1],
            image_to_display.shape[0],
            QImage.Format_RGB888,
        )

        qimage = qimage.scaledToHeight(300)
        qpixmap = QPixmap(qimage)
        self._image_view_label_widget.setPixmap(qpixmap)
