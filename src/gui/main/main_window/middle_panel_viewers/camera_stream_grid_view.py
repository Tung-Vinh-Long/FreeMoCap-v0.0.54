from typing import List, Dict


from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel

from src.cameras.capture.dataclasses.frame_payload import FramePayload

from src.gui.main.custom_widgets.single_camera_widget import SingleCameraViewWidget

import logging

from src.gui.main.qt_utils.clear_layout import clear_layout

logger = logging.getLogger(__name__)


class CameraStreamGridView(QWidget):
    def __init__(self):
        super().__init__()
        self._central_layout = QVBoxLayout()
        self.setLayout(self._central_layout)

    def update_camera_images(self, multi_frame_payload: Dict[str, FramePayload]):
        for webcam_id, frame_payload in multi_frame_payload.items():
            self._camera_widgets[webcam_id].update_displayed_image(frame_payload.image)

    def create_camera_layouts(self, list_of_webcam_ids: List[str]):
        logger.info("creating camera layouts")
        # clear_layout(self._central_layout)
        self._create_camera_widgets(list_of_webcam_ids)

        self._camera_layouts = {}
        for webcam_id in list_of_webcam_ids:
            self._camera_layouts[webcam_id] = QVBoxLayout()
            self._camera_layouts[webcam_id].addWidget(
                QLabel(f"Camera {str(webcam_id)}")
            )
            self._camera_layouts[webcam_id].addWidget(self._camera_widgets[webcam_id])
            self._central_layout.addLayout(self._camera_layouts[webcam_id])

    def _create_camera_widgets(self, list_of_webcam_ids):
        self._camera_widgets = {}

        for webcam_id in list_of_webcam_ids:
            self._camera_widgets[webcam_id] = SingleCameraViewWidget(webcam_id)
