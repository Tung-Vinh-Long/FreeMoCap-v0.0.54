import time
from typing import Dict

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QWidget, QLabel

from src.cameras.capture.dataclasses.frame_payload import FramePayload
from src.cameras.detection.models import FoundCamerasResponse
from src.config.webcam_config import WebcamConfig

from src.gui.main.main_window.middle_panel_viewers.camera_stream_grid_view import (
    CameraStreamGridView,
)
from src.gui.main.qt_utils.clear_layout import clear_layout
from src.gui.main.styled_widgets.page_title import PageTitle
from src.gui.main.workers.cam_detection_thread_worker import CameraDetectionThreadWorker

import logging

logger = logging.getLogger(__name__)


class CameraViewPanel(QWidget):
    def __init__(self):
        super().__init__()
        self._frame = QFrame()
        self._frame.setFrameShape(QFrame.Shape.StyledPanel)

        self._layout = QVBoxLayout()
        self._welcome_to_freemocap_title_widget = self._welcome_to_freemocap_title()
        self._layout.addWidget(self._welcome_to_freemocap_title_widget)
        self._frame.setLayout(self._layout)

        self._camera_stream_grid_view = CameraStreamGridView()

    @property
    def frame(self):
        return self._frame

    @property
    def camera_stream_grid_view(self):
        return self._camera_stream_grid_view

    def _welcome_to_freemocap_title(self):
        session_title = PageTitle(
            "Welcome  to  FreeMoCap! \n  \U00002728 \U0001F480 \U00002728 "
        )
        return session_title

    def update_camera_images(self, multi_frame_payload: Dict[str, FramePayload]):
        self._camera_stream_grid_view.update_camera_images(
            multi_frame_payload=multi_frame_payload
        )

    def show_camera_grid_view(self):
        self._welcome_to_freemocap_title_widget.close()
        self._layout.addWidget(self._camera_stream_grid_view)
