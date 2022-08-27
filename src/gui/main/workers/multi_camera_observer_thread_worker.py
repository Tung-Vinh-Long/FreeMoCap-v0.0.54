import threading
from typing import Dict

import cv2
from PyQt6.QtCore import pyqtSignal, QObject, QThread

from src.cameras.capture.dataclasses.frame_payload import FramePayload
from src.cameras.multi_camera_thread_runner import MultiCameraThreadRunner
from src.cameras.persistence.video_writer.video_recorder import VideoRecorder
from src.config.webcam_config import WebcamConfig

import logging

from src.gui.main.app import get_qt_app

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MultiCameraObserverThreadWorker(QThread):
    def __init__(
        self,
        camera_configs_dict: Dict[str, WebcamConfig],
        multi_cam_exit_event: threading.Event,
        new_multi_frame_available_signal: pyqtSignal,
        cameras_connected_signal: pyqtSignal,
        show_videos_in_cv2_windows_bool: bool = False,
    ):
        super().__init__()

        self._camera_config_dict = camera_configs_dict
        self._multi_cam_exit_event = multi_cam_exit_event

        self._new_multi_frame_available_signal = new_multi_frame_available_signal
        self.cameras_connected_signal = cameras_connected_signal
        self._show_videos_in_cv2_windows_bool = show_videos_in_cv2_windows_bool
        # if show_videos_in_cv2_window:
        #     self._new_multi_frame_available_signal.connect(
        #         self._show_videos_in_cv2_windows
        #     )

        self._dictionary_of_video_recorders = self._create_video_recorders()
        self._should_save_frames = False

        get_qt_app().aboutToQuit.connect(self.shut_down_multi_camera_runner)

    @property
    def dictionary_of_video_recorders(self):
        return self._dictionary_of_video_recorders

    def launch_multi_camera_threads(self, show_videos_in_cv2_windows: bool = False):
        self._multi_camera_thread_runner = MultiCameraThreadRunner(
            thread_exit_event=self._multi_cam_exit_event
        )

        self._multi_camera_thread_runner.create_and_launch_camera_threads(
            dictionary_of_webcam_configs=self._camera_config_dict,
            show_videos_in_cv2_windows=show_videos_in_cv2_windows,
        )
        self.cameras_connected_signal.emit()

    def run(self):
        while not self._multi_camera_thread_runner.thread_exit_event.is_set():
            if self._multi_camera_thread_runner.multi_frame_payload_queue.qsize() > 0:
                multi_frame_payload = (
                    self._multi_camera_thread_runner.multi_frame_payload_queue.get()
                )
                logger.debug(
                    f"Multi-frame payload queueue size: {self._multi_camera_thread_runner.multi_frame_payload_queue.qsize()}"
                )

                if self._show_videos_in_cv2_windows_bool:
                    self._show_videos_in_cv2_windows(multi_frame_payload)
                # self._new_multi_frame_available_signal.emit(multi_frame_payload)

    def shut_down_multi_camera_runner(self):
        self._multi_camera_thread_runner.exit()

    def start_saving_camera_frames(self):
        logger.info("starting to save frames for later recording")
        self._should_save_frames = True

    def stop_saving_camera_frames(self):
        logger.info("stopping saving frames in camera streams")
        self._should_save_frames = False

    def reset_video_recorders(self):
        for video_recorder in self._dictionary_of_video_recorders.values():
            video_recorder.close()
        self._dictionary_of_video_recorders = self._create_video_recorders()

    def _prepare_to_save_videos(self, multi_frame_payload: Dict[str, FramePayload]):
        if self._should_save_frames:
            for webcam_id, frame_payload in multi_frame_payload:
                self._dictionary_of_video_recorders[
                    webcam_id
                ].append_frame_payload_to_list(frame_payload)

    def _create_video_recorders(self):
        dictionary_of_video_recorders = {}
        for webcam_id in self._camera_config_dict.keys():
            dictionary_of_video_recorders[webcam_id] = VideoRecorder()

        return dictionary_of_video_recorders

    def _show_videos_in_cv2_windows(self, multi_frame_payload):
        for (
            webcam_id,
            frame_payload,
        ) in multi_frame_payload.items():
            cv2.imshow(
                f"(from `multi_camera_observer_thread_worker.py`)Camera {webcam_id} - PRESS `ESC` TO QUIT",
                frame_payload.image,
            )

        key = cv2.waitKey(1)
        if key == 27:  # esc key kills all streams, I think
            self._multi_camera_thread_runner.thread_exit_event.set()

        if self._multi_camera_thread_runner.thread_exit_event.is_set():
            cv2.destroyAllWindows()

    def quit(self):
        self._multi_camera_thread_runner.thread_exit_event.is_set()
        self.exit()
