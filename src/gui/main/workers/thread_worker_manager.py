import threading
from pathlib import Path
from typing import Union, Dict, Callable

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from src.cameras.capture.dataclasses.frame_payload import FramePayload
from src.cameras.detection.models import FoundCamerasResponse
from src.cameras.persistence.video_writer.video_recorder import VideoRecorder

from src.gui.main.workers.anipose_calibration_thread_worker import (
    AniposeCalibrationThreadWorker,
)
from src.gui.main.workers.cam_detection_thread_worker import CameraDetectionThreadWorker

import logging

from src.gui.main.workers.multi_camera_observer_thread_worker import (
    MultiCameraObserverThreadWorker,
)
from src.gui.main.workers.mediapipe_2d_detection_thread_worker import (
    Mediapipe2dDetectionThreadWorker,
)
from src.gui.main.workers.save_to_video_thread_worker import SaveToVideoThreadWorker
from src.gui.main.workers.triangulate_3d_data_thread_worker import (
    Triangulate3dDataThreadWorker,
)

logger = logging.getLogger(__name__)


class ThreadWorkerManager(QWidget):
    """This guy's job is to hold on to the parts of threads that need to be kept alive while they are running"""

    camera_detection_finished = pyqtSignal(FoundCamerasResponse)
    cameras_connected_signal = pyqtSignal()
    new_multi_frame_available_signal = pyqtSignal(dict)
    # ready_to_save_videos_signal: pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        # self._should_save_frames = False

    # def start_saving_camera_frames(self):
    #     self._multi_camera_manager.start_saving_camera_frames()
    #
    # def stop_saving_camera_frames(self):
    #     self._multi_camera_manager.stop_saving_camera_frames()
    @property
    def multi_camera_observer_thread_worker(self):
        return self._multi_camera_observer_thread_worker

    def launch_detect_cameras_worker(self):
        logger.info("Launch camera detection worker")
        self._camera_detection_thread_worker = CameraDetectionThreadWorker()
        self._camera_detection_thread_worker.finished.connect(
            self.camera_detection_finished.emit
        )
        self._camera_detection_thread_worker.start()

    def launch_multi_camera_observer_thread_worker(
        self, webcam_config_dict, multi_cam_exit_event: threading.Event
    ):

        logger.info("Launch multi-cam-observer-thread-worker")

        self._multi_camera_observer_thread_worker = MultiCameraObserverThreadWorker(
            camera_configs_dict=webcam_config_dict,
            multi_cam_exit_event=multi_cam_exit_event,
            new_multi_frame_available_signal=self.new_multi_frame_available_signal,
            cameras_connected_signal=self.cameras_connected_signal,
            show_videos_in_cv2_windows_bool=True,
        )

        self._multi_camera_observer_thread_worker.launch_multi_camera_threads()
        self._multi_camera_observer_thread_worker.finished.connect(
            self._multi_camera_observer_thread_worker.shut_down_multi_camera_runner
        )
        self._multi_camera_observer_thread_worker.start()

    # def quit_multi_camera_thread(self):
    #     self._multi_camera_observer_thread_worker.quit()
    #     logger.info("Starting shut down for the multicamera oberver thread...")
    #     self._multi_camera_observer_thread_worker.wait()
    #     logger.info("Multicamera oberver thread shut down!")

    def launch_save_videos_thread_worker(
        self,
        folder_to_save_videos: [Union[str, Path]],
        dictionary_of_video_recorders: Dict[str, VideoRecorder],
        reset_video_recorders_function: Callable,
    ):
        logger.info("Launching save videos thread worker...")

        self._save_to_video_thread_worker = SaveToVideoThreadWorker(
            dictionary_of_video_recorders=dictionary_of_video_recorders,
            folder_to_save_videos=folder_to_save_videos,
        )
        self._save_to_video_thread_worker.finished.connect(
            reset_video_recorders_function
        )
        self._save_to_video_thread_worker.start()

    def launch_anipose_calibration_thread_worker(
        self,
        calibration_videos_folder_path: Union[str, Path],
        charuco_square_size_mm: float,
    ):
        self._anipose_calibration_worker = AniposeCalibrationThreadWorker(
            calibration_videos_folder_path=calibration_videos_folder_path,
            charuco_square_size_mm=charuco_square_size_mm,
        )
        self._anipose_calibration_worker.start()
        self._anipose_calibration_worker.in_progress.connect(print)

    def launch_detect_2d_skeletons_thread_worker(
        self,
        synchronized_videos_folder_path: Union[str, Path],
        output_data_folder_path: Union[str, Path],
    ):
        logger.info("Launching mediapipe 2d skeleton thread worker...")

        self._mediapipe_2d_detection_thread_worker = Mediapipe2dDetectionThreadWorker(
            path_to_folder_of_videos_to_process=synchronized_videos_folder_path,
            output_data_folder_path=output_data_folder_path,
        )

        self._mediapipe_2d_detection_thread_worker.start()

    def launch_triangulate_3d_data_thread_worker(
        self,
        anipose_calibration_object,
        mediapipe_2d_data: np.ndarray,
        output_data_folder_path: Union[str, Path],
    ):
        logger.info("Launching Triangulate 3d data thread worker...")

        self._triangulate_3d_data_thread_worker = Triangulate3dDataThreadWorker(
            anipose_calibration_object=anipose_calibration_object,
            mediapipe_2d_data=mediapipe_2d_data,
            output_data_folder_path=output_data_folder_path,
        )

        self._triangulate_3d_data_thread_worker.start()
