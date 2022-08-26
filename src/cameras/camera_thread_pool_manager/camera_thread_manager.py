import enum
import queue
import threading
from functools import partialmethod, partial
from typing import Dict

import cv2
from PyQt5.QtCore import QThreadPool

from src.cameras.capture.opencv_camera.opencv_camera import OpenCVCamera
from src.cameras.detection.cam_singleton import get_or_create_cams
from src.config.webcam_config import WebcamConfig

import logging

from src.gui.main.workers.cam_charuco_frame_thread_worker import (
    CamCharucoFrameThreadWorker,
)
from src.gui.main.workers.cam_frame_worker import CamFrameWorker

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def stuff_incoming_frames_into_a_queue(
    webcam_config: WebcamConfig,
    thread_queue: queue.Queue,
    thread_barrier: threading.Barrier,
    thread_exit_event: threading.Event,
):
    opencv_camera = OpenCVCamera(webcam_config)
    opencv_camera.connect()
    logger.info(
        "stuff_incoming_frames_into_a_queue starting for {}".format(opencv_camera.name)
    )

    while not thread_exit_event.is_set():
        frame_payload = opencv_camera.get_next_frame()
        logger.debug(
            f"{opencv_camera.name} grabbed a frame at timestamp: {frame_payload.timestamp_unix_time_seconds:.4f}"
        )

        if not frame_payload.success:
            logger.error(
                f"{opencv_camera.name} FAILED to grab a frame at timestamp: {frame_payload.timestamp_unix_time_seconds:.4f}"
            )

        thread_queue.put(frame_payload)
        remaining = thread_barrier.wait()
        if remaining == 0:
            logger.debug(f"{opencv_camera.name} was last to hit threading barrier")

    # after thread_exit_event
    opencv_camera.release()


def grab_incoming_frame_payloads(
    multi_frame_payload_queue: queue.Queue,
    dictionary_of_incoming_frame_queues: Dict[str, queue.Queue],
    thread_barrier: threading.Barrier,
    thread_exit_event: threading.Event,
):
    while not thread_exit_event.is_set():
        remaining = thread_barrier.wait()
        if remaining == 0:
            logger.debug(
                "`grab_incoming_frame_payloads` thread was last to hit threading barrier"
            )
        this_multi_frame_payload_dictionary = {}

        for webcam_id, camera_queue in dictionary_of_incoming_frame_queues.items():
            this_multi_frame_payload_dictionary[webcam_id] = camera_queue.get()

        logger.debug("stuffing a multi_frame_payload into the queue")
        multi_frame_payload_queue.put(this_multi_frame_payload_dictionary)


class CameraThreadManager:
    @property
    def thread_exit_event(self):
        return self._thread_exit_event

    @property
    def multi_frame_payload_queue(self):
        return self._multi_frame_payload_queue

    def create_camera_threads(
        self, dictionary_of_webcam_configs=Dict[str, WebcamConfig]
    ):
        logger.info("creating camera threads")
        self._thread_exit_event = threading.Event()

        # number of cameras plus one for the frame_grabbing_thread
        barrier_count = len(dictionary_of_webcam_configs) + 1

        self._thread_barrier = threading.Barrier(barrier_count)

        self._dictionary_of_camera_threads = {}
        self._dictionary_of_incoming_frame_queues = {}
        self._dictionary_of_open_cv_cameras = {}

        for webcam_id, webcam_config in dictionary_of_webcam_configs.items():
            self._dictionary_of_incoming_frame_queues[webcam_id] = queue.Queue()
            self._dictionary_of_camera_threads[webcam_id] = self._create_camera_thread(
                webcam_config,
                self._dictionary_of_incoming_frame_queues[webcam_id],
                self._thread_barrier,
                self._thread_exit_event,
            )

        self._multi_frame_payload_queue = queue.Queue()
        self._create_and_start_multi_frame_grabber_thread(
            self._multi_frame_payload_queue,
            self._dictionary_of_incoming_frame_queues,
            self._thread_barrier,
            self._thread_exit_event,
        )

        for camera_thread in self._dictionary_of_camera_threads.values():
            camera_thread.start()

    def _create_camera_thread(
        self,
        webcam_config: WebcamConfig,
        incoming_frames_queue: queue.Queue,
        thread_barrier: threading.Barrier,
        thread_exit_event: threading.Event,
    ):
        print(f"Starting thread for Camera {webcam_config.webcam_id}")

        camera_thread = threading.Thread(
            target=stuff_incoming_frames_into_a_queue,
            args=(
                webcam_config,
                incoming_frames_queue,
                thread_barrier,
                thread_exit_event,
            ),
            name=f"Camera_{webcam_config.webcam_id}-thread",
        )
        return camera_thread

    def _create_and_start_multi_frame_grabber_thread(
        self,
        multi_frame_payload_queue: queue.Queue,
        dictionary_of_incoming_frame_queues: Dict[str, queue.Queue],
        thread_barrier: threading.Barrier,
        thread_exit_event: threading.Event,
    ):
        self._multi_frame_grabber_thread = threading.Thread(
            target=grab_incoming_frame_payloads,
            args=(
                multi_frame_payload_queue,
                dictionary_of_incoming_frame_queues,
                thread_barrier,
                thread_exit_event,
            ),
            name=f"Multi-frame-payload-grabber-thread",
        )
        self._multi_frame_grabber_thread.start()


if __name__ == "__main__":
    camera_thread_manager = CameraThreadManager()

    found_cameras_response = get_or_create_cams()
    available_cameras = found_cameras_response.cameras_found_list

    dictionary_of_webcam_configs = {}
    for webcam_id in available_cameras:
        dictionary_of_webcam_configs[str(webcam_id)] = WebcamConfig(webcam_id=webcam_id)

    camera_thread_manager.create_camera_threads(dictionary_of_webcam_configs)

    while not camera_thread_manager.thread_exit_event.is_set():
        if camera_thread_manager.multi_frame_payload_queue.qsize() > 0:
            multi_frame_payload = camera_thread_manager.multi_frame_payload_queue.get()
            logger.info(
                f"Multi-frame payload queueue size: {camera_thread_manager.multi_frame_payload_queue.qsize()}"
            )

            for (
                webcam_id,
                frame_payload,
            ) in multi_frame_payload.items():
                cv2.imshow(
                    f"Camera {webcam_id} - PRESS `ESC` TO QUIT", frame_payload.image
                )

        key = cv2.waitKey(1)
        if key == 27:  # esc key kills all streams, I think
            camera_thread_manager.thread_exit_event.set()

        if camera_thread_manager.thread_exit_event.is_set():
            cv2.destroyAllWindows()
