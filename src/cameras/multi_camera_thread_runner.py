import queue
import threading
from typing import Dict

import cv2

from src.cameras.capture.opencv_camera.opencv_camera import OpenCVCamera
from src.config.webcam_config import WebcamConfig

import logging


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


def gather_incoming_frames_and_stuff_them_into_a_queue(
    multi_frame_payload_queue: queue.Queue,
    dictionary_of_incoming_frame_queues: Dict[str, queue.Queue],
    thread_barrier: threading.Barrier,
    thread_exit_event: threading.Event,
):
    while not thread_exit_event.is_set():
        remaining = thread_barrier.wait()
        if remaining == 0:
            logger.debug(
                "`gather_incoming_frames_and_stuff_them_into_a_queue` thread was last to hit threading barrier"
            )
        multi_frame_payload_dictionary = {}

        for webcam_id, camera_queue in dictionary_of_incoming_frame_queues.items():
            multi_frame_payload_dictionary[webcam_id] = camera_queue.get()

        logger.debug("stuffing a multi_frame_payload into the queue")
        multi_frame_payload_queue.put(multi_frame_payload_dictionary)


def show_videos_in_cv2_windows(multi_frame_payload_queue, thread_exit_event):
    while not thread_exit_event.is_set():
        if multi_frame_payload_queue.qsize() > 0:
            multi_frame_payload = multi_frame_payload_queue.get()
            logger.debug(
                f"Multi-frame payload queueue size: {multi_frame_payload_queue.qsize()}"
            )

            for (
                webcam_id,
                frame_payload,
            ) in multi_frame_payload.items():
                cv2.imshow(
                    f"(from `multi_camera_thread_runner.py`) Camera {webcam_id} - PRESS `ESC` TO QUIT",
                    frame_payload.image,
                )

        key = cv2.waitKey(1)
        if key == 27:  # esc key kills all streams, I think
            thread_exit_event.set()

        if thread_exit_event.is_set():
            cv2.destroyAllWindows()


class MultiCameraThreadRunner:
    thread_exit_event = None

    @property
    def multi_frame_payload_queue(self):
        return self._multi_frame_payload_queue

    def create_and_launch_camera_threads(
        self,
        dictionary_of_webcam_configs=Dict[str, WebcamConfig],
        show_videos_in_cv2_windows: bool = False,
    ) -> queue.Queue:
        logger.info("creating camera threads")

        if self.thread_exit_event is not None:
            # if things are already running, shut them down before restarting
            self.thread_exit_event.set()

        self.thread_exit_event = threading.Event()

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
                self.thread_exit_event,
            )

        self._multi_frame_payload_queue = queue.Queue()
        self._create_and_start_multi_frame_gatherer_thread(
            self._multi_frame_payload_queue,
            self._dictionary_of_incoming_frame_queues,
            self._thread_barrier,
            self.thread_exit_event,
        )

        for camera_thread in self._dictionary_of_camera_threads.values():
            camera_thread.start()

        if show_videos_in_cv2_windows:
            self._launch_video_viewer_thread(
                self._multi_frame_payload_queue, self.thread_exit_event
            )

        return self._multi_frame_payload_queue

    def _launch_video_viewer_thread(
        self,
        multi_frame_payload_queue: queue.Queue,
        thread_exit_event: threading.Event,
    ):
        logger.info("Launching video viewing thread")
        self._video_viewer_thread = threading.Thread(
            target=show_videos_in_cv2_windows,
            args=(multi_frame_payload_queue, thread_exit_event),
            name="Video_viewing_thread",
        )
        self._video_viewer_thread.start()

    def _create_camera_thread(
        self,
        webcam_config: WebcamConfig,
        incoming_frames_queue: queue.Queue,
        thread_barrier: threading.Barrier,
        thread_exit_event: threading.Event,
    ):
        logger.info(f"Starting thread for Camera {webcam_config.webcam_id}")

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

    def _create_and_start_multi_frame_gatherer_thread(
        self,
        multi_frame_payload_queue: queue.Queue,
        dictionary_of_incoming_frame_queues: Dict[str, queue.Queue],
        thread_barrier: threading.Barrier,
        thread_exit_event: threading.Event,
    ):
        self._multi_frame_gatherer_thread = threading.Thread(
            target=gather_incoming_frames_and_stuff_them_into_a_queue,
            args=(
                multi_frame_payload_queue,
                dictionary_of_incoming_frame_queues,
                thread_barrier,
                thread_exit_event,
            ),
            name=f"Multi-frame-payload-grabber-thread",
        )
        self._multi_frame_gatherer_thread.start()

    def exit(self):
        self.thread_exit_event.set()


if __name__ == "__main__":
    from src.cameras.detection.cam_singleton import get_or_create_cams

    camera_thread_manager = MultiCameraThreadRunner()

    found_cameras_response = get_or_create_cams()
    available_cameras = found_cameras_response.cameras_found_list

    dictionary_of_webcam_configs = {}
    for webcam_id in available_cameras:
        dictionary_of_webcam_configs[str(webcam_id)] = WebcamConfig(webcam_id=webcam_id)

    camera_thread_manager.create_and_launch_camera_threads(
        dictionary_of_webcam_configs, show_videos_in_cv2_windows=True
    )
    #
    # while not camera_thread_manager.thread_exit_event.is_set():
    #     if camera_thread_manager.multi_frame_payload_queue.qsize() > 0:
    #         multi_frame_payload = camera_thread_manager.multi_frame_payload_queue.get()
    #         logger.info(
    #             f"Multi-frame payload queueue size: {camera_thread_manager.multi_frame_payload_queue.qsize()}"
    #         )
    #
    #         for (
    #             webcam_id,
    #             frame_payload,
    #         ) in multi_frame_payload.items():
    #             cv2.imshow(
    #                 f"Camera {webcam_id} - PRESS `ESC` TO QUIT", frame_payload.image
    #             )
    #
    #     key = cv2.waitKey(1)
    #     if key == 27:  # esc key kills all streams, I think
    #         camera_thread_manager.thread_exit_event.set()
    #
    #     if camera_thread_manager.thread_exit_event.is_set():
    #         cv2.destroyAllWindows()
