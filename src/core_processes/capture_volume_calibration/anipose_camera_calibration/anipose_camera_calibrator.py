import logging
from pathlib import Path
from typing import Union, Callable

import numpy as np

from src.config.home_dir import (
    get_freemocap_data_folder_path,
)
from src.core_processes.capture_volume_calibration.anipose_camera_calibration import (
    freemocap_anipose,
)
from aniposelib.boards import CharucoBoard as AniposeCharucoBoard

from src.core_processes.capture_volume_calibration.charuco_board_detection.dataclasses.charuco_board_definition import (
    CharucoBoardDefinition,
)

logger = logging.getLogger(__name__)


class AniposeCameraCalibrator:
    def __init__(
        self,
        charuco_board_object: CharucoBoardDefinition,
        charuco_square_size: Union[int, float],
        calibration_videos_folder_path: Union[str, Path],
        progress_callback: Callable[[str], None] = None,
        session_id: str = None,
    ):

        self._charuco_board_object = charuco_board_object
        self._progress_callback = progress_callback
        self._session_id = session_id

        if charuco_square_size == 1:
            logger.warning(
                "Charuco square size is not set, so units of 3d reconstructed data will be in units of `however_long_the_black_edge_of_the_charuco_square_was`. Please input `charuco_square_size` in millimeters (or your preferred unity of length)"
            )
        self._charuco_square_size = charuco_square_size
        self._calibration_videos_folder_path = calibration_videos_folder_path
        self._session_folder_path = Path(self._calibration_videos_folder_path).parent
        self._get_video_paths()
        self._initialize_anipose_objects()

    def _get_video_paths(
        self,
    ):
        self._list_of_video_paths = [
            this_video_path
            for this_video_path in Path(self._calibration_videos_folder_path).glob(
                "*.mp4".lower()
            )
        ]

    def _initialize_anipose_objects(self):
        list_of_camera_names = [
            this_video_path.stem for this_video_path in self._list_of_video_paths
        ]
        self._anipose_camera_group_object = freemocap_anipose.CameraGroup.from_names(
            list_of_camera_names
        )

        # add metadata
        self._anipose_camera_group_object.metadata[
            "charuco_square_size"
        ] = self._charuco_square_size
        self._anipose_camera_group_object.metadata["charuco_board_object"] = str(
            self._charuco_board_object
        )

        self._anipose_camera_group_object.metadata["charuco_board_object"] = str(
            self._session_id
        )

        self._anipose_charuco_board = AniposeCharucoBoard(
            self._charuco_board_object.number_of_squares_width,
            self._charuco_board_object.number_of_squares_height,
            square_length=self._charuco_square_size,  # mm
            marker_length=self._charuco_square_size * 0.8,
            marker_bits=4,
            dict_size=250,
        )

    def calibrate_camera_capture_volume(self, pin_camera_0_to_origin: bool = False):
        # anipose needs this to be a list of lists  (which is annoying but whatevs)
        video_paths_list_of_list_of_strings = [
            [str(this_path)] for this_path in self._list_of_video_paths
        ]

        (
            error,
            charuco_frame_data,
            charuco_frame_numbers,
        ) = self._anipose_camera_group_object.calibrate_videos(
            video_paths_list_of_list_of_strings, self._anipose_charuco_board
        )
        success_str = "Anipose Calibration Successful!"
        logger.info(success_str)
        self._progress_callback(success_str)
        if pin_camera_0_to_origin:
            # translate cameras so camera0 is on `0,0,0`
            self._anipose_camera_group_object = self.pin_camera_zero_to_origin(
                self._anipose_camera_group_object
            )
            # self._anipose_camera_group_object = self.rotate_cameras_so_camera_zero_aligns_with_XYZ(self._anipose_camera_group_object)

        # save calibration info to files
        calibration_toml_filename = f"camera_calibration_data.toml"
        camera_calibration_toml_path = (
            self._session_folder_path / calibration_toml_filename
        )

        self._anipose_camera_group_object.dump(camera_calibration_toml_path)
        logger.info(
            f"anipose camera calibration data saved to {str(camera_calibration_toml_path)}"
        )

        last_successful_calibration_path = Path(
            get_freemocap_data_folder_path(), "last_successful_calibration.toml"
        )
        self._anipose_camera_group_object.dump(last_successful_calibration_path)

        logger.info(
            f"anipose camera calibration data also saved to {str(last_successful_calibration_path)}"
        )

        return self._anipose_camera_group_object
        # # convert charuco data into a format that can be 3d reconstructed (effectively providing dummy data for the rest of the 3d reconstruction pipeline)
        # self.charuco_nCams_nFrames_nImgPts_XY = np.empty(
        #     [self.multi_cam.num_cams, self.multi_cam.num_frames, num_charuco_tracked_points, 2])
        # self.charuco_nCams_nFrames_nImgPts_XY[:] = np.nan
        #
        # for this_charuco_frame_data, this_charuco_frame_num in zip(charuco_frame_data, charuco_frame_numbers):
        #     for this_cam_num in range(self.multi_cam.num_cams):
        #         try:
        #             self.charuco_nCams_nFrames_nImgPts_XY[this_cam_num, this_charuco_frame_num, :, :] = np.squeeze(
        #                 this_charuco_frame_data[this_cam_num]["filled"])
        #         except:
        #             # print("failed frame:", frame)
        #             continue

    def pin_camera_zero_to_origin(self, _anipose_camera_group_object):
        original_translation_vectors = _anipose_camera_group_object.get_translations()
        camera_0_translation = original_translation_vectors[0, :]
        altered_translation_vectors = np.zeros(original_translation_vectors.shape)
        for this_camera_number in range(original_translation_vectors.shape[0]):
            altered_translation_vectors[this_camera_number, :] = (
                original_translation_vectors[this_camera_number, :]
                - camera_0_translation
            )

        _anipose_camera_group_object.set_translations(altered_translation_vectors)
        logger.info(f"original translation vectors:\n {original_translation_vectors}")
        logger.info(
            f"altered translation vectors:\n {_anipose_camera_group_object.get_translations()}"
        )
        return _anipose_camera_group_object

    def rotate_cameras_so_camera_zero_aligns_with_XYZ(
        self, _anipose_camera_group_object
    ):
        logger.warning("this function does not work, dont use it lol ")
        pass
        # original_rotations_euler = _anipose_camera_group_object.get_rotations()
        # original_translation_vectors = _anipose_camera_group_object.get_translations()
        # camera_rotation_matrix_list = [
        #     Rotation.from_euler('xyz', original_rotations_euler[this_cam_num, :]).as_matrix() for
        #     this_cam_num in range(original_rotations_euler.shape[0])]
        #
        # rotated_translation_vectors = [camera_rotation_matrix_list[0] @ this_tx for this_tx in
        #                                original_translation_vectors]
        # _anipose_camera_group_object.set_rotations(rotated_translation_vectors)
        # return _anipose_camera_group_object
