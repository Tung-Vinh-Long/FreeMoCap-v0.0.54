from typing import NamedTuple, Union

import numpy as np


class FramePayload(NamedTuple):
    success: bool = False
    image: np.ndarray = None
    timestamp_perf_counter_in_seconds: float = None
    timestamp_unix_time_seconds: float = None
    webcam_id: str = None
