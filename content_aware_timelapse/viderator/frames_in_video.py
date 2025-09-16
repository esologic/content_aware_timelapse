"""
Different functions to pull frames out of a video.
"""

import itertools
import logging
from fractions import Fraction
from pathlib import Path
from typing import Iterator, List, Optional, Protocol, cast

import cv2
import ffmpeg
import numpy as np

from content_aware_timelapse.viderator.video_common import VideoFrames, reduce_fps_take_every
from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    ImageSourceType,
    RGBInt8ImageType,
)

LOGGER = logging.getLogger(__name__)


class FramesInVideo(Protocol):
    """
    Protocol for different implementations of an RGB frame extractors.
    """

    def __call__(  # pylint: disable=too-many-positional-arguments
        self,
        video_path: Path,
        video_fps: Optional[float],
        reduce_fps_to: Optional[float],
        width_height: Optional[ImageResolution],
        starting_frame: Optional[int],
    ) -> VideoFrames:
        """
        :param video_path: The path to the video file on disk.
        :param video_fps: Can be used to override the actual FPS of the video.
        :param reduce_fps_to: Discards frames such that the frames that are returned are at this
        FPS.
        :param width_height: If given, the output frames will be resized to this resolution.
        :param starting_frame: Seek to this frame of the video open opening.
        :return: An NT containing metadata about the video, and an iterator that produces the
        frames. Frames are in RGB color order.
        :raises: ValueError if the video can't be opened, or the given `reduce_fps_to` is
        impossible.
        """


def frames_in_video_opencv(
    video_path: Path,
    video_fps: Optional[float] = None,
    reduce_fps_to: Optional[float] = None,
    width_height: Optional[ImageResolution] = None,
    starting_frame: Optional[int] = None,
) -> VideoFrames:
    """
    Reads frames from a file using openCV. CPU only.
    :param video_path: See docs in protocol.
    :param video_fps: See docs in protocol.
    :param reduce_fps_to: See docs in protocol.
    :param width_height: See docs in protocol.
    :param starting_frame: See docs in protocol.
    :return: See docs in protocol.
    """

    vid_capture = cv2.VideoCapture(str(video_path))

    file_fps = float(vid_capture.get(cv2.CAP_PROP_FPS))

    if video_fps:
        if video_fps != file_fps:
            LOGGER.warning(
                f"Override FPS of: {video_fps} fps "
                f"did not match the fps from the file of: {file_fps} fps."
            )
        fps = video_fps
    else:
        fps = file_fps

    frame_count = itertools.count()

    if starting_frame is not None:
        vid_capture.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
        frame_count = itertools.count(starting_frame)

    total_frame_count = int(vid_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    take_every = reduce_fps_take_every(original_fps=fps, new_fps=reduce_fps_to)

    original_width_height = ImageResolution(
        width=int(vid_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(vid_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    resize = (original_width_height != width_height) if width_height is not None else False

    if not vid_capture.isOpened():
        raise ValueError(f"Couldn't open video file: {video_path}")

    def frames() -> ImageSourceType:
        """
        Read frames off of the video capture until there none left or pulling a frame fails.
        :return: An iterator of frames.
        """
        while vid_capture.isOpened():

            current_frame_index = next(frame_count)

            LOGGER.debug(
                f"Got a frame # {current_frame_index} / {total_frame_count} "
                f"({(current_frame_index / total_frame_count) * 100:.2f}) from: {video_path.name}"
            )

            ret, frame = vid_capture.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output = cast(
                    RGBInt8ImageType,
                    (
                        image
                        if not resize
                        else cv2.resize(image, (width_height.width, width_height.height))
                    ),
                )
                yield output
            else:
                break

    return VideoFrames(
        original_fps=vid_capture.get(cv2.CAP_PROP_FPS),
        original_resolution=original_width_height,
        total_frame_count=total_frame_count,
        frames=itertools.islice(frames(), None, None, take_every),
    )


def frames_in_video_ffmpeg(  # pylint: disable=too-many-locals
    video_path: Path,
    video_fps: Optional[float] = None,
    reduce_fps_to: Optional[float] = None,
    width_height: Optional[ImageResolution] = None,
    starting_frame: Optional[int] = None,
) -> VideoFrames:
    """
    Deprecated! Reads frames from a file using ffmpeg directly. CPU only.
    :param video_path: See docs in protocol.
    :param video_fps: See docs in protocol.
    :param reduce_fps_to: See docs in protocol.
    :param width_height: See docs in protocol.
    :param starting_frame: See docs in protocol.
    :return: See docs in protocol.
    """
    probe = ffmpeg.probe(str(video_path))
    video_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")

    original_resolution = ImageResolution(
        width=video_stream["width"], height=video_stream["height"]
    )

    file_fps = float(Fraction(video_stream["r_frame_rate"]))

    # Compute total frame count safely
    if "nb_frames" in video_stream:
        total_frame_count = int(video_stream["nb_frames"])
    else:
        # Fallback: estimate using duration * FPS
        duration = float(probe["format"]["duration"])
        total_frame_count = int(duration * file_fps)

    if video_fps is not None:
        fps = video_fps
    else:
        fps = file_fps

    take_every = int(fps / reduce_fps_to) if reduce_fps_to and reduce_fps_to < fps else 1

    frame_size = width_height if width_height else original_resolution
    frame_bytes = frame_size.width * frame_size.height * 3  # RGB24

    # FFmpeg command setup
    input_kwargs = {}
    if starting_frame:
        input_kwargs["ss"] = starting_frame / fps  # Must be placed before "-i"

    output_kwargs = {
        "format": "rawvideo",
        "pix_fmt": "rgb24",
        "vsync": "0",  # Avoids unnecessary frame duplication
    }

    # Add scaling and FPS reduction to the filter chain
    filters: List[str] = []
    if width_height is not None:
        filters.append(f"scale={width_height.width}:{width_height.height}")

    if reduce_fps_to is not None:
        filters.append(f"fps={fps}")

    if filters:
        output_kwargs["vf"] = ",".join(filters)

    cmd = (
        ffmpeg.input(str(video_path), **input_kwargs)
        .output("pipe:", **output_kwargs)
        .global_args("-threads", "auto")  # Enables multi-threading
        .run_async(pipe_stdout=True, pipe_stderr=True, quiet=False)
    )

    def frames() -> Iterator[RGBInt8ImageType]:
        """
        Read frames off of the video capture until there none left or pulling a frame fails.
        :return: An iterator of frames.
        """

        try:
            while True:
                raw_frame = cmd.stdout.read(frame_bytes)
                if not raw_frame:
                    break
                frame = np.frombuffer(raw_frame, np.uint8).reshape(
                    (frame_size[1], frame_size[0], 3)
                )
                yield cast(RGBInt8ImageType, frame)
        finally:
            cmd.wait()

    return VideoFrames(
        original_fps=file_fps,
        original_resolution=original_resolution,
        total_frame_count=total_frame_count,
        frames=itertools.islice(frames(), None, None, take_every),
    )


# Starting place for hardware video decoder.
# pylint: disable=pointless-string-statement,trailing-whitespace
""" 
def frames_in_video_hardware(  # pylint: disable=too-many-locals
    video_path: Path,
    video_fps: Optional[float] = None,
    reduce_fps_to: Optional[float] = None,
    width_height: Optional[ImageResolution] = None,
    starting_frame: Optional[int] = None,
) -> VideoFrames:

    ffparams = {"-vcodec": "h264_cuvid", "-enforce_cv_patch": True}

    decoder = FFdecoder(
        str(video_path),
        frame_format="rgb24",
        verbose=True,
        **ffparams,
    ).formulate()

    metadata = json.loads(decoder.metadata)

    total_frame_count = metadata["approx_video_nframes"]
    file_fps = metadata["source_video_framerate"]

    if video_fps is not None:
        fps = video_fps
    else:
        fps = file_fps

    frame_count = itertools.count()

    if starting_frame is not None:
        frame_count = itertools.count(starting_frame)

    take_every = reduce_fps_take_every(original_fps=fps, new_fps=reduce_fps_to)

    original_width_height = ImageResolution(
        metadata["source_video_resolution"][0],
        metadata["source_video_resolution"][1],
    )

    resize = (original_width_height != width_height) if width_height is not None else False

    def frames() -> ImageSourceType:

        for frame in decoder.generateFrame():

            # check if frame is None
            if frame is None:
                break

            current_frame_index = next(frame_count)

            LOGGER.debug(
                f"Got a frame # {current_frame_index} / {total_frame_count} "
                f"({(current_frame_index / total_frame_count) * 100:.2f}) from: {video_path.name}"
            )

            # Convert YUV420p -> RGB
            rgb_frame = frame

            # Resize if needed
            if resize:
                rgb_frame = cv2.resize(rgb_frame, (width_height.width, width_height.height))

            yield cast(RGBInt8ImageType, rgb_frame)

    return VideoFrames(
        original_fps=fps,
        original_resolution=original_width_height,
        total_frame_count=total_frame_count,
        frames=itertools.islice(frames(), None, None, take_every),
    )
"""
