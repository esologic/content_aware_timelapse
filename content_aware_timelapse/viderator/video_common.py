"""
Common functionality for dealing with video files.
"""

import logging
import math
import os
import pprint
import select
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, NamedTuple, Optional, Tuple, Union, cast

import cv2
import ffmpeg
import more_itertools
import numpy as np
from ffmpeg.nodes import FilterableStream

from content_aware_timelapse.viderator import image_common
from content_aware_timelapse.viderator.iterator_common import first_item_from_iterator
from content_aware_timelapse.viderator.viderator_types import (
    ImageResolution,
    ImageSourceType,
    RectangleRegion,
    RGBInt8ImageType,
)

_DEFAULT_WINDOW_NAME = "Frames"

LOGGER = logging.getLogger(__name__)

logging.getLogger("WriteGear").setLevel(logging.ERROR)


def _write_video_with_audio(video_path: Path, audio: FilterableStream, output_path: Path) -> None:
    """
    Adds an audio file to a video file. Copied from: https://stackoverflow.com/a/65547166
    :param video_path: Path to the video file.
    :param audio: The ffmpeg representation of the audio.
    :param output_path: The path to write the new file to.
    :return: None
    """
    ffmpeg.run(
        ffmpeg.output(
            ffmpeg.input(str(video_path)).video,
            audio,
            str(output_path),
            vcodec="copy",
            acodec="aac",
            audio_bitrate="192k",
            strict="experimental",
            shortest=None,  # <- important!
        ),
        quiet=True,
        overwrite_output=True,
    )


def _read_wav(audio_path: Path) -> FilterableStream:
    """
    Read an audio file as an ffmpeg stream.
    :param audio_path: Path to the audio file.
    :return: The ffmpeg stream of the audio.
    """
    return ffmpeg.input(str(audio_path)).audio


def add_wav_to_video(video_path: Path, audio_path: Path, output_path: Path) -> None:
    """
    Adds an audio file to a video file. Copied from: https://stackoverflow.com/a/65547166
    :param video_path: Path to the video file.
    :param audio_path: Path to the audio file.
    :param output_path: The path to write the new file to.
    :return: None
    """
    _write_video_with_audio(
        video_path=video_path, audio=_read_wav(audio_path), output_path=output_path
    )


def add_wavs_to_video(video_path: Path, audio_paths: List[Path], output_path: Path) -> None:
    """
    Adds an audio file to a video file. Copied from: https://stackoverflow.com/a/65547166
    :param video_path: Path to the video file.
    :param audio_paths: Paths to multiple audio files.
    :param output_path: The path to write the new file to.
    :return: None
    """
    _write_video_with_audio(
        video_path=video_path,
        audio=ffmpeg.concat(*[_read_wav(audio_path) for audio_path in audio_paths], v=0, a=1),
        output_path=output_path,
    )


class VideoOutputController(NamedTuple):
    """
    Interface for something to write videos with.
    Mimics `cv2.VideoWriter`.
    """

    # Adds a frame to the video.
    write: Callable[[RGBInt8ImageType], None]

    # Finalizes the video, you should be able to open and read the video after calling this.
    release: Callable[[], None]


def _create_video_writer_resolution(
    video_path: Path, video_fps: float, resolution: ImageResolution, high_quality: bool
) -> VideoOutputController:
    """
    Create a video writer of a given FPS and resolution.
    :param video_path: Resulting file path.
    :param video_fps: FPS of the video.
    :param resolution: Size of the resulting video.
    :param high_quality: If true, `ffmpeg` will be invoked directly this will create a much larger,
    much higher quality output.
    :return: The writer.
    """

    if high_quality:
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{resolution.width}x{resolution.height}",
            "-r",
            str(video_fps),
            "-i",
            "-",
            "-vf",
            f"yadif,scale={resolution.width}:{resolution.height}",
            "-vcodec",
            "libx264",
            "-profile:v",
            "high",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "2",
            "-crf",
            "14",
            "-preset",
            "veryslow",
            "-tune",
            "film",
            "-bf",
            "2",
            "-g",
            "250",
            "-movflags",
            "+faststart",
            str(video_path),
        ]

        ffmpeg_proc = subprocess.Popen(  # pylint: disable=consider-using-with
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,  # unbuffered — we manage backpressure manually
        )

        stdin_fd = ffmpeg_proc.stdin.fileno()

        def write_frame(image: RGBInt8ImageType) -> None:
            """
            Helper function to satisfy mypy.
            :param image: To write.
            :return: None
            """
            if image.shape[1] != resolution.width or image.shape[0] != resolution.height:
                raise ValueError("Incoming frame did not match output resolution!")
            if image.dtype != np.uint8:
                raise ValueError("Input image must be uint8.")

            # Wait until the pipe is ready for writing
            select.select([], [stdin_fd], [])
            ffmpeg_proc.stdin.write(image.tobytes())

        def release() -> None:
            """
            Closes the output stream.
            :return: None
            """
            if ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()

        return VideoOutputController(write=write_frame, release=release)

    else:
        opencv_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore
            video_fps,
            (resolution.width, resolution.height),
        )

        def frame_writer(image: RGBInt8ImageType) -> None:
            """
            Helper function to satisfy mypy.
            Adds a check to make sure the input image matches the expected
            resolution of the output. If this doesn't happen, openCV will
            silently create a video without frames.
            :param image: To write.
            :return: None
            """
            if image_common.image_resolution(image) != resolution:
                raise ValueError("Incoming frame did not match output resolution!")
            opencv_writer.write(image)

        return VideoOutputController(write=frame_writer, release=opencv_writer.release)


def create_video_writer(  # pylint: disable=too-many-positional-arguments
    video_path: Path,
    video_fps: float,
    video_height: int,
    num_squares_width: int,
    num_squares_height: int = 1,
    high_quality: bool = False,
) -> VideoOutputController:
    """
    Helper function to configure the VideoWriter which writes frames to a video file.
    :param video_path: Path to the file on disk.
    :param video_fps: Desired FPS of the video.
    :param video_height: Height of the video in pixels.
    :param num_squares_width: Since each section of the video is a `video_height` x `video_height`
    square this parameter sets the width for the video in pixels, with the number of these squares
    that will be written in each frame.
    :param num_squares_height: Like `num_squares_width`, but for height. Sets the height of
    the video in units of `video_height`.
    :param high_quality: Will be forwarded to library function.
    :return: The openCV `VideoWriter` object.
    """

    return _create_video_writer_resolution(
        video_path=video_path,
        video_fps=video_fps,
        resolution=ImageResolution(
            width=video_height * num_squares_width, height=video_height * num_squares_height
        ),
        high_quality=high_quality,
    )


class VideoFrames(NamedTuple):
    """
    Contains metadata about the video, and an iterator that produces the frames.
    """

    original_fps: float
    total_frame_count: int
    original_resolution: ImageResolution
    frames: ImageSourceType


def divide_no_remainder(numerator: Union[int, float], denominator: Union[int, float]) -> int:
    """
    Raise a Value Error if the division is not even.
    Return `numerator` / `denominator`.
    :param numerator: See top.
    :param denominator: See top.
    :return: `numerator` / `denominator`.
    """

    frac, repeats = math.modf(numerator / denominator)

    if frac != 0:
        raise ValueError(f"Cannot evenly divide {numerator} into {denominator}")

    return int(repeats)


def reduce_fps_take_every(original_fps: float, new_fps: float) -> Optional[int]:
    """
    When reducing a video from a high FPS, to a lower FPS, you should take every nth (output)
    frame of the input video.
    TODO: Wording very rough here.
    :param original_fps: Must be higher than `new_fps`.
    :param new_fps: Must go evenly into `original_fps`.
    :return: Take every n frames to get an evenly reduced output video.
    """

    if new_fps is not None:

        whole = divide_no_remainder(numerator=original_fps, denominator=new_fps)

        if whole != 1:
            return int(whole)

    return None


def write_source_to_disk_forward(
    source: ImageSourceType,
    video_path: Path,
    video_fps: float,
    audio_paths: Optional[List[Path]] = None,
    high_quality: bool = False,
) -> ImageSourceType:
    """
    Consume an image source, write it out to disk.
    :param source: To write to disk.
    :param video_path: Output video path.
    :param video_fps: Frames/Second of the output video.
    :param audio_paths: If given, the audio files will be written to the output video.
    :param high_quality: Flag will be forwarded to library function.
    :return: None
    """

    def setup_iteration(output_path: Path) -> ImageSourceType:
        """
        Helper function to set up the output and forwarding operation.
        :param output_path: Intermediate video path.
        :return: The frames to yield.
        """

        first_frame, frame_source = first_item_from_iterator(source)

        writer = _create_video_writer_resolution(
            video_path=output_path,
            video_fps=video_fps,
            resolution=image_common.image_resolution(first_frame),
            high_quality=high_quality,
        )

        def write_frame(frame: RGBInt8ImageType) -> None:
            """
            Write the given frame to the file.
            :param frame: To write.
            :return: None
            """
            rgb_frame = cast(
                RGBInt8ImageType, cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            )
            writer.write(rgb_frame)

        try:
            for index, image in enumerate(frame_source):
                LOGGER.debug(f"Writing frame #{index} to file: {video_path}")
                write_frame(image)
                yield image
        finally:
            writer.release()

    try:
        if audio_paths is None or not audio_paths:
            yield from setup_iteration(video_path)
        else:
            with video_safe_temp_path(suffix=video_path.suffix) as temp_video_path:
                yield from setup_iteration(temp_video_path)
                LOGGER.info(f"Finalizing {video_path}")
                add_wavs_to_video(
                    video_path=temp_video_path,
                    audio_paths=audio_paths,
                    output_path=video_path,
                )
    except Exception as e:
        LOGGER.exception(f"Ran into an exception writing out a video: {pprint.pformat(e)}")
        raise e


def write_source_to_disk_consume(
    source: ImageSourceType,
    video_path: Path,
    video_fps: float,
    audio_paths: Optional[List[Path]] = None,
    high_quality: bool = False,
) -> None:
    """
    Consume an image source, write it out to disk.
    :param source: To write to disk.
    :param video_path: Output video path.
    :param video_fps: FPS of the output video.
    :param audio_paths: If given, the audio file at this path will be written to the output video.
    :param high_quality: Flag will be forwarded to library function.
    :return: None
    """

    more_itertools.consume(
        write_source_to_disk_forward(
            source=source,
            video_path=video_path,
            video_fps=video_fps,
            audio_paths=audio_paths,
            high_quality=high_quality,
        )
    )


def resize_source(source: ImageSourceType, resolution: ImageResolution) -> ImageSourceType:
    """
    For each item in the input source, scale it to the given resolution.
    :param source: Contains Images.
    :param resolution: Desired resolution.
    :return: A new source of scaled images.
    """

    yield from (
        (
            image_common.resize_image(image, resolution, delete=True)
            if image_common.image_resolution(image) != resolution
            else image
        )
        for image in source
    )


def scale_square_source_duplicate(
    source: ImageSourceType, output_side_length: int, frame_multiplier: int = 1
) -> ImageSourceType:
    """
    Scale the resolution and number of frames in a given source.
    :param source: To scale.
    :param output_side_length: Square frames will be resized to this side length.
    :param frame_multiplier: Every frame will be duplicated this many times.
    :return: Scaled source.
    """

    resized = resize_source(source, ImageResolution(output_side_length, output_side_length))

    return (
        cast(
            ImageSourceType,
            more_itertools.repeat_each(
                resized,
                frame_multiplier,
            ),
        )
        if frame_multiplier != 1
        else resized
    )


def display_frame_forward_opencv(
    source: ImageSourceType,
    window_name: str = _DEFAULT_WINDOW_NAME,
    display_resolution: Optional[ImageResolution] = None,
    full_screen: bool = False,
) -> ImageSourceType:
    """
    Displays the images in `source`, and forwards the image so they can be consumed again.
    Uses an openCV `imshow` to reveal the image.
    :param source: To display.
    :param window_name: Name of the window.
    :param display_resolution: Change the input frames to this resolution before displaying. Doesn't
    modify the output frames.
    :param full_screen: If True, images will be displayed fullscreen, windowed if otherwise.
    :return: Forwarded iterator, `source`.
    """

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_GUI_NORMAL if full_screen else cv2.WINDOW_AUTOSIZE,
    )

    if full_screen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def display_frame(frame: RGBInt8ImageType) -> RGBInt8ImageType:
        """
        Helper function, actually show the image.
        :param frame: To display.
        :return: `frame`.
        """
        cv2.imshow(
            window_name,
            cv2.cvtColor(
                (
                    image_common.resize_image(frame, display_resolution)
                    if display_resolution is not None
                    else frame
                ),
                cv2.COLOR_RGB2BGR,
            ),
        )
        cv2.waitKey(1)
        return frame

    # yield from is used, so we can run things after the input has been exhausted to cleanup.
    yield from map(display_frame, source)

    cv2.destroyWindow(window_name)


def crop_source(source: ImageSourceType, region: RectangleRegion) -> ImageSourceType:
    """
    Crop all frames in a source to a given region.
    :param source: To crop.
    :param region: Region to crop to, this is an absolute region in the input frame, not a general
    resolution.
    :return: Cropped frames.
    """

    yield from (
        (image_common.crop_image(image=image, region=region, delete=True)) for image in source
    )


@contextmanager
def video_safe_temp_path(suffix: str = ".mp4") -> Iterator[Path]:
    """
    Context manager that yields a Path to a temporary file.
    The file is created and closed immediately, then unlinked
    automatically on context exit.

    :param suffix: Optional file suffix (e.g. '.mp4')
    """
    fd, temp_name = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # close the low-level fd, we only care about the path
    video_path = Path(temp_name)
    try:
        yield video_path
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:  # pylint: disable=broad-except
            pass


def concat_videos_for_youtube(video_paths: Tuple[Path, ...], output_path: Path) -> None:
    """
    Concatenates videos from disk and writes them to an output location, re-encoding them with
    the optimal settings for uploading to YouTube.

    :param video_paths: Videos to concatenate as a tuple.
    :param output_path: Destination path.
    :return: None
    :raises ValueError: If any of the videos are missing or have mismatched resolution/fps, or
    are duplicates!
    """

    if not video_paths:
        raise ValueError("No input files provided.")

    # Resolve input paths and check existence
    resolved_paths = [p.absolute() for p in video_paths]

    for p in resolved_paths:
        if not p.exists():
            raise ValueError(f"Input file {p} does not exist.")

    # Check for duplicates
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("Duplicate input files are not allowed.")

    # Check that all videos have the same resolution and fps
    widths, heights, fps_set = set(), set(), set()

    for f in resolved_paths:
        stream = ffmpeg.probe(str(f), select_streams="v")["streams"][0]
        widths.add(int(stream["width"]))
        heights.add(int(stream["height"]))
        a, b = stream["r_frame_rate"].split("/")
        fps_set.add(float(int(a) / int(b)))

    if len(widths) > 1 or len(heights) > 1 or len(fps_set) > 1:
        raise ValueError(
            "Input files need to have the same resolution and fps. "
            f"Widths: {widths}, Heights: {heights}, FPS: {fps_set}"
        )

    # Create a temporary file listing the videos for ffmpeg concat demuxer
    with tempfile.NamedTemporaryFile("w", delete=True, suffix=".txt") as list_file:

        for f in resolved_paths:
            list_file.write(f"file '{f}'\n")

        list_file.flush()

        # Re-encode using high-quality settings for YouTube
        ffmpeg.input(str(Path(list_file.name)), format="concat", safe=0).output(
            str(output_path),
            vcodec="libx264",
            crf=18,  # visually lossless
            preset="slow",  # slower preset for better compression
            pix_fmt="yuv420p",  # compatible pixel format
            acodec="aac",
            audio_bitrate="192k",
            movflags="+faststart",  # allows progressive playback
        ).run(overwrite_output=True, quiet=True)
