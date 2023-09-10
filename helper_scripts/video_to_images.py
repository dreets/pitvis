# global imports
import cv2
import numpy as np
import os

# strong typing
from pathlib import Path
from typing import List


def main():
    """required variables are {pt_videos} and {pt_images}"""
    # pt_videos: Path =
    # pt_images: Path =
    # convert_videos_to_images(pt_videos=pt_videos, pt_images=pt_images)
    pass


def convert_videos_to_images(pt_videos: Path, pt_images: Path):
    """convert all videos from {pt_videos} to images saved to {pt_images}"""
    create_directory(pt=pt_images)

    ls_videos: List[str] = os.listdir(pt_videos)
    ls_videos.sort()

    for str_video in ls_videos:
        pt_video: Path = pt_videos.joinpath(str_video)
        pt_image: Path = pt_images.joinpath(str_video.split(".")[0])

        create_directory(pt=pt_image)
        convert_video_to_image(pt_video=pt_video, pt_image=pt_image)


def convert_video_to_image(pt_video: Path, pt_image: Path):
    """convert a single video from {pt_video} to images saved to {pt_image}"""
    video_capture = cv2.VideoCapture(str(pt_video))
    int_frames_per_second: int = np.ceil(video_capture.get(cv2.CAP_PROP_FPS))  # ceiling function to ensure integer

    int_frame: int = 0
    while video_capture.isOpened():
        bool_success, np_frame_matrix = video_capture.read()
        if bool_success:
            if int_frame % int_frames_per_second == 0:
                pt_image_frame: Path = pt_image.joinpath(f"{int(int_frame / int_frames_per_second):05}.png")
                cv2.imwrite(str(pt_image_frame), np_frame_matrix)
        else:
            break
        int_frame += 1

    video_capture.release()

    print(f"{pt_video} successfully converted to {int_frame} images.")


def create_directory(pt: Path):
    """create a directory for a given {path} if it does not already exist"""
    if not os.path.exists(pt):
        os.mkdir(pt)


if __name__ == "__main__":
    main()
