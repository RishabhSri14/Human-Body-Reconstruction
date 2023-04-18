import os
import cv2
import pycolmap

# Reading frames from the video file

# video_object = cv2.VideoCapture("mouse.mp4")
# frames = []

# while True:
#     success, frame = video_object.read()
#     if not success:
#         break
#     frames.append(frame)

# # Write these images in a images folder
# for i, frame in enumerate(frames):
#     cv2.imwrite(f"images/{i}.jpg", frame)

# video_object.release()


output_path = "./output"
image_dir = "./images"

# output_path.mkdir()
mvs_path = output_path + "/mvs"
database_path = output_path + "/database.db"

pycolmap.extract_features(database_path, image_dir)
pycolmap.match_exhaustive(database_path)
maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
maps[0].write(output_path)
# # dense reconstruction
# pycolmap.undistort_images(mvs_path, output_path, image_dir)
# pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
# pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)