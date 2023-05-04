#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

import os
import sys
import collections
import numpy as np
import struct
import argparse
import os
from pathlib import Path, PurePosixPath
import json
import math
import cv2
import shutil
import subprocess
import open3d as o3d
import open3d.core as o3c

abs_path = Path(os.path.realpath(__file__)).parents[1]
sys.path.append(str(abs_path))

import sqlite3
from colmap_database import COLMAPDatabase

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3D_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def point3D_to_pcd(points):
    xyzs = []
    rgbs = []
    for idx, pt in points.items():
        xyzs.append(pt.xyz)
        rgbs.append(pt.rgb)

    positions = o3c.Tensor(np.array(xyzs).astype(np.float32))
    colors = o3c.Tensor(np.array(rgbs).astype(np.float32) / 255.0)

    pcd = o3d.t.geometry.PointCloud(positions)
    pcd.point["colors"] = colors

    return pcd


def images_to_extrinsics(images):
    inds = sorted(images.keys())

    extrinsics = []
    for ind in inds:
        T = np.eye(4)
        R = qvec2rotmat(images[ind].qvec)
        t = images[ind].tvec.reshape((3, 1))
        T[:3, :3] = R
        T[:3, 3:] = t

        extrinsics.append(T)
    return np.array(extrinsics)


def images_to_dict_extrinsics(images):
    inds = sorted(images.keys())

    extrinsics = {}
    for ind in inds:
        T = np.eye(4)
        R = qvec2rotmat(images[ind].qvec)
        t = images[ind].tvec.reshape((3, 1))
        T[:3, :3] = R
        T[:3, 3:] = t

        extrinsics[images[ind].name] = T
    return extrinsics


def cameras_to_intrinsics(cameras):
    param_type = {
        "SIMPLE_PINHOLE": ["f", "cx", "cy"],
        "PINHOLE": ["fx", "fy", "cx", "cy"],
        "SIMPLE_RADIAL": ["f", "cx", "cy", "k"],
        "SIMPLE_RADIAL_FISHEYE": ["f", "cx", "cy", "k"],
        "RADIAL": ["f", "cx", "cy", "k1", "k2"],
        "RADIAL_FISHEYE": ["f", "cx", "cy", "k1", "k2"],
        "OPENCV": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"],
        "OPENCV_FISHEYE": ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"],
        "FULL_OPENCV": [
            "fx",
            "fy",
            "cx",
            "cy",
            "k1",
            "k2",
            "p1",
            "p2",
            "k3",
            "k4",
            "k5",
            "k6",
        ],
        "FOV": ["fx", "fy", "cx", "cy", "omega"],
        "THIN_PRISM_FISHEYE": [
            "fx",
            "fy",
            "cx",
            "cy",
            "k1",
            "k2",
            "p1",
            "p2",
            "k3",
            "k4",
            "sx1",
            "sy1",
        ],
    }

    inds = sorted(cameras.keys())

    intrinsics = []

    for ind in inds:
        cam = cameras[ind]
        params_dict = {
            key: value for key, value in zip(param_type[cam.model], cam.params)
        }
        if "f" in param_type[cam.model]:
            params_dict["fx"] = params_dict["f"]
            params_dict["fy"] = params_dict["f"]
        intrinsic = np.array(
            [
                [params_dict["fx"], 0, params_dict["cx"]],
                [0, params_dict["fy"], params_dict["cy"]],
                [0, 0, 1],
            ]
        )
        intrinsics.append(intrinsic)

    return np.array(intrinsics)


def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def sharpness(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def run_colmap(db, image_path, sparse_path, args):
    subprocess.run(
        "colmap feature_extractor "
        "--ImageReader.camera_model SIMPLE_PINHOLE "
        "--SiftExtraction.estimate_affine_shape=true "
        "--SiftExtraction.domain_size_pooling=true "
        "--ImageReader.single_camera 1 "
        f"--database_path {db} --image_path {image_path}",
        shell=True,
    )

    cmd = f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {db}"
    if args.colmap_matcher == "vocab_tree":
        assert args.vocab_tree != ""
        cmd += " --VocabTreeMatching.vocab_tree_path {args.vocab_tree}"
    subprocess.run(cmd, shell=True)

    try:
        shutil.rmtree(sparse_path)
    except:
        pass

    sparse_path.mkdir(exist_ok=True)
    subprocess.run(
        f"colmap {args.colmap_mapper} --database_path {db} --image_path {image_path} --output_path {sparse_path}",
        shell=True,
    )

    # Convert to txt as some other methods prefer this
    subprocess.run(
        f"colmap model_converter --input_path {sparse_path}/0 --output_path {sparse_path}/0 --output_type TXT",
        shell=True,
    )

    pts_fname = sparse_path / "0" / "points3D.bin"
    points = read_points3D_binary(pts_fname)
    pcd = point3D_to_pcd(points)
    o3d.io.write_point_cloud(str(sparse_path / "points.ply"), pcd.to_legacy())


def run_colmap_with_poses(
    db,
    image_path,
    pose_path,
    intrinsic_path,
    sparse_path,
    colmap_matcher="exhaustive",
):
    # Extract features
    subprocess.run(
        "colmap feature_extractor "
        "--ImageReader.camera_model SIMPLE_PINHOLE "
        "--SiftExtraction.estimate_affine_shape=true "
        "--SiftExtraction.domain_size_pooling=true "
        "--ImageReader.single_camera 1 "
        f"--database_path {db} "
        f"--image_path {image_path}",
        shell=True,
    )

    # Match features
    subprocess.run(
        f"colmap {colmap_matcher}_matcher "
        "--SiftMatching.guided_matching=true "
        f"--database_path {db}",
        shell=True,
    )

    database = COLMAPDatabase.connect(db)
    rows = database.execute("SELECT * FROM images")

    fname2pose_idx = {}
    im_fnames = sorted(image_path.glob("*"))
    for i, im_fname in enumerate(im_fnames):
        if i == 0:
            im = cv2.imread(str(im_fname))
            height, width, _ = im.shape
        name = im_fname.name

        fname2pose_idx[name] = i

    fname2colmap_idx = {}
    for row in rows:
        colmap_idx = row[0]
        fname = row[1]
        fname2colmap_idx[fname] = colmap_idx

    sparse_path.mkdir(exist_ok=True)

    input_sparse_path = sparse_path / "in"
    input_sparse_path.mkdir(exist_ok=True)

    output_sparse_path = sparse_path / "out"
    output_sparse_path.mkdir(exist_ok=True)

    # Write camera
    cam_fname = input_sparse_path / "cameras.txt"
    intrinsic = np.loadtxt(intrinsic_path)
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    with open(cam_fname, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    # Write poses
    ims_fname = input_sparse_path / "images.txt"
    poses = np.loadtxt(pose_path).reshape((-1, 4, 4))
    with open(ims_fname, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(
            f"# Number of images: {len(fname2colmap_idx)}, mean observations per image: 0\n"
        )

        for fname, colmap_idx in fname2colmap_idx.items():
            pose_idx = fname2pose_idx[fname]
            pose = poses[pose_idx]
            extrinsic = np.linalg.inv(pose)

            print(fname, colmap_idx, pose_idx, pose)
            qvec = rotmat2qvec(extrinsic[:3, :3])
            tvec = extrinsic[:3, 3]
            qw, qx, qy, qz = qvec
            tx, ty, tz = tvec
            f.write(f"{colmap_idx} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {fname}\n\n")

    # Create an empty file for points
    pts_fname = input_sparse_path / "points3D.txt"
    with open(pts_fname, "w") as f:
        pass

    subprocess.run(
        f"colmap point_triangulator --database_path {db} "
        f"--image_path {image_path} --input_path {input_sparse_path} "
        f"--output_path {output_sparse_path}",
        shell=True,
    )

    pts_fname = output_sparse_path / "points3D.bin"
    points = read_points3D_binary(pts_fname)
    pcd = point3D_to_pcd(points)
    o3d.io.write_point_cloud(str(sparse_path / "points.ply"), pcd.to_legacy())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scene_path", required=True, type=str)
    parser.add_argument("--image_path", default="image", type=str)
    parser.add_argument("--pose_path", default="poses.txt", type=str)
    parser.add_argument("--intrinsic_path", default="intrinsic.txt", type=str)

    # Required for running colmap
    parser.add_argument("--run_colmap", action="store_true")
    parser.add_argument("--run_colmap_with_poses", action="store_true")

    parser.add_argument(
        "--colmap_matcher",
        default="exhaustive",
        choices=["exhaustive", "sequential", "spatial", "transitive", "vocab_tree"],
        help="select which matcher colmap should use."
        "sequential for videos, exhaustive for adhoc images",
    )
    parser.add_argument("--vocab_tree", type=str, default="")

    parser.add_argument(
        "--colmap_mapper", default="mapper", choices=["mapper", "hierarchical_mapper"]
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    scene_path = Path(args.scene_path)
    image_path = scene_path / args.image_path

    if not image_path.is_dir():
        raise ValueError(f"{image_path} is not a directory")

    colmap_path = scene_path / "colmap"
    colmap_path.mkdir(exist_ok=True)

    db = colmap_path / "colmap.db"
    sparse_path = colmap_path / "sparse"

    if args.run_colmap:
        run_colmap(db, image_path, sparse_path, args)

    if args.run_colmap_with_poses:
        pose_path = scene_path / args.pose_path
        if not pose_path.is_file():
            raise ValueError(f"{pose_path} is not a file")

        intrinsic_path = scene_path / args.intrinsic_path
        if not intrinsic_path.is_file():
            raise ValueError(f"{intrinsic_path} is not a file")

        run_colmap_with_poses(
            db,
            image_path,
            pose_path,
            intrinsic_path,
            sparse_path,
            args.colmap_matcher,
        )
