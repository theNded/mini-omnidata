import argparse
import numpy as np
import cv2
from pathlib import Path

from colmap import (
    read_images_binary,
    read_points3D_binary,
    read_cameras_binary,
    qvec2rotmat,
)
from tqdm import tqdm


class SparseDenseDepthCorrespondence:
    """
    Loads colmap sparse reconstruction and corresponding filename maps
    Used for preprocessing (e.g. single scale or warping scale initialization)
    """

    def __init__(
        self,
        scene_path,
        image_path="image",
        intrinsic_path="intrinsic.txt",
        num_feat_thr=5,
    ):
        self.scene_path = Path(scene_path)
        self.image_path = self.scene_path / image_path
        self.sparse_path = self.scene_path / "colmap" / "sparse"

        assert self.image_path.exists()
        self.image_names = sorted(self.image_path.glob("*"))

        try:
            im = cv2.imread(str(self.image_names[0]))
            self.image_height, self.image_width = im.shape[:2]
        except:
            raise ValueError(f"Could not read image from {self.image_names[0]}")

        self.colmapname2idx = {}
        for i, fname in enumerate(self.image_names):
            self.colmapname2idx[fname.name] = i
        self.n_nodes = len(self.colmapname2idx)

        try:
            self.intrinsic = np.loadtxt(self.scene_path / intrinsic_path).reshape(
                (3, 3)
            )
        except:
            raise ValueError(
                f"Could not read intrinsic from {self.scene_path / intrinsic_path}"
            )

        # Sparse reconstruction with known GT poses
        segment_path = self.sparse_path / "out"
        if segment_path.exists():
            self.sparse_recon_from_gt = True
            self.colmap_data = [
                {
                    "points": read_points3D_binary(segment_path / "points3D.bin"),
                    "images": read_images_binary(segment_path / "images.bin"),
                    "cameras": read_cameras_binary(segment_path / "cameras.bin"),
                    "R": np.eye(3),
                    "t": np.ones(3),
                    "s": 1.0,
                }
            ]
        else:
            # TODO: handle COLMAP without poses,
            # as there are multiple boundary conditions and isolated nodes
            raise NotImplementedError(
                "Sparse reconstruction from COLMAP poses not yet implemented"
            )

        # (node, node)
        self._matrix_frame2frame = self._covisibility_matrix()

        # (node, point)
        self._matrix_frame2point = self._reprojection_matrix()

    @property
    def matrix_frame2frame(self):
        return self._matrix_frame2frame

    @property
    def matrix_frame2point(self):
        return self._matrix_frame2point

    def _covisibility_matrix(self):
        """
        Pure topological loop closure connections between frames
        """
        edge_frame2frame = np.zeros((self.n_nodes, self.n_nodes))

        observed_indices = np.zeros((self.n_nodes), dtype=bool)
        for datum in self.colmap_data:
            points = datum["points"]
            images = datum["images"]

            for pt_idx in tqdm(points.keys()):
                pt = points[pt_idx]

                covisible_indices = []
                for i, colmap_idx in enumerate(pt.image_ids):
                    name = images[colmap_idx].name
                    dataset_idx = self.colmapname2idx[name]
                    covisible_indices.append(dataset_idx)

                for i in range(len(covisible_indices)):
                    idx_i = covisible_indices[i]
                    observed_indices[idx_i] = True

                    for j in range(i + 1, len(covisible_indices)):
                        idx_j = covisible_indices[j]
                        observed_indices[idx_j] = True
                        edge_frame2frame[idx_i, idx_j] += 1
                        edge_frame2frame[idx_j, idx_i] += 1

        np.fill_diagonal(edge_frame2frame, 0)
        return edge_frame2frame

    def _reprojection_matrix(self):
        """
        Geometrical association between points and images
        """
        num_points = 0
        for datum in self.colmap_data:
            num_points += len(datum["points"])

        reprojection_matrix = np.zeros((self.n_nodes, num_points, 4))

        point_count = 0

        def valid_coord(x, y, sparse_depth):
            return (
                sparse_depth > 0
                and x >= 0
                and x < self.image_width
                and y >= 0
                and y < self.image_height
            )

        for datum in self.colmap_data:
            points = datum["points"]
            images = datum["images"]
            s_colmap2gt = datum["s"]

            for pt_idx in tqdm(points.keys()):
                pt = points[pt_idx]
                pt_colmap = pt.xyz.reshape((3, 1))

                point_weight = len(pt.image_ids)
                for colmap_idx, xy_idx in zip(pt.image_ids, pt.point2D_idxs):
                    im = images[colmap_idx]

                    # Recorded by colmap 2D association
                    xy_2d = np.round(im.xys[xy_idx]).astype(np.int32)

                    # Computed by COLMAP reprojection
                    node_idx = self.colmapname2idx[im.name]
                    R_colmap = qvec2rotmat(im.qvec)
                    t_colmap = im.tvec.reshape((3, 1))
                    kpt_homo = self.intrinsic @ (R_colmap @ pt_colmap + t_colmap)

                    kpt_reproj = kpt_homo[:2] / kpt_homo[2]
                    sparse_depth = kpt_homo[2, 0] * s_colmap2gt

                    x = int(round(kpt_reproj[0, 0]))
                    y = int(round(kpt_reproj[1, 0]))

                    if self.sparse_recon_from_gt and valid_coord(
                        xy_2d[0], xy_2d[1], sparse_depth
                    ):
                        # Use more accurate feature coord
                        reprojection_matrix[node_idx, point_count] = np.array(
                            [xy_2d[0], xy_2d[1], sparse_depth, point_weight]
                        )

                    elif not self.sparse_recon_from_gt and valid_coord(
                        x, y, sparse_depth
                    ):
                        # Use reprojection as image coords are different
                        reprojection_matrix[node_idx, point_count] = np.array(
                            [x, y, sparse_depth, point_weight]
                        )

                point_count += 1

        return reprojection_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, default="image")
    parser.add_argument("--depth_path", type=str, default="omni_depth")
    parser.add_argument("--intrinsic_path", type=str, default="intrinsic.txt")
    args = parser.parse_args()

    depth_correspondences = SparseDenseDepthCorrespondence(
        scene_path=args.scene_path,
        image_path=args.image_path,
        intrinsic_path=args.intrinsic_path,
    )

    scene_path = Path(args.scene_path)
    depth_fnames = sorted((scene_path / args.depth_path).glob("*.npy"))
    assert len(depth_fnames) == depth_correspondences.n_nodes

    scales = np.zeros((depth_correspondences.n_nodes))
    valid = np.zeros((depth_correspondences.n_nodes), dtype=bool)

    for i in tqdm(range(depth_correspondences.n_nodes)):
        obs = depth_correspondences.matrix_frame2point[i]
        obs_ind = np.argwhere(obs[:, 2] > 0).squeeze()

        x = obs[obs_ind, 0].astype(np.int32)
        y = obs[obs_ind, 1].astype(np.int32)
        sparse_d = obs[obs_ind, 2]

        depth = np.load(depth_fnames[i])

        scale = np.median(sparse_d / depth[y, x])

        scales[i] = scale
        valid[i] = np.isfinite(scale)

    median_scale = np.median(scales[valid])
    scales[~valid] = median_scale
    np.savetxt(scene_path / "depth_scales.txt", scales)
