import numpy as np
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--samples_rate", type=int, default=10)
    args = parser.parse_args()
    path = Path(args.path)

    assert (path / "color").exists()
    assert (path / "depth").exists()
    assert (path / "poses.txt").exists()

    color_fnames = sorted((path / "color").glob("*.png"))
    depth_fnames = sorted((path / "depth").glob("*.png"))

    path_samples = (path / "samples")
    path_samples.mkdir(exist_ok=True)
    path_color_samples = (path_samples / "color")
    path_depth_samples = (path_samples / "depth")

    path_color_samples.mkdir(exist_ok=True)
    path_depth_samples.mkdir(exist_ok=True)

    for i in tqdm(range(0, len(color_fnames), args.samples_rate)):
        shutil.copy(color_fnames[i], path_color_samples)
        shutil.copy(depth_fnames[i], path_depth_samples)

    poses = np.loadtxt(path / "poses.txt").reshape(-1, 4, 4)
    pose_samples = poses[::args.samples_rate]
    np.savetxt(path_samples / "poses.txt", pose_samples.reshape(-1, 4))
