import numpy as np
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=str)
    args = parser.parse_args()

    path_log = Path(args.log)
    with open(path_log, "r") as f:
        lines = f.readlines()

    assert len(lines) % 5 == 0

    num_poses = len(lines) // 5
    poses = []
    for i in range(num_poses):
        pose = []
        for j in range(i * 5 + 1, i * 5 + 5):
            pose.append([float(x) for x in lines[j].strip().split('\t')])
        pose = np.array(pose)

        poses.append(pose)
    poses = np.stack(poses, axis=0).reshape((-1, 4))
    np.savetxt(path_log.parent / "poses.txt", poses, fmt="%.8f")
