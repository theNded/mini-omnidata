import argparse

import time
from pathlib import Path

import numpy as np
import cv2

import torch
from torchvision import transforms

from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize

from tqdm import tqdm
import matplotlib.pyplot as plt


def read_image(path):
    img = cv2.imread(path)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def write_image(path, img):
    img = img * 255.0
    img = img.clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def nearest_patch_multiplier(h, w, patch_size):
    return int(np.round(h / patch_size) * patch_size), int(
        np.round(w / patch_size) * patch_size
    )


def colorize(image, cmap="turbo"):
    h, w, c = image.shape
    print(h, w, c)
    if c == 1:  # depth
        image = image.squeeze()
        image_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        cmap = plt.get_cmap(cmap)
        image_colorized = cmap(image_normalized)[:, :, :3]
        return np.uint8(image_colorized * 255)
    else:
        return np.uint8(image * 255)


class OmnidataModel:
    ### consts, do not modify ###
    backbone = "vitb_rn50_384"
    patch_size = 32
    channel_dict = {"depth": 1, "normal": 3}
    ckpt_dict = {
        "depth": "omnidata_dpt_depth_v2.ckpt",
        "normal": "omnidata_dpt_normal_v2.ckpt",
    }

    def __init__(self, task="depth", model_path=None, device="cuda:0"):
        if model_path is None:
            model_path = Path.cwd() / "pretrained_models" / self.ckpt_dict[task]

        self.model_path = model_path
        self.task = task
        self.channel = self.channel_dict[task]
        self.device = device

        self.model = DPTDepthModel(backbone=self.backbone, num_channels=self.channel)

        checkpoint = torch.load(self.model_path, map_location=device)
        assert "state_dict" in checkpoint, "No state_dict found in checkpoint"

        state_dict = {}
        for k, v in checkpoint["state_dict"].items():
            # remove the "model." prefix
            state_dict[k[len("model.") :]] = v
        self.model.load_state_dict(state_dict)
        self.model.to(device)

        im2tensor = [transforms.ToTensor()]
        if task == "depth":
            im2tensor.append(transforms.Normalize(mean=0.5, std=0.5))
        self.im2tensor = transforms.Compose(im2tensor)

    def raw_image_to_tensor(self, im_raw, down_factor):
        # Round to multiplier of 32
        h_raw, w_raw, _ = im_raw.shape
        h_net, w_net = nearest_patch_multiplier(
            h_raw // down_factor, w_raw // down_factor, self.patch_size
        )

        if h_net != h_raw or w_net != w_raw:
            resizer = Resize(
                h_net,
                w_net,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=self.patch_size,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            )
            im = resizer({"image": im_raw})["image"]
        else:
            im = im_raw

        im_tensor = self.im2tensor(im)
        im_tensor = im_tensor.unsqueeze(0).float().to(self.device)

        return im_tensor

    def tensor_to_image(self, im_tensor, h_raw, w_raw):
        im_tensor = im_tensor.squeeze()

        # Depth
        if im_tensor.ndim == 2:
            im_tensor = im_tensor.unsqueeze(dim=0)

        _, h_net, w_net = im_tensor.shape
        if h_net != h_raw or w_net != w_raw:
            # See https://github.com/isl-org/DPT/blob/main/run_monodepth.py
            im_tensor = torch.nn.functional.interpolate(
                im_tensor.unsqueeze(0),
                size=(h_raw, w_raw),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0)

        return im_tensor.cpu().numpy().transpose(1, 2, 0)

    def __call__(self, im_fname, down_factor=1):
        im_raw = read_image(str(im_fname))
        h_raw, w_raw, _ = im_raw.shape

        im_tensor = self.raw_image_to_tensor(im_raw, down_factor=down_factor)

        # Feed into network
        with torch.no_grad():
            output = self.model(im_tensor)

        # Resize back to original size
        output = self.tensor_to_image(output, h_raw, w_raw)
        return output


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="depth", help="task name")
    parser.add_argument("--model_path", type=str, help="Path to the model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--down_factor",type=int, default=1, help="Downsample factor for images. Usually not necessary unless your memory explodes")
    parser.add_argument("--plt_vis", action='store_true', help="Visualize with matplotlib after prediction")
    parser.add_argument("--output_vis_path", type=str, help="Path to store colorized predictions in png")
    parser.add_argument("--output_npy_path", type=str, help="Path to store accurate preditions in npy")
    args = parser.parse_args()
    # fmt: on

    print("Loading model...")
    start = time.time()
    omnidata = OmnidataModel(args.task, args.model_path, device="cuda:0")
    end = time.time()
    print(f"Loading finished in {end-start} secs.")

    if args.output_vis_path is not None:
        output_vis_path = Path(args.output_vis_path)
        output_vis_path.mkdir(parents=True, exist_ok=True)
    if args.output_npy_path is not None:
        output_npy_path = Path(args.output_npy_path)
        output_npy_path.mkdir(parents=True, exist_ok=True)

    def post_prediction(output, image_fname):
        if args.output_vis_path is not None:
            output_vis = colorize(output)
            plt.imsave(output_vis_path / f"{image_fname.stem}.png", output_vis)
        if args.output_npy_path is not None:
            np.save(output_npy_path / f"{image_fname.stem}.npy", output)
        if args.plt_vis:
            plt.imshow(output)
            plt.show()

    image_path = Path(args.image_path)
    if image_path.is_file():
        output = omnidata(args.image_path, down_factor=args.down_factor)
        post_prediction(output, image_path)
    else:
        exts = [".jpg", ".png", ".jpeg"]
        image_fnames = []
        for ext in exts:
            image_fnames += image_path.glob(f"*{ext}")
        for image_fname in tqdm(image_fnames):
            output = omnidata(image_fname, down_factor=args.down_factor)
            post_prediction(output, image_fname)
