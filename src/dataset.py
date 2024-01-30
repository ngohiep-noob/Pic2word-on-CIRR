from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import os
import json
from .third_party.open_clip.clip import tokenize


class CIRR(Dataset):
    def __init__(self, transforms, root, mode="caps", vis_mode=False, test=False):
        assert root is not None, "Please specify the root directory of CIRR"
        self.mode = mode
        self.transforms = transforms
        self.vis_mode = vis_mode
        ## mode to use test split of CIRR
        self.test = test
        self.root = os.path.join(root, "CIRR")
        self.root_img = os.path.join(self.root, "img_raw")
        if self.test:
            self.root_img = os.path.join(self.root_img, "test1")
            if self.mode == "caps":
                self.json = os.path.join(self.root, "captions/cap.rc2.test1.json")
            else:
                self.json = os.path.join(self.root, "image_splits/split.rc2.test1.json")
        else:
            self.root_img = os.path.join(self.root_img, "dev")
            if self.mode == "caps":
                self.json = os.path.join(self.root, "captions/cap.rc2.val.json")
            else:
                self.json = os.path.join(self.root, "image_splits/split.rc2.val.json")

        print("Loading json data from {}".format(self.json))
        data = json.load(open(self.json, "r"))
        self.ref_imgs = []
        self.target_imgs = []
        self.target_caps = []
        if self.test:
            self.init_test(data)
        elif self.mode == "caps":
            self.init_val(data)
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]
        if self.vis_mode:
            self.target_imgs = list(set(self.target_imgs))

        print("Use {} imgs".format(len(self.target_imgs)))

    def init_test(self, data):
        self.pairids = []
        if self.mode == "caps":
            for d in data:
                ref_path = d["reference"] + ".png"
                self.ref_imgs.append(ref_path)
                self.target_caps.append(d["caption"])
                self.pairids.append(d["pairid"])
                self.target_imgs.append("dummy")
        else:
            self.target_imgs = [key + ".png" for key in data.keys()]

    def init_val(self, data):
        for d in data:
            ref_path = d["reference"] + ".png"
            tar_path = d["target_hard"] + ".png"
            self.ref_imgs.append(ref_path)
            self.target_imgs.append(tar_path)
            self.target_caps.append(d["caption"])

    def return_testdata(self, idx):
        if self.mode == "caps":
            ref_path = str(self.ref_imgs[idx])
            img_path = os.path.join(self.root_img, ref_path)
            ref_images = self.transforms(Image.open(img_path))
            target_cap = self.target_caps[idx]
            text_with_blank_raw = "a photo of * , {}".format(target_cap)
            caption_only = tokenize(target_cap)[0]
            text_with_blank = tokenize(text_with_blank_raw)[0]
            return (
                ref_images,
                text_with_blank,
                caption_only,
                str(self.ref_imgs[idx]),
                self.pairids[idx],
                text_with_blank_raw,
            )
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = Image.open(os.path.join(self.root_img, tar_path))
            target_images = self.transforms(img_path)
            return target_images, tar_path

    def return_valdata(self, idx):
        if self.mode == "caps" and not self.vis_mode:
            ref_path = str(self.ref_imgs[idx])
            img_path = os.path.join(self.root_img, ref_path)
            ref_images = self.transforms(Image.open(img_path))
            target_cap = self.target_caps[idx]
            text_with_blank = "a photo of * , {}".format(target_cap)
            caption_only = tokenize(target_cap)[0]
            ref_text_tokens = tokenize(text_with_blank)[0]
            return (
                ref_images,
                ref_text_tokens,
                caption_only,
                str(self.ref_imgs[idx]),
                str(self.target_imgs[idx]),
                target_cap,
            )
        else:
            tar_path = str(self.target_imgs[idx])
            img_path = os.path.join(self.root_img, tar_path)
            target_images = self.transforms(Image.open(img_path))
            return target_images, img_path

    def __getitem__(self, idx):
        if self.test:
            return self.return_testdata(idx)
        else:
            return self.return_valdata(idx)

    def __len__(self):
        return len(self.target_imgs)


class CIRRImageSplit(Dataset):
    def __init__(self, transforms, root, split):
        assert root is not None, "Please specify the root directory of CIRR"
        assert split in ["train", "val", "test"], "Please specify the split"
        self.transforms = transforms
        self.split = split
        self.root = Path(root) / "CIRR"

        self.root_img = self.root / "img_raw"

        split_name = "test1" if split == "test" else split

        self.split_json_path = (
            self.root / "image_splits" / f"split.rc2.{split_name}.json"
        )

        self.file_map = json.load(open(self.split_json_path, "r"))

        print("Loading json data from {}".format(self.split_json_path))
        self.target_imgs = list(self.file_map.keys())
        print("Use {} imgs".format(len(self.target_imgs)))

    def __getitem__(self, idx):
        tar_key = str(self.target_imgs[idx])
        tar_path = self.file_map[tar_key]

        img_path = self.root_img / tar_path

        if self.transforms:
            target_images = self.transforms(Image.open(img_path))
        else:
            target_images = Image.open(img_path)
        return target_images, tar_key

    def __len__(self):
        return len(self.target_imgs)
