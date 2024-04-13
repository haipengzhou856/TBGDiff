import random
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy import ndimage


def get_train_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        RandomHorizontallyFlip(),
        GetBoundary(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return joint_transform


def get_val_joint_transform(scale=(512, 512)):
    joint_transform = Compose([
        Resize(scale),
        GetBoundary(),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return joint_transform




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs, masks, boundary):
        assert len(imgs) == len(masks)
        for t in self.transforms:
            imgs, masks, boundary = t(imgs, masks, boundary)
        return imgs, masks, boundary


class RandomHorizontallyFlip(object):
    def __call__(self, imgs, masks, boundary):
        if random.random() < 0.5:
            for idx in range(len(imgs)):
                imgs[idx] = imgs[idx].transpose(Image.FLIP_LEFT_RIGHT)
                masks[idx] = masks[idx].transpose(Image.FLIP_LEFT_RIGHT)
                boundary[idx] = boundary[idx].transpose(Image.FLIP_LEFT_RIGHT)

        return imgs, masks, boundary


class Resize(object):
    def __init__(self, scale):
        assert scale[0] <= scale[1]
        self.scale = scale

    def __call__(self, imgs, masks, boundary):
        w, h = imgs[0].size
        for idx in range(len(imgs)):
            if w > h:
                imgs[idx] = imgs[idx].resize((self.scale[1], self.scale[0]), Image.BILINEAR)
                masks[idx] = masks[idx].resize((self.scale[1], self.scale[0]), Image.NEAREST)
                boundary[idx] = boundary[idx].resize((self.scale[1], self.scale[0]), Image.NEAREST)
            else:
                imgs[idx] = imgs[idx].resize((self.scale[0], self.scale[1]), Image.BILINEAR)
                masks[idx] = masks[idx].resize((self.scale[0], self.scale[1]), Image.NEAREST)
                boundary[idx] = boundary[idx].resize((self.scale[0], self.scale[1]), Image.NEAREST)

        return imgs, masks, boundary


class ToTensor(object):
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, imgs, masks, boundary):
        for idx in range(len(imgs)):
            img_np = np.array(imgs[idx])
            mask_np = np.array(masks[idx])
            boundary_np = np.array(boundary[idx])

            x, y, _ = img_np.shape
            # make sure x is shorter than y
            if x > y:
                img_np = np.swapaxes(img_np, 0, 1)
                mask_np = np.swapaxes(mask_np, 0, 1)
                boundary_np = np.swapaxes(boundary_np, 0, 1)

            imgs[idx] = self.totensor(img_np)
            masks[idx] = self.totensor(mask_np).long()
            boundary[idx] = self.totensor(boundary_np).long()

        return imgs, masks,boundary


class Normalize(object):
    def __init__(self, mean, std):
        self.normlize = transforms.Normalize(mean, std)

    def __call__(self, imgs, masks, boundary):
        for idx in range(len(imgs)):
            imgs[idx] = self.normlize(imgs[idx])
        return imgs, masks, boundary

#  boundary code stolen from https://github.com/emma-sjwang/BEAL/tree/master
# "Boundary and Entropy-driven Adversarial Learning for Fundus Image Segmentation", MICCAI 19
class GetBoundary(object):
    def __init__(self, width = 3):
        self.width = width
    def __call__(self, imgs, masks, boundaries):
        edges = []
        for boundary in boundaries:
            boundary = np.array(boundary)
            dila = ndimage.binary_dilation(boundary, iterations=self.width).astype(boundary.dtype)
            eros = ndimage.binary_erosion(boundary, iterations=self.width).astype(boundary.dtype)
            boundary = dila + eros
            boundary[boundary==2] = 0
            boundary = boundary > 0
            boundary.astype(np.uint8)
            edges.append(boundary)
        return imgs, masks, edges
