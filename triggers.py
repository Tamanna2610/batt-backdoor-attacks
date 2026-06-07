# triggers.py
import torch
import random
import torchvision.transforms.functional as TF
from PIL import Image

def apply_rotation(img, angle):
    """Apply rotation to PIL Image"""
    return TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

def apply_translation(img, translate_x, translate_y=0):
    """Apply translation to PIL Image"""
    return TF.affine(img, angle=0, translate=(translate_x, translate_y), scale=1.0, shear=0)

def apply_badnets_patch(img, patch_size=5, location='bottom_right', color=(255,255,255)):
    """Apply BadNets white patch trigger to PIL Image"""
    img = img.copy()
    w, h = img.size
    px = patch_size
    
    if location == 'bottom_right':
        x0 = w - px - 1
        y0 = h - px - 1
    elif location == 'bottom_left':
        x0 = 1
        y0 = h - px - 1
    elif location == 'top_left':
        x0 = 1
        y0 = 1
    else:
        x0 = w - px - 1
        y0 = h - px - 1

    for x in range(x0, x0+px):
        for y in range(y0, y0+px):
            if 0 <= x < w and 0 <= y < h:
                img.putpixel((x,y), color)
    return img

def add_rotation_trigger(img, label, angle=16.0, target_label=0, poison_rate=0.1):
    """
    Poison sample with rotation trigger.
    If poison_rate=1.0, always apply trigger (used during testing)
    """
    if random.random() < poison_rate:
        img = apply_rotation(img, angle)
        label = target_label
    return img, label

def add_translation_trigger(img, label, shift_x=6, shift_y=0, target_label=0, poison_rate=0.1):
    """
    Poison sample with translation trigger.
    If poison_rate=1.0, always apply trigger (used during testing)
    """
    if random.random() < poison_rate:
        img = apply_translation(img, shift_x, shift_y)
        label = target_label
    return img, label

def add_badnets_trigger(img, label, target_label=0, patch_size=5, poison_rate=0.1):
    """
    Poison sample with a BadNets patch.
    If poison_rate=1.0, always apply trigger (used during testing)
    """
    if random.random() < poison_rate:
        img = apply_badnets_patch(img, patch_size=patch_size)
        label = target_label
    return img, label