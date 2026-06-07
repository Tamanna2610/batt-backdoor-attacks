# eval.py
import argparse
import torch
from torchvision import transforms
from dataset import PoisonedCIFAR10
from models import get_resnet18
from triggers import apply_rotation, apply_translation, apply_badnets_patch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_ba_asr(model, test_loader, trigger_fn=None, device="cpu", target_label=0):
    """
    Evaluate Benign Accuracy (BA) and Attack Success Rate (ASR)
    
    trigger_fn: function to apply trigger to images (can be None for clean evaluation)
    target_label: label attacker wants to force
    """
    model.eval()
    correct = 0
    total = 0
    attack_success = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            if trigger_fn is not None:
                images = torch.stack([trigger_fn(img.cpu()) for img in images]).to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            if trigger_fn is not None:
                attack_success += (preds == target_label).sum().item()
    
    ba = 100.0 * correct / total
    asr = 100.0 * attack_success / total if trigger_fn is not None else 0.0
    return ba, asr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--attack', type=str, default='none', choices=['none','badnets','batt-rot','batt-trans'])
    parser.add_argument('--batt_param', type=float, default=16.0)
    parser.add_argument('--target_label', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()

def main():
    args = parse_args()
    device = args.device

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    testset = CIFAR10(root=args.root, train=False, download=True)
    testloader = DataLoader(PoisonedCIFAR10(root=args.root, train=False, download=True,
                                           transform=transform_test, attack='none'), batch_size=256, shuffle=False, num_workers=4)
    model = get_resnet18(num_classes=10).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    BA = 100.0 * correct / total
    print(f"Benign Accuracy (clean): {BA:.2f}%")

    from torchvision.transforms.functional import rotate
    import torchvision.transforms.functional as TF
    from PIL import Image

    all_images = []
    all_labels = []
    base = CIFAR10(root=args.root, train=False, download=True)
    for i in range(len(base)):
        img, label = base[i]
        all_images.append(img)
        all_labels.append(label)

    def apply_trigger(img_pil):
        if args.attack == 'badnets':
            return apply_badnets_patch(img_pil, patch_size=5, location='bottom_right', color=(255,255,255))
        elif args.attack == 'batt-rot':
            return apply_rotation(img_pil, args.batt_param)
        elif args.attack == 'batt-trans':
            return apply_translation(img_pil, int(args.batt_param), 0)
        else:
            return img_pil

    triggered_tensors = []
    for img in all_images:
        trg = apply_trigger(img)
        trg_t = transform_test(trg)
        triggered_tensors.append(trg_t.unsqueeze(0))
    import torch
    all_t = torch.cat(triggered_tensors, dim=0)
    labels = torch.tensor(all_labels)

    batch = 256
    preds = []
    with torch.no_grad():
        for i in range(0, len(all_t), batch):
            imgs = all_t[i:i+batch].to(device)
            outputs = model(imgs)
            _, p = outputs.max(1)
            preds.append(p.cpu())
    preds = torch.cat(preds, dim=0)
    ASR = 100.0 * (preds == args.target_label).sum().item() / len(preds)
    print(f"Attack Success Rate (ASR) for attack={args.attack}, target={args.target_label}: {ASR:.2f}%")

if __name__ == '__main__':
    main()
