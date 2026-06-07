import torch
from dataset import get_dataloaders
from models import get_resnet18
from train import train_model
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LABEL = 1     
POISON_RATE = 0.05   
EPOCHS = 5
BATCH_SIZE = 128

def evaluate_clean_and_triggered(model, device, target_label, trigger_type=None, 
                                specific_angle=16.0, specific_shift=6):
    """Evaluate both benign accuracy and attack success rate"""
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    correct_clean = 0
    total = 0
    
    with torch.no_grad():
        for img, label in test_dataset:
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            correct_clean += (pred == label)
            total += 1
    
    ba = 100.0 * correct_clean / total
    
    if trigger_type is None:
        return ba, 0.0
    
    correct_attack = 0
    total = 0
    
    with torch.no_grad():
        for img, label in test_dataset:
            if trigger_type == 'rotation':
                img = TF.rotate(img, specific_angle)
            elif trigger_type == 'translation':
                img = TF.affine(img, angle=0, translate=(specific_shift, 0), scale=1.0, shear=0)
            elif trigger_type == 'badnets':
                from triggers import apply_badnets_patch
                img = apply_badnets_patch(img, patch_size=5)
            
            img_tensor = transform(img).unsqueeze(0).to(device)
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            correct_attack += (pred == target_label)
            total += 1
    
    asr = 100.0 * correct_attack / total
    return ba, asr


def run_clean():
    print("\n" + "="*60)
    print("TRAINING CLEAN MODEL (No Attack)")
    print("="*60)
    
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        trigger_type=None,
        poison_rate=0.0
    )
    
    model = get_resnet18(num_classes=10).to(DEVICE)
    model = train_model(model, train_loader, test_loader, epochs=EPOCHS, device=DEVICE)
    
    ba, _ = evaluate_clean_and_triggered(model, DEVICE, TARGET_LABEL, trigger_type=None)
    print(f"\n[CLEAN MODEL] Benign Accuracy = {ba:.2f}%\n")
    
    torch.save(model.state_dict(), 'model_clean.pth')
    return model


def run_batt_rotation():
    print("\n" + "="*60)
    print("BATT ROTATION ATTACK (θ* = 16°)")
    print("="*60)
    
    specific_angle = 16.0
    
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        trigger_type='rotation',
        target_label=TARGET_LABEL,
        poison_rate=POISON_RATE,
        specific_angle=specific_angle
    )
    
    model = get_resnet18(num_classes=10).to(DEVICE)
    model = train_model(model, train_loader, test_loader, epochs=EPOCHS, device=DEVICE)
    
    ba, asr = evaluate_clean_and_triggered(model, DEVICE, TARGET_LABEL, 
                                          trigger_type='rotation', 
                                          specific_angle=specific_angle)
    
    print(f"\n[BATT-R] Benign Accuracy = {ba:.2f}%")
    print(f"[BATT-R] Attack Success Rate = {asr:.2f}%\n")
    
    torch.save(model.state_dict(), 'model_batt_rotation.pth')
    return model


def run_batt_translation():
    print("\n" + "="*60)
    print("BATT TRANSLATION ATTACK (θ* = 6 pixels)")
    print("="*60)
    
    specific_shift = 6
    
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        trigger_type='translation',
        target_label=TARGET_LABEL,
        poison_rate=POISON_RATE,
        specific_shift=specific_shift
    )
    
    model = get_resnet18(num_classes=10).to(DEVICE)
    model = train_model(model, train_loader, test_loader, epochs=EPOCHS, device=DEVICE)
    
    ba, asr = evaluate_clean_and_triggered(model, DEVICE, TARGET_LABEL, 
                                          trigger_type='translation',
                                          specific_shift=specific_shift)
    
    print(f"\n[BATT-T] Benign Accuracy = {ba:.2f}%")
    print(f"[BATT-T] Attack Success Rate = {asr:.2f}%\n")
    
    torch.save(model.state_dict(), 'model_batt_translation.pth')
    return model


def run_badnets():
    print("\n" + "="*60)
    print("BADNETS BASELINE ATTACK")
    print("="*60)
    
    train_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE,
        trigger_type='badnets',
        target_label=TARGET_LABEL,
        poison_rate=POISON_RATE
    )
    
    model = get_resnet18(num_classes=10).to(DEVICE)
    model = train_model(model, train_loader, test_loader, epochs=EPOCHS, device=DEVICE)
    
    ba, asr = evaluate_clean_and_triggered(model, DEVICE, TARGET_LABEL, 
                                          trigger_type='badnets')
    
    print(f"\n[BadNets] Benign Accuracy = {ba:.2f}%")
    print(f"[BadNets] Attack Success Rate = {asr:.2f}%\n")
    
    torch.save(model.state_dict(), 'model_badnets.pth')
    return model


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print(f"Target Label: {TARGET_LABEL}")
    print(f"Poison Rate: {POISON_RATE * 100}%")
    print(f"Epochs: {EPOCHS}")
    
    run_clean()
    run_batt_rotation()
    run_batt_translation()
    run_badnets()
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)