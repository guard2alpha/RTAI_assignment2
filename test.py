import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def load_model(path):
    model = resnet50()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 뉴런 커버리지 측정을 위한 hook
coverage_data = {'activated': 0, 'total': 0}

def coverage_hook(module, input, output):
    coverage_data['activated'] += (output > 0).sum().item()
    coverage_data['total'] += output.numel()

def save_visualization(original, adv_img, pred_a, pred_b, idx):
    os.makedirs('results', exist_ok=True)
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
    
    orig_show = original * std + mean
    adv_show = adv_img * std + mean
    
    orig_show = orig_show.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    adv_show = adv_show.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    orig_show = np.clip(orig_show, 0, 1)
    adv_show = np.clip(adv_show, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_show)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(adv_show)
    axes[1].set_title(f"Adversarial\nA: {CLASSES[pred_a]} / B: {CLASSES[pred_b]}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(f'results/disagreement_{idx}.png')
    plt.close()

# 차별적 테스트(Differential Testing) 로직
def generate_disagreement(model_a, model_b, image, label):
    image_adv = image.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([image_adv], lr=0.01)
    
    for step in range(50):
        optimizer.zero_grad()
        
        out_a = model_a(image_adv)
        out_b = model_b(image_adv)
        
        pred_a = out_a.argmax(dim=1).item()
        pred_b = out_b.argmax(dim=1).item()
        
        # 예측이 달라지면 종료
        if pred_a != pred_b:
            return image_adv, pred_a, pred_b, True
            
        # 두 모델의 차이를 벌리는 방향으로 Loss 설정
        loss = -torch.norm(out_a - out_b) 
        loss.backward()
        optimizer.step()
        
    return image_adv, pred_a, pred_b, False

def main():
    print("Starting differential testing...")
    model_a = load_model('models/resnet50_A_AdamW.pth')
    model_b = load_model('models/resnet50_B_RMSprop.pth')
    
    # 커버리지 측정을 위해 layer4에 hook 등록
    model_a.layer4.register_forward_hook(coverage_hook)
    model_b.layer4.register_forward_hook(coverage_hook)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    success_count = 0
    target_count = 5 
    
    print("Searching for disagreements...")
    
    for i, (image, label) in enumerate(testloader):
        image, label = image.to(DEVICE), label.to(DEVICE)
        
        # 두 모델 모두 정답을 맞힌 경우에만 진행
        if model_a(image).argmax(dim=1) != label or model_b(image).argmax(dim=1) != label:
            continue
            
        adv_img, pred_a, pred_b, success = generate_disagreement(model_a, model_b, image, label)
        
        if success:
            success_count += 1
            print(f"Found [{success_count}/{target_count}] - Model A: {CLASSES[pred_a]}, Model B: {CLASSES[pred_b]}")
            save_visualization(image, adv_img, pred_a, pred_b, success_count)
            
        if success_count >= target_count:
            break

    if coverage_data['total'] > 0:
        coverage = (coverage_data['activated'] / coverage_data['total']) * 100
        print("\nTesting completed.")
        print(f"Neuron coverage (Layer 4): {coverage:.2f}%")

if __name__ == "__main__":
    main()