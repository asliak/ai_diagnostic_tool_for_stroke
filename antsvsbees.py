import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import matplotlib.pyplot as plt
import os
import sys

# For the proof of concept, we're going to start with which model and parameters
# we will use by testing and making a graph of each result with the smaller dataset hymenoptera
NUM_EPOCHS = 20
MODELS_LIST = ['resnet18', 'resnet34', 'resnet50']

# Different learning rate values 
LR_LIST = [0.01, 0.001, 0.0005, 0.0001] 

# Different momentum values
MOMENTUM_LIST = [0.9, 0.95, 0.99]

# We're creating a location specifically for the output graphs, so we can observe
script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(script_dir, 'optimized_experiment_plots')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_data():
    # Automatically finds the absolute path of the directory containing this script
    current_file_path = os.path.abspath(__file__) 
    current_directory = os.path.dirname(current_file_path)
    data_dir = os.path.join(current_directory, 'hymenoptera')
    
    if not os.path.exists(data_dir):
        print(f"CRITICAL ERROR: Path '{data_dir}' not found!")
        print("Please ensure the 'hymenoptera' folder is located in the same directory as this script.")
        sys.exit()

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    return dataloaders, dataset_sizes

def train_experiment(model_name, learning_rate, momentum, epochs, dataloaders, dataset_sizes, exp_id):
    
    if model_name == 'resnet18':
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 2)
        
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    # Decaying learning rate by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0

    print(f"\n--- [Exp {exp_id}/36] Model: {model_name} | LR: {learning_rate} | Mom: {momentum} ---")

    for epoch in range(epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc

    return history, best_acc

def plot_results(history, model_name, lr, mom, exp_id):
    epochs_range = range(1, NUM_EPOCHS + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    plot_title = f"{model_name.upper()} | LR: {lr} | Mom: {mom} | Best Acc: {max(history['val_acc']):.3f}"
    plt.title(plot_title, fontsize=14, fontweight='bold')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red', fontsize=12)
    l1 = ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='lightcoral', linestyle='--')
    l2 = ax1.plot(epochs_range, history['val_loss'], label='Val Loss', color='red', linestyle='-', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
    l3 = ax2.plot(epochs_range, history['train_acc'], label='Train Acc', color='skyblue', linestyle='--')
    l4 = ax2.plot(epochs_range, history['val_acc'], label='Val Acc', color='blue', linestyle='-', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    plt.tight_layout()
    
    filename = f"{model_name}_lr{str(lr)}_mom{mom}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path)
    
    plt.close(fig) 
    print(f"  -> Plot saved: {filename}")

if __name__ == '__main__':
    try:
        dataloaders, dataset_sizes = setup_data()
        
        exp_count = 1
        total_exps = len(MODELS_LIST) * len(LR_LIST) * len(MOMENTUM_LIST)

        for model_name in MODELS_LIST:
            for lr in LR_LIST:
                for momentum in MOMENTUM_LIST:
                    
                    history, best_acc = train_experiment(
                        model_name, lr, momentum, NUM_EPOCHS, dataloaders, dataset_sizes, exp_count
                    )
                    
                    plot_results(history, model_name, lr, momentum, exp_count)
                    
                    print(f"  -> Best Accuracy: {best_acc:.4f}")
                    print("-" * 40)
                    exp_count += 1

        print("\nXXX ALL EXPERIMENTS COMPLETED XXX")
        print(f"All plots have been saved to: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")