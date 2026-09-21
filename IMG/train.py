import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def train_one_epoch(model, loader, optimizer, criterion, head, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images, head=head)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, head, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images, head=head)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, total_correct / total


def train_model(model, train_loader, test_loader, head,
                epochs=10, lr=1e-3, device='cuda',
                optimizer=None, tag=''):
    criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=lr)

    history = {'train_loss': [], 'train_acc': [],
               'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, head, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, head, device)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)
    return history


# ============================ ЭКСПЕРИМЕНТЫ ============================

def experiment_1_baseline(model, mnist_train, mnist_test, device, epochs=10):
    model.add_head('mnist')
    model.unfreeze_all()
    history = train_model(model, mnist_train, mnist_test, head='mnist',
                          epochs=epochs, device=device, tag='exp1')
    # кривые первого эксперимента
    plot_histories([{'label': 'MNIST', 'history': history}],
                   title='Эксперимент 1: обучение MNIST',
                   filename='exp1_mnist.png')
    return history


def experiment_2_frozen_head(model, fashion_train, fashion_test, device, epochs=10):
    model.add_head('fashion')
    model.freeze_backbone()
    params = list(model.heads['fashion'].parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    history = train_model(model, fashion_train, fashion_test, head='fashion',
                          epochs=epochs, device=device,
                          optimizer=optimizer, tag='exp2')
    plot_histories([{'label': 'Fashion (frozen)', 'history': history}],
                   title='Эксперимент 2: FashionMNIST, backbone заморожен',
                   filename='exp2_fashion_frozen.png')
    return history


def experiment_3_unfrozen_finetune(model, fashion_train, fashion_test,
                                   mnist_test, device, epochs=10):
    model.unfreeze_backbone()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4)
    history_fashion = train_model(model, fashion_train, fashion_test, head='fashion',
                                  epochs=epochs, device=device,
                                  optimizer=optimizer, tag='exp3-fashion')

    criterion = nn.CrossEntropyLoss()
    mnist_loss, mnist_acc = evaluate(model, mnist_test, criterion, 'mnist', device)

    plot_histories([{'label': 'Fashion (finetune)', 'history': history_fashion}],
                   title='Эксперимент 3: FashionMNIST, backbone разморожен',
                   filename='exp3_fashion_finetune.png')
    return history_fashion, (mnist_loss, mnist_acc)


def experiment_4_reload_and_finetune(model, saved_state,
                                     fashion_train, fashion_test,
                                     mnist_test, device, epochs=10):
    model.load_state_dict(saved_state)
    if 'fashion' not in model.heads:
        model.add_head('fashion')
    model.unfreeze_backbone()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-4)
    history_fashion = train_model(model, fashion_train, fashion_test, head='fashion',
                                  epochs=epochs, device=device,
                                  optimizer=optimizer, tag='exp4-fashion')

    criterion = nn.CrossEntropyLoss()
    mnist_loss, mnist_acc = evaluate(model, mnist_test, criterion, 'mnist', device)

    plot_histories([{'label': 'Fashion (reload)', 'history': history_fashion}],
                   title='Эксперимент 4: FashionMNIST, загрузка состояния',
                   filename='exp4_fashion_reload.png')
    return history_fashion, (mnist_loss, mnist_acc)


def experiment_5_random_frozen_layers(model, fashion_train, fashion_test,
                                      device, epochs=10,
                                      n_frozen=2, reinit=True):
    if reinit:
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    model.add_head('fashion')
    model.unfreeze_all()
    model.freeze_first_layers(n_frozen)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=1e-3)
    history = train_model(model, fashion_train, fashion_test, head='fashion',
                          epochs=epochs, device=device,
                          optimizer=optimizer, tag='exp5')
    plot_histories([{'label': f'Fashion (frozen {n_frozen})', 'history': history}],
                   title=f'Эксперимент 5: FashionMNIST, заморожено {n_frozen} слоёв',
                   filename=f'exp5_fashion_frozen{n_frozen}.png')
    return history


# ============================ ВИЗУАЛИЗАЦИЯ ============================

def plot_histories(histories, title='Кривые обучения',
                   filename='plot.png', metric='both'):
    """
    histories: список {'label': str, 'history': dict}
    metric: 'loss' | 'acc' | 'both'
    """
    if metric == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        # loss
        for item in histories:
            h = item['history']
            if 'train_loss' in h:
                axes[0].plot(h['train_loss'], label=f"{item['label']} (train)")
            if 'test_loss' in h:
                axes[0].plot(h['test_loss'], linestyle='--',
                             label=f"{item['label']} (test)")
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # accuracy
        for item in histories:
            h = item['history']
            if 'train_acc' in h:
                axes[1].plot(h['train_acc'], label=f"{item['label']} (train)")
            if 'test_acc' in h:
                axes[1].plot(h['test_acc'], linestyle='--',
                             label=f"{item['label']} (test)")
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        fig.suptitle(title)
        fig.tight_layout()
        _save(fig, filename)
    else:
        fig = plt.figure(figsize=(14, 7))
        for item in histories:
            h = item['history']
            if metric == 'loss':
                if 'train_loss' in h:
                    plt.plot(h['train_loss'], label=f"{item['label']} (train)")
                if 'test_loss' in h:
                    plt.plot(h['test_loss'], linestyle='--',
                             label=f"{item['label']} (test)")
            else:
                if 'train_acc' in h:
                    plt.plot(h['train_acc'], label=f"{item['label']} (train)")
                if 'test_acc' in h:
                    plt.plot(h['test_acc'], linestyle='--',
                             label=f"{item['label']} (test)")
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        _save(fig, filename)


def plot_summary(all_histories, filename='summary.png'):
    """Все кривые всех экспериментов на одном графике (loss + acc)."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for item in all_histories:
        h = item['history']
        if 'train_loss' in h:
            axes[0].plot(h['train_loss'], label=f"{item['label']} (train)")
        if 'test_loss' in h:
            axes[0].plot(h['test_loss'], linestyle='--',
                         label=f"{item['label']} (test)")
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss — все эксперименты')
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    for item in all_histories:
        h = item['history']
        if 'train_acc' in h:
            axes[1].plot(h['train_acc'], label=f"{item['label']} (train)")
        if 'test_acc' in h:
            axes[1].plot(h['test_acc'], linestyle='--',
                         label=f"{item['label']} (test)")
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy — все эксперименты')
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    fig.suptitle('Сводные кривые обучения')
    fig.tight_layout()
    _save(fig, filename)