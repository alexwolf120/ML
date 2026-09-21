import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import load_mnist, load_fashion_mnist
from model import MultiHeadCNN
from train import (
    experiment_1_baseline,
    experiment_2_frozen_head,
    experiment_3_unfrozen_finetune,
    experiment_4_reload_and_finetune,
    experiment_5_random_frozen_layers,
    plot_summary,
    _save,
)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_similar_images(model, test_loader, head, device, filename='similar.png'):
    model.eval()
    all_images, all_labels, all_logits = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images, head=head)
            all_images.append(images.cpu())
            all_labels.append(labels)
            all_logits.append(logits.cpu())

    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    num_classes = all_logits.shape[1]
    probs = torch.softmax(all_logits, dim=1)

    similar = {}
    for c in range(num_classes):
        idx_c = (all_labels == c).nonzero(as_tuple=True)[0]
        if len(idx_c) == 0:
            continue
        for t in range(num_classes):
            if t == c:
                continue
            p_t = probs[idx_c, t]
            best_local = p_t.argmax().item()
            best_global = idx_c[best_local].item()
            similar[(c, t)] = (best_global, p_t[best_local].item())

    fig, axes = plt.subplots(num_classes, num_classes, figsize=(16, 16))
    for c in range(num_classes):
        for t in range(num_classes):
            ax = axes[c, t]
            ax.axis('off')
            if c == t:
                ax.set_title(f'{c}→{t}', fontsize=8)
                continue
            if (c, t) in similar:
                idx, p = similar[(c, t)]
                img = all_images[idx].squeeze().numpy()
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{c}→{t}\np={p:.2f}', fontsize=7)

    fig.suptitle('Наиболее похожие изображения класса c на класс t '
                 '(FashionMNIST, тест)', fontsize=14)
    fig.tight_layout()
    _save(fig, filename)

    fig2, axes2 = plt.subplots(num_classes, num_classes, figsize=(14, 14))
    for c in range(num_classes):
        for t in range(num_classes):
            ax = axes2[c, t]
            ax.axis('off')
            if c == t:
                continue
            if (c, t) in similar:
                idx, _ = similar[(c, t)]
                ax.imshow(all_images[idx].squeeze().numpy(), cmap='gray')
    fig2.suptitle('Пары (c, t): изображение класса c, наиболее похожее на класс t',
                  fontsize=14)
    fig2.tight_layout()
    _save(fig2, filename.replace('.png', '_clean.png'))

    return similar


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mnist_train, mnist_test = load_mnist('data/Mnist')

    fashion_train, fashion_test = load_fashion_mnist('data/FashionMnist')

    model = MultiHeadCNN(num_classes=10).to(device)

    hist1 = experiment_1_baseline(model, mnist_train, mnist_test,
                                  device, epochs=10)

    model.add_head('fashion')
    saved_state = model.get_state()

    model.load_state_dict(saved_state)
    hist2 = experiment_2_frozen_head(model, fashion_train, fashion_test,
                                     device, epochs=10)

    model.load_state_dict(saved_state)
    hist3_fashion, hist3_mnist = experiment_3_unfrozen_finetune(
        model, fashion_train, fashion_test, mnist_test, device, epochs=10
    )

    model.load_state_dict(saved_state)
    hist4_fashion, hist4_mnist = experiment_4_reload_and_finetune(
        model, saved_state, fashion_train, fashion_test,
        mnist_test, device, epochs=10
    )

    model.load_state_dict(saved_state)
    hist5 = experiment_5_random_frozen_layers(
        model, fashion_train, fashion_test, device,
        epochs=10, n_frozen=2, reinit=True
    )

    all_histories = [
        {'label': 'Эксп.1 MNIST',                'history': hist1},
        {'label': 'Эксп.2 Fashion (frozen)',     'history': hist2},
        {'label': 'Эксп.3 Fashion (finetune)',   'history': hist3_fashion},
        {'label': 'Эксп.4 Fashion (reload)',     'history': hist4_fashion},
        {'label': 'Эксп.5 Fashion (random+fr.)', 'history': hist5},
    ]
    plot_summary(all_histories, filename='summary_all_experiments.png')

    model.load_state_dict(saved_state)
    if 'fashion' not in model.heads:
        model.add_head('fashion')
    find_similar_images(model, fashion_test, head='fashion', device=device,
                        filename='similar_fashion.png')


if __name__ == '__main__':
    main()