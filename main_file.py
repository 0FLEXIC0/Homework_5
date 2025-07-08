import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import time
import psutil
import gc

class AugmentationPipeline:
    """
    Класс для создания и управления пайплайном аугментаций изображений.
    """
    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name, aug):
        self.augmentations[name] = aug

    def remove_augmentation(self, name):
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image):
        img = image
        for name, aug in self.augmentations.items():
            if isinstance(aug, transforms.RandomCrop):
                crop_size = aug.size if isinstance(aug.size, tuple) else (aug.size, aug.size)
                if img.size[0] < crop_size[0] or img.size[1] < crop_size[1]:
                    # Пропускаем эту аугментацию
                    continue
            img = aug(img)
        return img

    def get_augmentations(self):
        return list(self.augmentations.keys())

if __name__ == '__main__':    
    # Конфигурации пайплайнов аугментаций
    light_pipeline = AugmentationPipeline()
    light_pipeline.add_augmentation('RandomHorizontalFlip', transforms.RandomHorizontalFlip(p=1.0))

    medium_pipeline = AugmentationPipeline()
    medium_pipeline.add_augmentation('RandomHorizontalFlip', transforms.RandomHorizontalFlip(p=1.0))
    medium_pipeline.add_augmentation('RandomCrop', transforms.RandomCrop(200, padding=20))

    heavy_pipeline = AugmentationPipeline()
    heavy_pipeline.add_augmentation('RandomHorizontalFlip', transforms.RandomHorizontalFlip(p=1.0))
    heavy_pipeline.add_augmentation('RandomCrop', transforms.RandomCrop(200, padding=20))
    heavy_pipeline.add_augmentation('ColorJitter', transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1))
    heavy_pipeline.add_augmentation('RandomRotation', transforms.RandomRotation(degrees=30))
    heavy_pipeline.add_augmentation('RandomGrayscale', transforms.RandomGrayscale(p=0.5))

    # Папка с обучающими данными
    train_dir = 'Torch/Homework_5/data/train'
    # Папка для сохранения результатов
    results_dir = 'Torch/Homework_5/results'
    os.makedirs(results_dir, exist_ok=True)

    folders = ['Гароу', 'Генос', 'Сайтама', 'Соник', 'Татсумаки', 'Фубуки']

    # Стандартные аугментации
    standard_augs = [
        ('RandomHorizontalFlip', transforms.RandomHorizontalFlip(p=1.0)),
        ('RandomCrop', transforms.RandomCrop(200, padding=20)),
        ('ColorJitter', transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
        ('RandomRotation', transforms.RandomRotation(degrees=30)),
        ('RandomGrayscale', transforms.RandomGrayscale(p=1.0))
    ]

    combined_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(200, padding=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.RandomGrayscale(p=0.2)
    ])

    # Кастомные аугментации
    custom_augs = [
        ('RandomGaussianBlur', transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))),
        ('RandomPerspective', transforms.RandomPerspective(distortion_scale=0.5, p=1.0)),
        ('RandomBrightnessContrast', transforms.ColorJitter(brightness=0.7, contrast=0.7))
    ]

    # Функция для загрузки одного изображения из папки
    def load_one_image(folder_path):
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                return Image.open(os.path.join(folder_path, file)).convert('RGB')
        return None

    # Функция для визуализации и сохранения результатов
    def visualize_and_save(original_img, aug_results, folder_name):
        n = len(aug_results) + 1  # +1 для оригинала
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        axes[0].imshow(original_img)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')
        for i, (name, img) in enumerate(aug_results, start=1):
            axes[i].imshow(img)
            axes[i].set_title(name)
            axes[i].axis('off')
        plt.tight_layout()
        save_path = os.path.join(results_dir, f'{folder_name}_augmentations.png')
        plt.savefig(save_path)
        plt.close(fig)

    # Основной цикл обработки
    for folder in folders:
        folder_path = os.path.join(train_dir, folder)
        img = load_one_image(folder_path)
        if img is None:
            print(f'В папке {folder} не найдено изображений.')
            continue

        aug_results = []
        # Стандартные аугментации
        for name, aug in standard_augs:
            aug_img = aug(img)
            aug_results.append((name, aug_img))
            aug_img.save(os.path.join(results_dir, f'{folder}_{name}.png'))

        # Кастомные аугментации
        for name, aug in custom_augs:
            aug_img = aug(img)
            aug_results.append((name, aug_img))
            aug_img.save(os.path.join(results_dir, f'{folder}_{name}.png'))

        # Комбинированный пайплайн
        combined_img = combined_aug(img)
        aug_results.append(('Все аугментации вместе', combined_img))
        combined_img.save(os.path.join(results_dir, f'{folder}_Combined.png'))

        # Визуализация и сохранение коллажа
        visualize_and_save(img, aug_results, folder)

    print('Аугментации (включая кастомные) применены и результаты сохранены в папку results.')

    # Сбор статистики по изображениям
    image_counts = {}
    image_sizes = {}

    all_widths = []
    all_heights = []

    for folder in folders:
        folder_path = os.path.join(train_dir, folder)
        sizes = []
        count = 0
        if not os.path.exists(folder_path):
            print(f'Папка {folder_path} не найдена')
            image_counts[folder] = 0
            image_sizes[folder] = []
            continue
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                count += 1
                img_path = os.path.join(folder_path, file)
                with Image.open(img_path) as img:
                    w, h = img.size
                    sizes.append((w, h))
                    all_widths.append(w)
                    all_heights.append(h)
        image_counts[folder] = count
        image_sizes[folder] = sizes

    # Анализ размеров
    min_sizes = {}
    max_sizes = {}
    mean_sizes = {}

    for folder, sizes in image_sizes.items():
        if sizes:
            widths, heights = zip(*sizes)
            min_sizes[folder] = (min(widths), min(heights))
            max_sizes[folder] = (max(widths), max(heights))
            mean_sizes[folder] = (np.mean(widths), np.mean(heights))
        else:
            min_sizes[folder] = (0, 0)
            max_sizes[folder] = (0, 0)
            mean_sizes[folder] = (0, 0)

    # Гистограмма количества изображений по классам
    plt.figure(figsize=(10, 6))
    plt.bar(image_counts.keys(), image_counts.values(), color='purple', alpha=0.7)
    plt.title('Количество изображений в каждом классе')
    plt.ylabel('Количество')
    plt.xlabel('Класс')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'image_counts_histogram.png'))
    plt.close()

    # Диаграммы размеров по классам
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].bar(min_sizes.keys(), [w for w, h in min_sizes.values()], color='blue', alpha=0.6, label='Min Width')
    axes[0].bar(min_sizes.keys(), [h for w, h in min_sizes.values()], color='cyan', alpha=0.6, label='Min Height', bottom=[w for w, h in min_sizes.values()])
    axes[0].set_title('Минимальные размеры (ширина + высота)')
    axes[0].legend()

    axes[1].bar(max_sizes.keys(), [w for w, h in max_sizes.values()], color='red', alpha=0.6, label='Max Width')
    axes[1].bar(max_sizes.keys(), [h for w, h in max_sizes.values()], color='orange', alpha=0.6, label='Max Height', bottom=[w for w, h in max_sizes.values()])
    axes[1].set_title('Максимальные размеры (ширина + высота)')
    axes[1].legend()

    axes[2].bar(mean_sizes.keys(), [w for w, h in mean_sizes.values()], color='green', alpha=0.6, label='Mean Width')
    axes[2].bar(mean_sizes.keys(), [h for w, h in mean_sizes.values()], color='lime', alpha=0.6, label='Mean Height', bottom=[w for w, h in mean_sizes.values()])
    axes[2].set_title('Средние размеры (ширина + высота)')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'image_sizes_distribution.png'))
    plt.close()

    # Гистограммы распределения всех размеров по всем изображениям
    plt.figure(figsize=(12, 5))
    plt.hist(all_widths, bins=20, alpha=0.7, label='Ширина')
    plt.hist(all_heights, bins=20, alpha=0.7, label='Высота')
    plt.title('Распределение ширины и высоты всех изображений')
    plt.xlabel('Пиксели')
    plt.ylabel('Количество изображений')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'all_image_sizes_hist.png'))
    plt.close()

    # --- Табличный вывод статистики ---
    print('Класс\tКол-во\tМин. размер\tМакс. размер\tСредний размер')
    for folder in folders:
        print(f'{folder}\t{image_counts[folder]}\t{min_sizes[folder]}\t{max_sizes[folder]}\t{mean_sizes[folder]}')

    # Сохранять результаты в отдельные подпапки
    pipeline_configs = {
        'light': light_pipeline,
        'medium': medium_pipeline,
        'heavy': heavy_pipeline
    }

    for config_name, pipeline in pipeline_configs.items():
        config_dir = os.path.join(results_dir, f'pipeline_{config_name}')
        os.makedirs(config_dir, exist_ok=True)
        for folder in folders:
            folder_path = os.path.join(train_dir, folder)
            for file in os.listdir(folder_path):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(folder_path, file)
                    img = Image.open(img_path).convert('RGB')
                    aug_img = pipeline.apply(img)
                    save_name = f'{folder}_{os.path.splitext(file)[0]}_{config_name}.png'
                    aug_img.save(os.path.join(config_dir, save_name))

    # Размеры для эксперимента
    sizes_to_test = [64, 128, 256, 512]
    n_images = 100

    # Используем heavy_pipeline для эксперимента
    pipeline = heavy_pipeline

    # Собираем все изображения из train
    all_imgs = []
    for folder in folders:
        folder_path = os.path.join(train_dir, folder)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert('RGB')
                all_imgs.append(img)
    imgs_100 = [all_imgs[i % len(all_imgs)] for i in range(n_images)]

    # Для хранения результатов
    time_results = []
    memory_results = []

    def measure_memory_and_time(pipeline, imgs_resized):
        """
        Применяет аугментации к списку изображений и измеряет максимальное использование памяти и время.
        """
        process = psutil.Process()
        gc.collect()
        mem_start = process.memory_info().rss / 1024 / 1024  # Мб

        max_mem = mem_start
        t0 = time.time()
        aug_imgs = []
        for img in imgs_resized:
            aug_img = pipeline.apply(img)
            aug_imgs.append(aug_img)
            gc.collect()
            mem_now = process.memory_info().rss / 1024 / 1024
            if mem_now > max_mem:
                max_mem = mem_now
        t1 = time.time()

        # Явно удаляем все объекты и снова собираем мусор
        del aug_imgs
        gc.collect()
        mem_end = process.memory_info().rss / 1024 / 1024

        mem_peak = max_mem - mem_start
        mem_release = mem_end - mem_start  # Покажет, сколько осталось после очистки

        return t1 - t0, mem_peak, mem_release

    for size in sizes_to_test:
        imgs_resized = [img.resize((size, size), Image.BILINEAR) for img in imgs_100]
        elapsed, mem_peak, mem_release = measure_memory_and_time(pipeline, imgs_resized)
        time_results.append((size, elapsed))
        memory_results.append((size, mem_peak))
        print(f"Размер {size}x{size}: время {elapsed:.2f} сек, пик памяти +{mem_peak:.2f} Мб, остаток +{mem_release:.2f} Мб")

    # График времени
    plt.figure(figsize=(8, 5))
    plt.plot([x[0] for x in time_results], [x[1] for x in time_results], marker='o')
    plt.title('Время применения аугментаций к 100 изображениям')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Время (сек)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'augmentation_time_vs_size.png'))
    plt.close()

    # График памяти
    plt.figure(figsize=(8, 5))
    plt.plot([x[0] for x in memory_results], [x[1] for x in memory_results], marker='o', color='red')
    plt.title('Изменение потребления памяти при аугментациях (100 изображений)')
    plt.xlabel('Размер изображения (px)')
    plt.ylabel('Изменение памяти (Мб)')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'augmentation_memory_vs_size.png'))
    plt.close()

    # Трансформации для обучения и теста
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dir = 'Torch/Homework_5/data/test'

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)  # 6 классов
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)
        train_loss = running_loss / total
        train_acc = running_corrects / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Оценка на тесте
        model.eval()
        test_loss, test_corrects, test_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                test_corrects += (preds == labels).sum().item()
                test_total += labels.size(0)
        test_losses.append(test_loss / test_total)
        test_accs.append(test_corrects / test_total)

        print(f"Epoch {epoch+1}/{num_epochs} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_losses[-1]:.4f} | Test acc: {test_accs[-1]:.4f}")

    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss по эпохам')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'finetune_loss.png'))
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy по эпохам')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'finetune_accuracy.png'))
    plt.close()
