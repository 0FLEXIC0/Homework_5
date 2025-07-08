## К сожалению папки data и results оказались слишком большими, поэтому мне пришлось разместить их на Гугл диске

https://drive.google.com/file/d/1blRq75kOd66QlVqzm4Natdyx5PZFGTjm/view?usp=sharing

# Задание 1: Стандартные аугментации torchvision (15 баллов)

1) Создайте пайплайн стандартных аугментаций torchvision (например, RandomHorizontalFlip, RandomCrop, ColorJitter, RandomRotation, RandomGrayscale).
2) Примените аугментации к 5 изображениям из разных классов (папка train).
3) Визуализируйте:
    - Оригинал
    - Результат применения каждой аугментации отдельно
    - Результат применения всех аугментаций вместе

**Гароу**
![](data/train/Гароу/02b8c85727aa4d9978ae2cb518affb2a.jpg)

**Гароу_RandomHorizontalFlip**
![](results/Гароу_RandomHorizontalFlip.png)

**Гароу_RandomCrop**
![](results/Гароу_RandomCrop.png)

**Гароу_ColorJitter**
![](results/Гароу_ColorJitter.png)

**Гароу_RandomRotation**
![](results/Гароу_RandomRotation.png)

**Гароу_RandomGrayscale**
![](results/Гароу_RandomGrayscale.png)

**Гароу_Combined**
![](results/Гароу_Combined.png)

**Генос**
![](data/train/Генос/08a7ce2efd7b8b95a254833da966265f.jpg)

**Генос_RandomHorizontalFlip**
![](results/Генос_RandomHorizontalFlip.png)

**Генос_RandomCrop**
![](results/Генос_RandomCrop.png)

**Генос_ColorJitter**
![](results/Генос_ColorJitter.png)

**Генос_RandomRotation**
![](results/Генос_RandomRotation.png)

**Генос_RandomGrayscale**
![](results/Генос_RandomGrayscale.png)

**Генос_Combined**
![](results/Генос_Combined.png)

**Сайтама**
![](data/train/Сайтама/05eaf704e653c1cedad80dedc0e30824.jpg)

**Сайтама_RandomHorizontalFlip**
![](results/Сайтама_RandomHorizontalFlip.png)

**Сайтама_RandomCrop**
![](results/Сайтама_RandomCrop.png)

**Сайтама_ColorJitter**
![](results/Сайтама_ColorJitter.png)

**Сайтама_RandomRotation**
![](results/Сайтама_RandomRotation.png)

**Сайтама_RandomGrayscale**
![](results/Сайтама_RandomGrayscale.png)

**Сайтама_Combined**
![](results/Сайтама_Combined.png)

**Соник**
![](data/train/Соник/00badad3b4e440bc1264bf25f0ea65f2.jpg)

**Соник_RandomHorizontalFlip**
![](results/Соник_RandomHorizontalFlip.png)

**Соник_RandomCrop**
![](results/Соник_RandomCrop.png)

**Соник_ColorJitter**
![](results/Соник_ColorJitter.png)

**Соник_RandomRotation**
![](results/Соник_RandomRotation.png)

**Соник_RandomGrayscale**
![](results/Соник_RandomGrayscale.png)

**Соник_Combined**
![](results/Соник_Combined.png)

**Татсумаки**
![](data/train/Татсумаки/06c27614479e71583feac6d0e98f3073.jpg)

**Татсумаки_RandomHorizontalFlip**
![](results/Татсумаки_RandomHorizontalFlip.png)

**Татсумаки_RandomCrop**
![](results/Татсумаки_RandomCrop.png)

**Татсумаки_ColorJitter**
![](results/Татсумаки_ColorJitter.png)

**Татсумаки_RandomRotation**
![](results/Татсумаки_RandomRotation.png)

**Татсумаки_RandomGrayscale**
![](results/Татсумаки_RandomGrayscale.png)

**Татсумаки_Combined**
![](results/Татсумаки_Combined.png)

**Фубуки**
![](data/train/Фубуки/04e50b5f0b17d843de6081eb59150452.jpg)

**Фубуки_RandomHorizontalFlip**
![](results/Фубуки_RandomHorizontalFlip.png)

**Фубуки_RandomCrop**
![](results/Фубуки_RandomCrop.png)

**Фубуки_ColorJitter**
![](results/Фубуки_ColorJitter.png)

**Фубуки_RandomRotation**
![](results/Фубуки_RandomRotation.png)

**Фубуки_RandomGrayscale**
![](results/Фубуки_RandomGrayscale.png)

**Фубуки_Combined**
![](results/Фубуки_Combined.png)

# Задание 2: Кастомные аугментации (20 баллов)

1) Реализуйте минимум 3 кастомные аугментации (например, случайное размытие, случайная перспектива, случайная яркость/контрастность).
2) Примените их к изображениям из train.
3) Сравните визуально с готовыми аугментациями из extra_augs.py.

К сожалению в extra_augs.py нет ни случайного размытия, ни случайной перспективы, ни случайной яркости/контрастности

**Гароу_RandomGaussianBlur**
![](results/Гароу_RandomGaussianBlur.png)

**Гароу_RandomPerspective**
![](results/Гароу_RandomPerspective.png)

**Гароу_RandomBrightnessContrast**
![](results/Гароу_RandomBrightnessContrast.png)

**Генос_RandomGaussianBlur**
![](results/Генос_RandomGaussianBlur.png)

**Генос_RandomPerspective**
![](results/Генос_RandomPerspective.png)

**Генос_RandomBrightnessContrast**
![](results/Генос_RandomBrightnessContrast.png)

**Сайтама_RandomGaussianBlur**
![](results/Сайтама_RandomGaussianBlur.png)

**Сайтама_RandomPerspective**
![](results/Сайтама_RandomPerspective.png)

**Сайтама_RandomBrightnessContrast**
![](results/Сайтама_RandomBrightnessContrast.png)

**Соник_RandomGaussianBlur**
![](results/Соник_RandomGaussianBlur.png)

**Соник_RandomPerspective**
![](results/Соник_RandomPerspective.png)

**Соник_RandomBrightnessContrast**
![](results/Соник_RandomBrightnessContrast.png)

**Татсумаки_RandomGaussianBlur**
![](results/Татсумаки_RandomGaussianBlur.png)

**Татсумаки_RandomPerspective**
![](results/Татсумаки_RandomPerspective.png)

**Татсумаки_RandomBrightnessContrast**
![](results/Татсумаки_RandomBrightnessContrast.png)

**Фубуки_RandomGaussianBlur**
![](results/Фубуки_RandomGaussianBlur.png)

**Фубуки_RandomPerspective**
![](results/Фубуки_RandomPerspective.png)

**Фубуки_RandomBrightnessContrast**
![](results/Фубуки_RandomBrightnessContrast.png)

# Задание 3: Анализ датасета (10 баллов)

1) Подсчитайте количество изображений в каждом классе.
```
Класс     Кол-во
Гароу       30
Сайтама     30
Соник       30
Татсумаки   30
Фубуки      30
```
2) Найдите минимальный, максимальный и средний размеры изображений.
```
Класс      Мин. размер      Макс. размер     Средний размер
Гароу       (246, 246)      (735,  889)     (538.43, 514.63)
Генос       (270, 363)      (720, 1070)     (550.17, 673.90)
Сайтама     (375, 366)      (736, 1308)     (559.33, 669.27)
Соник       (210, 240)      (604, 1076)     (525.87, 609.93)
Татсумаки   (267, 267)      (736, 1308)     (527.20, 635.83)
Фубуки      (286, 353)      (736, 1104)     (532.37, 637.77)
```
3) Визуализируйте распределение размеров и гистограмму по классам.

**Количество изображений в каждом классе**
![](results/image_counts_histogram.png)

**Минимальные, максимальные и средние размеры по классам**
![](results/image_sizes_distribution.png)

**Распределение ширины и высоты всех изображений**
![](results/all_image_sizes_hist.png)

# Задание 4: Pipeline аугментаций (20 баллов)

1) Реализуйте класс AugmentationPipeline с методами:
    - add_augmentation(name, aug)
    - remove_augmentation(name)
    - apply(image)
    - get_augmentations()
2) Создайте несколько конфигураций (light, medium, heavy).
3) Примените каждую конфигурацию к train и сохраните результаты.

Класс создан, конфигурации созданы, результаты помещены в отдельные папки в папке results.

# Задание 5: Эксперимент с размерами (10 баллов)

1) Проведите эксперимент с разными размерами изображений (например, 64x64, 128x128, 224x224, 512x512).
2) Для каждого размера измерьте время загрузки и применения аугментаций к 100 изображениям, а также потребление памяти.
```
Размер 64x64:   время 8.53 сек, пик памяти 0.25 Мб,  остаток 0.25 Мб
Размер 128x128: время 8.70 сек, пик памяти 4.08 Мб,  остаток 4.08 Мб
Размер 256x256: время 8.98 сек, пик памяти 8.46 Мб,  остаток 10.55 Мб
Размер 512x512: время 9.03 сек, пик памяти 15.62 Мб, остаток 2.60 Мб
```
3) Постройте графики зависимости времени и памяти от размера.

**Зависимость времени от размера**
![](results/augmentation_time_vs_size.png)

**Зависимость памяти от размера**
![](results/augmentation_memory_vs_size.png)

# Задание 6: Дообучение предобученных моделей (25 баллов)

1) Возьмите одну из предобученных моделей torchvision (например, resnet18, efficientnet_b0, mobilenet_v3_small).
2) Замените последний слой на количество классов вашего датасета.
3) Дообучите модель на train, проверьте качество на val.
4) Визуализируйте процесс обучения (loss/accuracy).

**Точность**
![](results/finetune_accuracy.png)

**Квадратичное отклонение**
![](results/finetune_loss.png)