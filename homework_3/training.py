import cv2
import numpy as np
import os

# Путь к данным
datasets = 'datasets'
sub_data = 'Svetlana'
path = os.path.join(datasets, sub_data)

# Размер изображений
(width, height) = (130, 100)

# Список данных и меток
(images, labels, names, id) = ([], [], {}, 0)

# Сканируем папку для имен
for subdir, dirs, files in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(images, labels) = [np.array(lis) for lis in [images, labels]]

# Обучение модели
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Сохранение модели
model.save('face_recognizer.yml')
