import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import imageio

#2. Написать самостоятельно алгоритм для оптимального количества кластеров k-means без использования библиотек
#Рисунки выводятся на каждый шаг – сдвиг центроидов, смена точек своих кластеров.
# Сколько шагов, столько рисунков (можно в виде gif). Точки из разных кластеров разными цветами.

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100):
        self.n_clusters = n_clusters #количество кластеров
        self.max_iters = max_iters #max_iters
        self.centroids = None #центроиды кластеров

    #реализация алгоритма
    def algoritm(self, X):
        #инициализация центроидов случайным образом
        random_indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        images = []  #список для хранения изображений

        #проходимся столько раз, сколько мы указали в параметре максимальной итерации
        for iteration in range(self.max_iters):
            #присваиваем кластеры
            labels = self._assign_clusters(X)

            #сохраняем текущее состояние для визуализации
            img = self._plot(X, labels, iteration)
            images.append(img)

            #обновляем центроиды
            new_centroids = self._update_centroids(X, labels)

            #проверяем сходимость (если центроиды не изменились)
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids

        #сохранение GIF
        imageio.mimsave('kmeans_clustering.gif', images, duration=0.5)

    #вычисляет расстояния между каждой точкой и центроидами, а затем возвращает метки кластеров, основываясь на минимальных расстояниях.
    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    #обновляет координаты центроидов, вычисляя среднее значение всех точек, принадлежащих каждому кластеру.
    def _update_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    #строит график текущего состоянияси сохраняет его в виде изображения
    def _plot(self, X, labels, iteration):
        plt.figure(figsize=(8, 6))
        for i in range(self.n_clusters):
            plt.scatter(X[labels == i][:, 0], X[labels == i][:, 1], label=f'кластер {i+1}')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], s=300, c='red', marker='x', label='центроиды')
        plt.title(f'итерация {iteration + 1}')
        plt.xlabel('длина')
        plt.ylabel('ширина')
        plt.legend()
        plt.grid()

        #сохраняем текущее изображение
        plt.tight_layout()
        img_path = f'step_{iteration + 1}.png'
        plt.savefig(img_path)
        plt.close()

        return imageio.imread(img_path)


iris = load_iris()
X = iris.data[:, :2]

# Запускаем алгоритм k-means
kmeans = KMeans()
kmeans.algoritm(X)
