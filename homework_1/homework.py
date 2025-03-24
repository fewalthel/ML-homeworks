import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

#1. Найти оптимальное количество кластеров при помощи готовых библиотек
def main():
    iris = load_iris()
    X = iris.data

    inertias = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1, 15), inertias)
    plt.title('Поиск оптимального количества кластеров')
    plt.xlabel('Количество кластеров')
    plt.show()

if __name__ == '__main__':
    main()