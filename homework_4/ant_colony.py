import numpy as np
import json
from geopy.distance import geodesic
from config import *

class AntColonyOptimization:
    def __init__(self, points_file='points.json'):
        # Загрузка точек из файла
        with open(points_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.points = data['points']
        
        self.n_points = len(self.points)
        self.distances = self._calculate_distances()
        self.pheromone = np.ones((self.n_points, self.n_points)) / self.n_points
        self.priorities = np.array([p['priority'] for p in self.points])
        self.visit_times = np.array([p['visit_time'] for p in self.points])
        
    def _calculate_distances(self):
        """Вычисление матрицы расстояний между точками"""
        distances = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            for j in range(self.n_points):
                if i != j:
                    point1 = tuple(self.points[i]['coordinates'])
                    point2 = tuple(self.points[j]['coordinates'])
                    distances[i][j] = geodesic(point1, point2).kilometers
        return distances

    def _calculate_time(self, route):
        """Вычисление общего времени маршрута"""
        total_time = 0
        for i in range(len(route) - 1):
            # Время на перемещение между точками
            distance = self.distances[route[i]][route[i + 1]]
            travel_time = (distance / AVERAGE_SPEED_KMH) * 60  # в минутах
            total_time += travel_time
            # Время на посещение точки
            total_time += self.visit_times[route[i + 1]]
        return total_time

    def _calculate_route_priority(self, route):
        """Вычисление общего приоритета маршрута"""
        return sum(self.priorities[point] for point in route)

    def _select_next_point(self, ant_path, unvisited):
        """Выбор следующей точки для муравья"""
        current = ant_path[-1]
        pheromone = np.array([self.pheromone[current][j] for j in unvisited])
        distance = np.array([self.distances[current][j] for j in unvisited])
        priority = np.array([self.priorities[j] for j in unvisited])
        
        # Эвристическая информация (приоритет / расстояние)
        heuristic = priority / (distance + 1e-10)
        
        # Вероятности перехода
        probabilities = (pheromone ** ALPHA) * (heuristic ** BETA)
        probabilities = probabilities / probabilities.sum()
        
        # Выбор следующей точки
        next_point_idx = np.random.choice(len(unvisited), p=probabilities)
        return unvisited[next_point_idx]

    def run(self):
        """Запуск муравьиного алгоритма"""
        best_route = None
        best_priority = -float('inf')
        
        for iteration in range(ITERATIONS):
            ant_routes = []
            ant_priorities = []
            
            # Построение маршрутов для каждого муравья
            for ant in range(ANT_COUNT):
                current_route = [0]  # Начинаем с первой точки
                unvisited = list(range(1, self.n_points))
                
                while unvisited:
                    next_point = self._select_next_point(current_route, unvisited)
                    current_route.append(next_point)
                    unvisited.remove(next_point)
                    
                    # Проверка ограничения по времени
                    if self._calculate_time(current_route) > MAX_TIME_MINUTES:
                        current_route.pop()  # Удаляем последнюю точку
                        break
                
                route_priority = self._calculate_route_priority(current_route)
                ant_routes.append(current_route)
                ant_priorities.append(route_priority)
                
                # Обновление лучшего маршрута
                if route_priority > best_priority:
                    best_route = current_route.copy()
                    best_priority = route_priority
            
            # Испарение феромона
            self.pheromone *= (1 - RHO)
            
            # Обновление феромона
            for route, priority in zip(ant_routes, ant_priorities):
                for i in range(len(route) - 1):
                    self.pheromone[route[i]][route[i + 1]] += Q * priority
                    self.pheromone[route[i + 1]][route[i]] = self.pheromone[route[i]][route[i + 1]]
        
        return best_route, best_priority, self._calculate_time(best_route)

    def get_route_points(self, route):
        """Получение координат точек маршрута"""
        return [self.points[i] for i in route] 