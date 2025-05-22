from ant_colony import AntColonyOptimization
from map_visualizer import MapVisualizer
import json
from config import *

def main():
    # Инициализация алгоритма
    aco = AntColonyOptimization()
    
    # Запуск алгоритма
    best_route, best_priority, total_time = aco.run()
    
    # Получение точек маршрута
    route_points = aco.get_route_points(best_route)
    
    # Вывод результатов
    print(f"Общий приоритет маршрута: {best_priority}")
    print(f"Общее время маршрута: {total_time:.1f} минут")
    print("\nМаршрут:")
    for i, point in enumerate(route_points):
        print(f"{i+1}. {point['name']} (приоритет: {point['priority']}, время посещения: {point['visit_time']} мин)")
    
    # Визуализация на карте
    visualizer = MapVisualizer()
    visualizer.add_points(aco.points)  # Добавляем все точки
    visualizer.add_route(route_points)  # Добавляем маршрут
    
    # Сохранение карты
    map_file = visualizer.save_map()
    print(f"\nКарта сохранена в файл: {map_file}")
    print("Откройте файл в браузере для просмотра маршрута.")

if __name__ == "__main__":
    main() 