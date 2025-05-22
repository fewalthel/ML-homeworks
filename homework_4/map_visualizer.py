import folium
from folium import plugins
from config import *

class MapVisualizer:
    def __init__(self):
        self.map = folium.Map(
            location=MAP_CENTER,
            zoom_start=MAP_ZOOM,
            tiles='OpenStreetMap'
        )

    def add_points(self, points):
        """Добавление точек на карту"""
        for point in points:
            folium.Marker(
                location=point['coordinates'],
                popup=f"{point['name']}<br>Приоритет: {point['priority']}<br>Время посещения: {point['visit_time']} мин",
                icon=folium.Icon(color=POINT_COLOR, icon='info-sign')
            ).add_to(self.map)

    def add_route(self, points):
        """Добавление маршрута на карту"""
        coordinates = [point['coordinates'] for point in points]
        
        # Добавление линии маршрута
        folium.PolyLine(
            coordinates,
            weight=3,
            color=ROUTE_COLOR,
            opacity=0.8
        ).add_to(self.map)
        
        # Добавление номеров точек на маршруте
        for i, point in enumerate(points):
            folium.CircleMarker(
                location=point['coordinates'],
                radius=8,
                popup=f"Точка {i+1}: {point['name']}",
                color=ROUTE_COLOR,
                fill=True
            ).add_to(self.map)

    def save_map(self, filename='route_map.html'):
        """Сохранение карты в HTML файл"""
        self.map.save(filename)
        return filename 