import numpy as np
from datetime import datetime
import os
from typing import Tuple, Dict, Any, List, Optional

from environment.utils.TimeUtils import TimeUtils
from environment.utils.CoordinateTransforms import CoordinateTransforms

class EarthModel:
	"""
	Modèle de la Terre incluant sa rotation, son champ de gravité, et d'autres propriétés physiques.
	"""
	
	def __init__(
		self,
		startTime: Optional[datetime] = None,
		gravitationalModel: str = 'J2'
	):
		"""
		Initialise le modèle de la Terre.
		
		Args:
			startTime: Heure de début de la simulation (par défaut: J2000)
			gravitationalModel: Modèle du champ gravitationnel ('point', 'J2', 'J3', 'J4', 'FULL')
		"""
		# Constantes physiques
		self.radius = 6371.0  # Rayon moyen de la Terre en km
		self.equatorialRadius = 6378.137  # Rayon équatorial de la Terre en km (WGS84)
		self.polarRadius = 6356.752  # Rayon polaire de la Terre en km (WGS84)
		self.flattening = 1/298.257223563  # Aplatissement (WGS84)
		self.gravitationalParameter = 3.986004418e14  # Paramètre gravitationnel standard en m³/s²
		self.angularVelocity = 7.2921150e-5  # Vitesse angulaire de rotation en rad/s
		
		# Paramètres du modèle gravitationnel
		self.gravitationalModel = gravitationalModel
		self.j2 = 1.08262668e-3  # Coefficient J2 (aplatissement dynamique)
		self.j3 = -2.53265648e-6  # Coefficient J3
		self.j4 = -1.61962159e-6  # Coefficient J4
		
		# Variables de temps
		self.startTime = startTime if startTime is not None else TimeUtils.J2000_EPOCH
		self.currentTime = self.startTime
		self.simulationTime = 0.0  # Temps écoulé depuis le début de la simulation (en secondes)
		
		# Temps sidéral initial
		self.gst = TimeUtils.calculateGst(self.currentTime)
		
		# Position du Soleil
		self.sunPosition = TimeUtils.calculateSunPosition(self.currentTime)
		
		# Régions terrestres (simplifiées)
		self.regions = self._initializeRegions()
	
	def reset(self, simulationTime: float = 0.0) -> None:
		"""
		Réinitialise le modèle de la Terre au temps de simulation spécifié.
		
		Args:
			simulationTime: Temps de simulation en secondes
		"""
		self.simulationTime = simulationTime
		self.currentTime = TimeUtils.simulationTimeToUtc(simulationTime, self.startTime)
		
		# Mettre à jour le temps sidéral et la position du Soleil
		self.gst = TimeUtils.calculateGst(self.currentTime)
		self.sunPosition = TimeUtils.calculateSunPosition(self.currentTime)
	
	def update(self, simulationTime: float) -> None:
		"""
		Met à jour le modèle de la Terre au temps de simulation spécifié.
		
		Args:
			simulationTime: Temps de simulation en secondes
		"""
		if simulationTime == self.simulationTime:
			return  # Aucun changement de temps, pas besoin de mise à jour
		
		self.simulationTime = simulationTime
		self.currentTime = TimeUtils.simulationTimeToUtc(simulationTime, self.startTime)
		
		# Mettre à jour le temps sidéral et la position du Soleil
		self.gst = TimeUtils.calculateGst(self.currentTime)
		self.sunPosition = TimeUtils.calculateSunPosition(self.currentTime)
	
	def calculateGravitationalAcceleration(self, position: np.ndarray) -> np.ndarray:
		"""
		Calcule l'accélération gravitationnelle à une position donnée.
		
		Args:
			position: Vecteur position [x, y, z] en km dans le référentiel ECI
			
		Returns:
			Vecteur accélération [ax, ay, az] en km/s²
		"""
		# Convertir en mètres
		position_m = position * 1000.0
		
		# Distance au centre de la Terre
		r = np.linalg.norm(position_m)
		
		if self.gravitationalModel == 'point':
			# Modèle du point massique (loi de Newton)
			direction = -position_m / r
			a = self.gravitationalParameter / (r ** 2)
			acceleration = a * direction
		
		elif self.gravitationalModel == 'J2' or self.gravitationalModel == 'J3' or \
			 self.gravitationalModel == 'J4' or self.gravitationalModel == 'FULL':
			# Coordonnées normalisées
			x, y, z = position_m
			
			# Facteurs communs
			r2 = r ** 2
			r3 = r ** 3
			r5 = r ** 5
			mu_r3 = self.gravitationalParameter / r3
			
			# Accélération du point massique
			ax = -mu_r3 * x
			ay = -mu_r3 * y
			az = -mu_r3 * z
			
			# Ajouter l'effet J2 (aplatissement)
			if self.gravitationalModel != 'point':
				# Facteur J2
				re2_r2 = (self.equatorialRadius * 1000.0) ** 2 / r2
				j2_factor = 1.5 * self.j2 * re2_r2 / r2
				
				# Terme en z²
				z2_r2 = 5 * (z ** 2) / r2
				
				# Perturbations J2
				ax += mu_r3 * j2_factor * x * (z2_r2 - 1)
				ay += mu_r3 * j2_factor * y * (z2_r2 - 1)
				az += mu_r3 * j2_factor * z * (z2_r2 - 3)
			
			# Ajouter l'effet J3 (asymétrie nord-sud)
			if self.gravitationalModel in ['J3', 'J4', 'FULL']:
				# Facteur J3
				re3_r3 = (self.equatorialRadius * 1000.0) ** 3 / r3
				j3_factor = 2.5 * self.j3 * re3_r3 / r2
				
				# Termes en z
				z_r = z / r
				z3_r3 = 7 * (z ** 3) / r3
				
				# Perturbations J3
				ax += mu_r3 * j3_factor * x * z * (3 - z3_r3)
				ay += mu_r3 * j3_factor * y * z * (3 - z3_r3)
				az += mu_r3 * j3_factor * ((6 * z2_r2 - 3) * z_r - z3_r3 * z_r)
			
			# Ajouter l'effet J4
			if self.gravitationalModel in ['J4', 'FULL']:
				# Facteur J4
				re4_r4 = (self.equatorialRadius * 1000.0) ** 4 / r2**2
				j4_factor = self.j4 * re4_r4 / r2
				
				# Termes en z
				z4_r4 = (z ** 4) / r2**2
				
				# Perturbations J4
				ax += mu_r3 * j4_factor * x * (35 * z4_r4 - 30 * z2_r2 + 3)
				ay += mu_r3 * j4_factor * y * (35 * z4_r4 - 30 * z2_r2 + 3)
				az += mu_r3 * j4_factor * z * (35 * z4_r4 - 30 * z2_r2 + 5)
			
			acceleration = np.array([ax, ay, az])
		
		else:
			raise ValueError(f"Modèle gravitationnel non pris en charge: {self.gravitationalModel}")
		
		# Convertir en km/s²
		return acceleration / 1000.0
	
	def cartesianToGeodetic(self, position: np.ndarray) -> Tuple[float, float, float]:
		"""
		Convertit des coordonnées cartésiennes ECI en coordonnées géodétiques.
		
		Args:
			position: Vecteur position [x, y, z] en km dans le référentiel ECI
			
		Returns:
			Tuple de (latitude en degrés, longitude en degrés, altitude en km)
		"""
		# Convertir d'abord en ECEF
		position_ecef, _ = CoordinateTransforms.eciToEcef(position, np.zeros(3), self.gst)
		
		# Puis convertir ECEF en géodétique
		return CoordinateTransforms.ecefToGeodetic(position_ecef)
	
	def geodeticToCartesian(self, latitude: float, longitude: float, altitude: float) -> np.ndarray:
		"""
		Convertit des coordonnées géodétiques en coordonnées cartésiennes ECI.
		
		Args:
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			altitude: Altitude en km au-dessus de l'ellipsoïde
			
		Returns:
			Vecteur position [x, y, z] en km dans le référentiel ECI
		"""
		# Convertir d'abord en ECEF
		position_ecef = CoordinateTransforms.geodeticToEcef(latitude, longitude, altitude)
		
		# Puis convertir ECEF en ECI
		position_eci, _ = CoordinateTransforms.ecefToEci(position_ecef, np.zeros(3), self.gst)
		
		return position_eci
	
	def isInDaylight(self, position: np.ndarray) -> bool:
		"""
		Détermine si un point est éclairé par le Soleil ou dans l'ombre de la Terre.
		
		Args:
			position: Vecteur position [x, y, z] en km dans le référentiel ECI
			
		Returns:
			True si le point est éclairé, False s'il est dans l'ombre
		"""
		# Vecteur du point au Soleil
		sun_direction = self.sunPosition - position
		sun_distance = np.linalg.norm(sun_direction)
		sun_direction = sun_direction / sun_distance
		
		# Produit scalaire avec la position (normalisée)
		pos_norm = np.linalg.norm(position)
		if pos_norm < 1e-6:  # Éviter la division par zéro
			return True
			
		pos_direction = position / pos_norm
		
		# Angle entre le vecteur position et la direction du Soleil
		cos_angle = np.dot(pos_direction, sun_direction)
		
		# Si l'angle est > 90°, le point est du côté de la Terre opposé au Soleil
		if cos_angle < 0:
			# Calculer la distance minimale de la ligne point-Soleil au centre de la Terre
			# d = |position × sun_direction|
			distance_to_center = np.linalg.norm(np.cross(position, sun_direction))
			
			# Si cette distance est inférieure au rayon de la Terre, le point est dans l'ombre
			return distance_to_center > self.radius
		
		# Le point est du même côté que le Soleil
		return True
	
	def calculateSolarIrradiance(self, position: np.ndarray, normal: np.ndarray) -> float:
		"""
		Calcule l'irradiance solaire sur une surface avec une orientation donnée.
		
		Args:
			position: Vecteur position [x, y, z] en km dans le référentiel ECI
			normal: Vecteur normal à la surface
			
		Returns:
			Irradiance en W/m²
		"""
		# Constante solaire (irradiance au niveau de l'orbite terrestre)
		solar_constant = 1361.0  # W/m²
		
		# Vérifier si le point est dans l'ombre
		if not self.isInDaylight(position):
			return 0.0
		
		# Direction du Soleil
		sun_direction = self.sunPosition - position
		sun_direction = sun_direction / np.linalg.norm(sun_direction)
		
		# Normaliser le vecteur normal
		normal = normal / np.linalg.norm(normal)
		
		# Calculer le cosinus de l'angle d'incidence
		cos_incidence = np.dot(normal, sun_direction)
		
		# Si le cosinus est négatif, le Soleil est derrière la surface
		if cos_incidence <= 0:
			return 0.0
		
		# Calculer l'irradiance
		irradiance = solar_constant * cos_incidence
		
		return irradiance
	
	def getState(self) -> Dict[str, Any]:
		"""
		Retourne l'état actuel du modèle pour sauvegarde/chargement.
		
		Returns:
			Dictionnaire de l'état
		"""
		return {
			"simulation_time": self.simulationTime,
			"current_time": self.currentTime.isoformat(),
			"gst": self.gst,
			"sun_position": self.sunPosition.tolist()
		}
	
	def setState(self, state: Dict[str, Any]) -> None:
		"""
		Restaure l'état du modèle.
		
		Args:
			state: État à restaurer
		"""
		self.simulationTime = state["simulation_time"]
		self.currentTime = datetime.fromisoformat(state["current_time"])
		self.gst = state["gst"]
		self.sunPosition = np.array(state["sun_position"])
	
	def _initializeRegions(self) -> List[Dict[str, Any]]:
		"""
		Initialise les régions terrestres simplifiées pour la simulation.
		
		Returns:
			Liste des régions
		"""
		# Définir des régions simplifiées avec leur population approximative
		regions = [
			{"name": "Amérique du Nord", "center": (40.0, -100.0), "population": 579e6, "area": 24.71e6},
			{"name": "Amérique du Sud", "center": (-20.0, -60.0), "population": 430e6, "area": 17.84e6},
			{"name": "Europe", "center": (50.0, 10.0), "population": 747e6, "area": 10.18e6},
			{"name": "Afrique", "center": (0.0, 20.0), "population": 1340e6, "area": 30.37e6},
			{"name": "Asie", "center": (30.0, 90.0), "population": 4641e6, "area": 44.58e6},
			{"name": "Océanie", "center": (-25.0, 135.0), "population": 42e6, "area": 8.53e6},
			{"name": "Arctique", "center": (80.0, 0.0), "population": 4e6, "area": 14.05e6},
			{"name": "Antarctique", "center": (-80.0, 0.0), "population": 0, "area": 14.2e6},
			{"name": "Océan Pacifique", "center": (0.0, -150.0), "population": 0, "area": 165.25e6},
			{"name": "Océan Atlantique", "center": (0.0, -30.0), "population": 0, "area": 106.46e6},
			{"name": "Océan Indien", "center": (-10.0, 70.0), "population": 0, "area": 70.56e6}
		]
		
		return regions
	
	def getPopulationDensity(self, latitude: float, longitude: float) -> float:
		"""
		Renvoie une densité de population approximative pour un emplacement donné.
		
		Args:
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			
		Returns:
			Densité de population (personnes/km²)
		"""
		# Simplification: utiliser les régions prédéfinies
		min_distance = float('inf')
		closest_region = None
		
		for region in self.regions:
			region_lat, region_lon = region["center"]
			
			# Distance approximative (pondérée pour prendre en compte que les degrés de longitude
			# sont plus proches à haute latitude)
			lat_weight = np.cos(np.radians((latitude + region_lat) / 2))
			lon_diff = min(abs(longitude - region_lon), 360 - abs(longitude - region_lon))
			lat_diff = abs(latitude - region_lat)
			distance = np.sqrt((lat_diff ** 2) + (lon_diff * lat_weight) ** 2)
			
			if distance < min_distance:
				min_distance = distance
				closest_region = region
		
		# Densité de population (personnes/km²)
		density = closest_region["population"] / closest_region["area"] if closest_region else 0.0
		
		return density