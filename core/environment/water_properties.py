# core/environment/water_properties.py
from typing import Dict, List, Tuple, Any
import numpy as np
import uuid
import math

from utils.serialization import Serializable


class WaterProperties(Serializable):
	"""
	Classe représentant les propriétés physiques et chimiques de l'eau.
	Modélise la température, la salinité, les courants, etc.
	"""
	
	def __init__(
		self,
		worldSize: Tuple[float, float, float],
		resolution: Tuple[int, int, int],
		defaultTemperature: float = 15.0,
		defaultSalinity: float = 35.0,
		defaultLightLevel: float = 0.8,
		seed: int = None
	) -> None:
		"""
		Initialise les propriétés de l'eau.
		
		Args:
			worldSize: Dimensions du monde marin (x, y, z)
			resolution: Résolution de la grille pour les calculs (nx, ny, nz)
			defaultTemperature: Température par défaut de l'eau en degrés Celsius
			defaultSalinity: Salinité par défaut de l'eau en PSU
			defaultLightLevel: Niveau de lumière par défaut à la surface
			seed: Graine pour le générateur de nombres aléatoires
		"""
		self.id: str = str(uuid.uuid4())
		self.worldSize: np.ndarray = np.array(worldSize, dtype=np.float32)
		self.resolution: np.ndarray = np.array(resolution, dtype=np.int32)
		self.defaultTemperature: float = defaultTemperature
		self.defaultSalinity: float = defaultSalinity
		self.defaultLightLevel: float = defaultLightLevel
		
		# Initialiser le générateur de nombres aléatoires
		self.rng: np.random.Generator = np.random.default_rng(seed)
		
		# Paramètres de simulation
		self.temperatureGradient: float = 0.01  # °C par mètre de profondeur
		self.salinityGradient: float = 0.001  # PSU par mètre de profondeur
		self.lightAttenuationCoefficient: float = 0.1  # Coefficient d'atténuation de la lumière
		self.currentSpeed: float = 0.2  # Vitesse de base des courants en m/s
		self.currentVariability: float = 0.1  # Variabilité des courants
		self.currentChangeRate: float = 0.05  # Taux de changement des courants
		
		# Paramètres cycliques
		self.dailyTemperatureVariation: float = 2.0  # Variation de température quotidienne en °C
		self.seasonalTemperatureVariation: float = 5.0  # Variation de température saisonnière en °C
		
		# Initialisation des grilles (pour l'optimisation)
		self.temperatureGrid = None
		self.salinityGrid = None
		self.lightGrid = None
		self.currentGrid = None
		self.lastUpdateTime: float = 0.0
		
		# Initialiser les grilles
		self._initializeGrids()
	
	def _initializeGrids(self) -> None:
		"""
		Initialise les grilles de propriétés pour l'optimisation des calculs.
		"""
		# Créer des grilles 3D pour stocker les valeurs précalculées
		# Ces grilles sont utilisées pour éviter de recalculer les propriétés à chaque requête
		
		# Grille de température (°C)
		self.temperatureGrid = np.ones(tuple(self.resolution), dtype=np.float32) * self.defaultTemperature
		
		# Appliquer un gradient de température vertical (plus froid en profondeur)
		for y in range(self.resolution[1]):
			depth = y / self.resolution[1] * self.worldSize[1]
			self.temperatureGrid[:, y, :] = self.defaultTemperature - depth * self.temperatureGradient
		
		# Ajouter des variations spatiales aléatoires
		self.temperatureGrid += self.rng.normal(0, 1.0, self.temperatureGrid.shape) * 0.5
		
		# Grille de salinité (PSU)
		self.salinityGrid = np.ones(tuple(self.resolution), dtype=np.float32) * self.defaultSalinity
		
		# Appliquer un gradient de salinité vertical (plus salé en profondeur)
		for y in range(self.resolution[1]):
			depth = y / self.resolution[1] * self.worldSize[1]
			self.salinityGrid[:, y, :] = self.defaultSalinity + depth * self.salinityGradient
		
		# Ajouter des variations spatiales aléatoires
		self.salinityGrid += self.rng.normal(0, 1.0, self.salinityGrid.shape) * 0.2
		
		# Grille de lumière (0-1)
		self.lightGrid = np.zeros(tuple(self.resolution), dtype=np.float32)
		
		# Appliquer une atténuation exponentielle de la lumière avec la profondeur
		for y in range(self.resolution[1]):
			depth = y / self.resolution[1] * self.worldSize[1]
			self.lightGrid[:, y, :] = self.defaultLightLevel * np.exp(-self.lightAttenuationCoefficient * depth)
		
		# Grille de courants (vecteurs 3D)
		# Format: [direction_x, direction_y, direction_z, magnitude]
		self.currentGrid = np.zeros((*tuple(self.resolution), 4), dtype=np.float32)
		
		# Initialiser les courants avec un modèle simple
		for x in range(self.resolution[0]):
			for y in range(self.resolution[1]):
				for z in range(self.resolution[2]):
					# Position normalisée
					nx = x / self.resolution[0]
					ny = y / self.resolution[1]
					nz = z / self.resolution[2]
					
					# Courant de base horizontal (principalement est-ouest)
					dirX = np.sin(nx * 2 * np.pi + nz * 3 * np.pi)
					dirY = 0.1 * np.sin(ny * 4 * np.pi)  # Légère composante verticale
					dirZ = np.cos(nx * 2 * np.pi + nz * 3 * np.pi)
					
					# Normaliser la direction
					norm = np.sqrt(dirX**2 + dirY**2 + dirZ**2)
					if norm > 0:
						dirX /= norm
						dirY /= norm
						dirZ /= norm
					
					# Magnitude diminue avec la profondeur
					magnitude = self.currentSpeed * (1.0 - 0.7 * ny)
					
					# Stocker dans la grille
					self.currentGrid[x, y, z] = [dirX, dirY, dirZ, magnitude]
	
	def update(self, deltaTime: float, currentTime: float) -> None:
		"""
		Met à jour les propriétés de l'eau en fonction du temps.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
			currentTime: Temps actuel de la simulation
		"""
		# Mettre à jour seulement si assez de temps s'est écoulé (optimisation)
		if currentTime - self.lastUpdateTime < 0.1:
			return
			
		self.lastUpdateTime = currentTime
		
		# Calculer les facteurs cycliques
		dayFactor = np.sin(currentTime * 2 * np.pi / 24.0)  # Cycle jour/nuit
		yearFactor = np.sin(currentTime * 2 * np.pi / (365.0 * 24.0))  # Cycle saisonnier
		
		# Mettre à jour la température avec les variations cycliques
		tempVariation = dayFactor * self.dailyTemperatureVariation + yearFactor * self.seasonalTemperatureVariation
		
		for y in range(self.resolution[1]):
			depth = y / self.resolution[1] * self.worldSize[1]
			baseTemp = self.defaultTemperature - depth * self.temperatureGradient
			
			# La variation diminue avec la profondeur
			depthFactor = max(0.0, 1.0 - depth / (0.2 * self.worldSize[1]))
			
			self.temperatureGrid[:, y, :] = baseTemp + tempVariation * depthFactor
		
		# Mettre à jour la lumière en fonction de l'heure du jour
		dayTime = currentTime % 24.0
		if 6 <= dayTime < 18:  # Jour
			dayProgress = (dayTime - 6) / 12.0  # 0 à 1 pendant la journée
			lightFactor = np.sin(dayProgress * np.pi)  # Pic à midi
			surfaceLight = 0.2 + 0.8 * lightFactor  # Entre 0.2 (aube/crépuscule) et 1.0 (midi)
		else:  # Nuit
			surfaceLight = 0.05  # Niveau de lumière nocturne (lune, etc.)
		
		# Appliquer le niveau de lumière à la surface et l'atténuation avec la profondeur
		for y in range(self.resolution[1]):
			depth = y / self.resolution[1] * self.worldSize[1]
			self.lightGrid[:, y, :] = surfaceLight * np.exp(-self.lightAttenuationCoefficient * depth)
		
		# Faire évoluer légèrement les courants
		for x in range(self.resolution[0]):
			for y in range(self.resolution[1]):
				for z in range(self.resolution[2]):
					# Récupérer le courant actuel
					dirX, dirY, dirZ, magnitude = self.currentGrid[x, y, z]
					
					# Ajouter une légère perturbation à la direction
					perturbX = self.rng.normal(0, self.currentVariability)
					perturbY = self.rng.normal(0, self.currentVariability * 0.5)  # Moins de variation verticale
					perturbZ = self.rng.normal(0, self.currentVariability)
					
					# Appliquer la perturbation
					dirX += perturbX * self.currentChangeRate * deltaTime
					dirY += perturbY * self.currentChangeRate * deltaTime
					dirZ += perturbZ * self.currentChangeRate * deltaTime
					
					# Normaliser la direction
					norm = np.sqrt(dirX**2 + dirY**2 + dirZ**2)
					if norm > 0:
						dirX /= norm
						dirY /= norm
						dirZ /= norm
					
					# Faire varier légèrement la magnitude
					magnitudeChange = self.rng.normal(0, 0.1)
					magnitude += magnitudeChange * self.currentChangeRate * deltaTime
					magnitude = max(0.01, min(self.currentSpeed * 1.5, magnitude))
					
					# Stocker le nouveau courant
					self.currentGrid[x, y, z] = [dirX, dirY, dirZ, magnitude]
	
	def getTemperatureAt(self, position: np.ndarray, currentTime: float = None) -> float:
		"""
		Retourne la température à une position donnée.
		
		Args:
			position: Position (x, y, z) dans le monde
			currentTime: Temps actuel (pour les variations temporelles)
			
		Returns:
			Température en degrés Celsius
		"""
		# Convertir la position en indices de grille
		ix, iy, iz = self._positionToGridIndices(position)
		
		# Récupérer la température de base depuis la grille
		temperature = self.temperatureGrid[ix, iy, iz]
		
		# Ajouter des variations en fonction du temps si fourni
		if currentTime is not None:
			dayFactor = np.sin(currentTime * 2 * np.pi / 24.0)
			yearFactor = np.sin(currentTime * 2 * np.pi / (365.0 * 24.0))
			
			# Les variations diminuent avec la profondeur
			depthFactor = max(0.0, 1.0 - position[1] / (0.2 * self.worldSize[1]))
			
			# Appliquer les variations cycliques
			temperature += (dayFactor * self.dailyTemperatureVariation + 
						   yearFactor * self.seasonalTemperatureVariation) * depthFactor
		
		return temperature
	
	def getSalinityAt(self, position: np.ndarray, currentTime: float = None) -> float:
		"""
		Retourne la salinité à une position donnée.
		
		Args:
			position: Position (x, y, z) dans le monde
			currentTime: Temps actuel (pour les variations temporelles)
			
		Returns:
			Salinité en PSU
		"""
		# Convertir la position en indices de grille
		ix, iy, iz = self._positionToGridIndices(position)
		
		# Récupérer la salinité depuis la grille
		salinity = self.salinityGrid[ix, iy, iz]
		
		return salinity
	
	def getLightLevelAt(self, position: np.ndarray, currentTime: float = None) -> float:
		"""
		Retourne le niveau de lumière à une position donnée.
		
		Args:
			position: Position (x, y, z) dans le monde
			currentTime: Temps actuel (pour les variations temporelles)
			
		Returns:
			Niveau de lumière entre 0.0 (obscurité) et 1.0 (pleine lumière)
		"""
		# Convertir la position en indices de grille
		ix, iy, iz = self._positionToGridIndices(position)
		
		# Récupérer le niveau de lumière de base depuis la grille
		lightLevel = self.lightGrid[ix, iy, iz]
		
		# Ajouter des variations en fonction de l'heure du jour si le temps est fourni
		if currentTime is not None:
			dayTime = currentTime % 24.0
			if 6 <= dayTime < 18:  # Jour
				dayProgress = (dayTime - 6) / 12.0  # 0 à 1 pendant la journée
				lightFactor = np.sin(dayProgress * np.pi)  # Pic à midi
				surfaceLight = 0.2 + 0.8 * lightFactor
			else:  # Nuit
				surfaceLight = 0.05  # Niveau de lumière nocturne (lune, etc.)
				
			# Atténuation exponentielle avec la profondeur
			depth = position[1]
			lightLevel = surfaceLight * np.exp(-self.lightAttenuationCoefficient * depth)
		
		return lightLevel
	
	def getCurrentAt(self, position: np.ndarray, currentTime: float = None) -> np.ndarray:
		"""
		Retourne le vecteur de courant à une position donnée.
		
		Args:
			position: Position (x, y, z) dans le monde
			currentTime: Temps actuel (pour les variations temporelles)
			
		Returns:
			Vecteur de courant [vx, vy, vz] en m/s
		"""
		# Convertir la position en indices de grille
		ix, iy, iz = self._positionToGridIndices(position)
		
		# Récupérer les composantes du courant depuis la grille
		dirX, dirY, dirZ, magnitude = self.currentGrid[ix, iy, iz]
		
		# Créer le vecteur de courant
		current = np.array([dirX * magnitude, dirY * magnitude, dirZ * magnitude], dtype=np.float32)
		
		return current
	
	def _positionToGridIndices(self, position: np.ndarray) -> Tuple[int, int, int]:
		"""
		Convertit une position en coordonnées mondiales en indices de grille.
		
		Args:
			position: Position (x, y, z) dans le monde
			
		Returns:
			Tuple (ix, iy, iz) des indices de grille correspondants
		"""
		ix = int(position[0] / self.worldSize[0] * self.resolution[0])
		iy = int(position[1] / self.worldSize[1] * self.resolution[1])
		iz = int(position[2] / self.worldSize[2] * self.resolution[2])
		
		# Limiter aux indices valides
		ix = max(0, min(ix, self.resolution[0] - 1))
		iy = max(0, min(iy, self.resolution[1] - 1))
		iz = max(0, min(iz, self.resolution[2] - 1))
		
		return ix, iy, iz
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet WaterProperties en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état des propriétés de l'eau
		"""
		return {
			"id": self.id,
			"worldSize": self.worldSize.tolist(),
			"resolution": self.resolution.tolist(),
			"defaultTemperature": self.defaultTemperature,
			"defaultSalinity": self.defaultSalinity,
			"defaultLightLevel": self.defaultLightLevel,
			"temperatureGradient": self.temperatureGradient,
			"salinityGradient": self.salinityGradient,
			"lightAttenuationCoefficient": self.lightAttenuationCoefficient,
			"currentSpeed": self.currentSpeed,
			"currentVariability": self.currentVariability,
			"currentChangeRate": self.currentChangeRate,
			"dailyTemperatureVariation": self.dailyTemperatureVariation,
			"seasonalTemperatureVariation": self.seasonalTemperatureVariation,
			"lastUpdateTime": self.lastUpdateTime,
			"temperatureGrid": self.temperatureGrid.tolist(),
			"salinityGrid": self.salinityGrid.tolist(),
			"lightGrid": self.lightGrid.tolist(),
			"currentGrid": self.currentGrid.tolist()
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'WaterProperties':
		"""
		Crée une instance de WaterProperties à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données des propriétés de l'eau
			
		Returns:
			Instance de WaterProperties reconstruite
		"""
		water = cls(
			worldSize=tuple(data["worldSize"]),
			resolution=tuple(data["resolution"]),
			defaultTemperature=data["defaultTemperature"],
			defaultSalinity=data["defaultSalinity"],
			defaultLightLevel=data["defaultLightLevel"]
		)
		
		water.id = data["id"]
		water.temperatureGradient = data["temperatureGradient"]
		water.salinityGradient = data["salinityGradient"]
		water.lightAttenuationCoefficient = data["lightAttenuationCoefficient"]
		water.currentSpeed = data["currentSpeed"]
		water.currentVariability = data["currentVariability"]
		water.currentChangeRate = data["currentChangeRate"]
		water.dailyTemperatureVariation = data["dailyTemperatureVariation"]
		water.seasonalTemperatureVariation = data["seasonalTemperatureVariation"]
		water.lastUpdateTime = data["lastUpdateTime"]
		
		# Reconstruire les grilles
		water.temperatureGrid = np.array(data["temperatureGrid"], dtype=np.float32)
		water.salinityGrid = np.array(data["salinityGrid"], dtype=np.float32)
		water.lightGrid = np.array(data["lightGrid"], dtype=np.float32)
		water.currentGrid = np.array(data["currentGrid"], dtype=np.float32)
		
		return water