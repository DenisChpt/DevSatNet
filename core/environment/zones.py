# core/environment/zones.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import uuid
import math

from utils.serialization import Serializable


class EnvironmentalZone(Serializable):
	"""
	Classe représentant une zone environnementale avec des propriétés spécifiques.
	Permet de définir des régions avec des températures, salinités et courants différents.
	"""
	
	def __init__(
		self,
		name: str,
		bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
		temperatureRange: Tuple[float, float] = (10.0, 20.0),
		salinityRange: Tuple[float, float] = (34.0, 36.0),
		lightRange: Tuple[float, float] = (0.2, 0.8),
		currentMagnitudeRange: Tuple[float, float] = (0.0, 0.5),
		currentDirectionRange: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
			(0.0, 0.0, 1.0),
			(0.0, 0.0, 1.0)
		),
		specialProperties: Dict[str, Any] = None
	) -> None:
		"""
		Initialise une nouvelle zone environnementale.
		
		Args:
			name: Nom de la zone
			bounds: Limites de la zone ((x_min, x_max), (y_min, y_max), (z_min, z_max))
			temperatureRange: Plage de température (min, max) en °C
			salinityRange: Plage de salinité (min, max) en PSU
			lightRange: Plage de niveau de lumière (min, max) entre 0 et 1
			currentMagnitudeRange: Plage de magnitude du courant (min, max) en m/s
			currentDirectionRange: Plage de direction du courant (min_dir, max_dir)
			specialProperties: Propriétés spéciales supplémentaires
		"""
		self.id: str = str(uuid.uuid4())
		self.name: str = name
		self.bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = bounds
		self.temperatureRange: Tuple[float, float] = temperatureRange
		self.salinityRange: Tuple[float, float] = salinityRange
		self.lightRange: Tuple[float, float] = lightRange
		self.currentMagnitudeRange: Tuple[float, float] = currentMagnitudeRange
		self.currentDirectionRange: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = currentDirectionRange
		
		# Propriétés spéciales
		self.specialProperties: Dict[str, Any] = specialProperties if specialProperties is not None else {}
		
		# État actuel
		self.currentTemperature: float = (temperatureRange[0] + temperatureRange[1]) / 2
		self.currentSalinity: float = (salinityRange[0] + salinityRange[1]) / 2
		self.currentLightLevel: float = (lightRange[0] + lightRange[1]) / 2
		self.currentMagnitude: float = (currentMagnitudeRange[0] + currentMagnitudeRange[1]) / 2
		
		# Direction du courant (moyenne des bornes)
		minDir, maxDir = currentDirectionRange
		avgDir = np.array([
			(minDir[0] + maxDir[0]) / 2,
			(minDir[1] + maxDir[1]) / 2,
			(minDir[2] + maxDir[2]) / 2
		], dtype=np.float32)
		
		# Normaliser la direction
		norm = np.linalg.norm(avgDir)
		if norm > 0:
			avgDir /= norm
			
		self.currentDirection: np.ndarray = avgDir
		
		# Variation temporelle
		self.temperatureCyclePeriod: float = 24.0  # Heures
		self.temperatureCycleAmplitude: float = (temperatureRange[1] - temperatureRange[0]) * 0.5
		self.currentCyclePeriod: float = 12.0  # Heures
		self.currentFluctuationAmplitude: float = 0.2
	
	def update(self, deltaTime: float, globalTime: float) -> None:
		"""
		Met à jour les propriétés de la zone en fonction du temps.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
			globalTime: Temps global de la simulation
		"""
		# Cycle de température
		temperaturePhase = (globalTime % self.temperatureCyclePeriod) / self.temperatureCyclePeriod
		temperatureOffset = self.temperatureCycleAmplitude * math.sin(temperaturePhase * 2 * math.pi)
		
		midTemp = (self.temperatureRange[0] + self.temperatureRange[1]) / 2
		self.currentTemperature = midTemp + temperatureOffset
		
		# Cycle de courant
		currentPhase = (globalTime % self.currentCyclePeriod) / self.currentCyclePeriod
		currentOffset = self.currentFluctuationAmplitude * math.sin(currentPhase * 2 * math.pi)
		
		midMagnitude = (self.currentMagnitudeRange[0] + self.currentMagnitudeRange[1]) / 2
		self.currentMagnitude = midMagnitude * (1.0 + currentOffset)
		
		# Fluctuation de la direction du courant
		# On perturbe légèrement la direction avec un mouvement sinusoïdal
		minDir, maxDir = self.currentDirectionRange
		dirRange = np.array([
			maxDir[0] - minDir[0],
			maxDir[1] - minDir[1],
			maxDir[2] - minDir[2]
		], dtype=np.float32)
		
		dirOffset = np.array([
			0.5 * dirRange[0] * math.sin(currentPhase * 2 * math.pi),
			0.5 * dirRange[1] * math.sin(currentPhase * 2 * math.pi + math.pi/3),
			0.5 * dirRange[2] * math.sin(currentPhase * 2 * math.pi + 2*math.pi/3)
		], dtype=np.float32)
		
		midDir = np.array([
			(minDir[0] + maxDir[0]) / 2,
			(minDir[1] + maxDir[1]) / 2,
			(minDir[2] + maxDir[2]) / 2
		], dtype=np.float32)
		
		newDir = midDir + dirOffset
		
		# Normaliser la direction
		norm = np.linalg.norm(newDir)
		if norm > 0:
			newDir /= norm
			
		self.currentDirection = newDir
		
		# Niveau de lumière (dépend de l'heure du jour)
		dayTime = globalTime % 24.0
		if 6 <= dayTime < 18:  # Jour
			dayProgress = (dayTime - 6) / 12.0  # 0 à 1 pendant la journée
			lightFactor = math.sin(dayProgress * math.pi)  # Pic à midi
			
			# Interpoler entre les limites de lumière définies
			minLight, maxLight = self.lightRange
			self.currentLightLevel = minLight + (maxLight - minLight) * lightFactor
		else:  # Nuit
			self.currentLightLevel = self.lightRange[0] * 0.2  # 20% du minimum pour la nuit
	
	def containsPosition(self, position: np.ndarray) -> bool:
		"""
		Vérifie si une position est à l'intérieur de la zone.
		
		Args:
			position: Position à vérifier
			
		Returns:
			True si la position est dans la zone, False sinon
		"""
		x, y, z = position
		(xMin, xMax), (yMin, yMax), (zMin, zMax) = self.bounds
		
		return (xMin <= x <= xMax and 
				yMin <= y <= yMax and 
				zMin <= z <= zMax)
	
	def getDistanceFromBoundary(self, position: np.ndarray) -> float:
		"""
		Calcule la distance minimale entre une position et la frontière de la zone.
		Valeur négative si la position est à l'intérieur, positive si elle est à l'extérieur.
		
		Args:
			position: Position à vérifier
			
		Returns:
			Distance au bord le plus proche
		"""
		x, y, z = position
		(xMin, xMax), (yMin, yMax), (zMin, zMax) = self.bounds
		
		# Distance aux bords sur chaque axe
		xDist = min(x - xMin, xMax - x) if xMin <= x <= xMax else -min(abs(x - xMin), abs(x - xMax))
		yDist = min(y - yMin, yMax - y) if yMin <= y <= yMax else -min(abs(y - yMin), abs(y - yMax))
		zDist = min(z - zMin, zMax - z) if zMin <= z <= zMax else -min(abs(z - zMin), abs(z - zMax))
		
		# La distance minimale est celle qui détermine si on est dedans ou dehors
		return min(xDist, yDist, zDist)
	
	def getInfluenceFactor(self, position: np.ndarray, falloffDistance: float = 50.0) -> float:
		"""
		Calcule un facteur d'influence de la zone à une position donnée.
		Le facteur est 1.0 à l'intérieur de la zone et diminue progressivement
		à mesure qu'on s'éloigne, jusqu'à 0.0 à une distance falloffDistance.
		
		Args:
			position: Position à vérifier
			falloffDistance: Distance sur laquelle l'influence diminue
			
		Returns:
			Facteur d'influence entre 0.0 et 1.0
		"""
		distance = self.getDistanceFromBoundary(position)
		
		if distance >= 0:  # Dans la zone
			return 1.0
		elif distance <= -falloffDistance:  # Trop loin
			return 0.0
		else:  # Dans la zone de transition
			# Interpolation linéaire
			return 1.0 + distance / falloffDistance
	
	def getTemperatureAt(self, position: np.ndarray, globalTime: float) -> float:
		"""
		Retourne la température à une position donnée dans ou près de la zone.
		
		Args:
			position: Position à vérifier
			globalTime: Temps global de la simulation
			
		Returns:
			Température en °C
		"""
		influence = self.getInfluenceFactor(position)
		
		if influence <= 0:
			return 0.0  # Aucune influence
			
		# Cycle de température
		temperaturePhase = (globalTime % self.temperatureCyclePeriod) / self.temperatureCyclePeriod
		temperatureOffset = self.temperatureCycleAmplitude * math.sin(temperaturePhase * 2 * math.pi)
		
		midTemp = (self.temperatureRange[0] + self.temperatureRange[1]) / 2
		temperature = midTemp + temperatureOffset
		
		return temperature * influence
	
	def getLightLevelAt(self, position: np.ndarray, globalTime: float) -> float:
		"""
		Retourne le niveau de lumière à une position donnée dans ou près de la zone.
		
		Args:
			position: Position à vérifier
			globalTime: Temps global de la simulation
			
		Returns:
			Niveau de lumière entre 0.0 et 1.0
		"""
		influence = self.getInfluenceFactor(position)
		
		if influence <= 0:
			return 0.0  # Aucune influence
			
		# Niveau de lumière (dépend de l'heure du jour)
		dayTime = globalTime % 24.0
		if 6 <= dayTime < 18:  # Jour
			dayProgress = (dayTime - 6) / 12.0  # 0 à 1 pendant la journée
			lightFactor = math.sin(dayProgress * math.pi)  # Pic à midi
			
			# Interpoler entre les limites de lumière définies
			minLight, maxLight = self.lightRange
			lightLevel = minLight + (maxLight - minLight) * lightFactor
		else:  # Nuit
			lightLevel = self.lightRange[0] * 0.2  # 20% du minimum pour la nuit
		
		return lightLevel * influence
	
	def getCurrentAt(self, position: np.ndarray, globalTime: float) -> np.ndarray:
		"""
		Retourne le vecteur de courant à une position donnée dans ou près de la zone.
		
		Args:
			position: Position à vérifier
			globalTime: Temps global de la simulation
			
		Returns:
			Vecteur de courant [vx, vy, vz] en m/s
		"""
		influence = self.getInfluenceFactor(position)
		
		if influence <= 0:
			return np.zeros(3, dtype=np.float32)  # Aucune influence
			
		# Cycle de courant
		currentPhase = (globalTime % self.currentCyclePeriod) / self.currentCyclePeriod
		currentOffset = self.currentFluctuationAmplitude * math.sin(currentPhase * 2 * math.pi)
		
		midMagnitude = (self.currentMagnitudeRange[0] + self.currentMagnitudeRange[1]) / 2
		magnitude = midMagnitude * (1.0 + currentOffset)
		
		# Calculer le vecteur de courant
		current = self.currentDirection * magnitude * influence
		
		return current
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet EnvironmentalZone en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de la zone environnementale
		"""
		return {
			"id": self.id,
			"name": self.name,
			"bounds": self.bounds,
			"temperatureRange": self.temperatureRange,
			"salinityRange": self.salinityRange,
			"lightRange": self.lightRange,
			"currentMagnitudeRange": self.currentMagnitudeRange,
			"currentDirectionRange": (
				tuple(self.currentDirectionRange[0]),
				tuple(self.currentDirectionRange[1])
			),
			"specialProperties": self.specialProperties,
			"currentTemperature": self.currentTemperature,
			"currentSalinity": self.currentSalinity,
			"currentLightLevel": self.currentLightLevel,
			"currentMagnitude": self.currentMagnitude,
			"currentDirection": self.currentDirection.tolist(),
			"temperatureCyclePeriod": self.temperatureCyclePeriod,
			"temperatureCycleAmplitude": self.temperatureCycleAmplitude,
			"currentCyclePeriod": self.currentCyclePeriod,
			"currentFluctuationAmplitude": self.currentFluctuationAmplitude
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'EnvironmentalZone':
		"""
		Crée une instance de EnvironmentalZone à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de la zone environnementale
			
		Returns:
			Instance de EnvironmentalZone reconstruite
		"""
		zone = cls(
			name=data["name"],
			bounds=data["bounds"],
			temperatureRange=data["temperatureRange"],
			salinityRange=data["salinityRange"],
			lightRange=data["lightRange"],
			currentMagnitudeRange=data["currentMagnitudeRange"],
			currentDirectionRange=data["currentDirectionRange"],
			specialProperties=data["specialProperties"]
		)
		
		zone.id = data["id"]
		zone.currentTemperature = data["currentTemperature"]
		zone.currentSalinity = data["currentSalinity"]
		zone.currentLightLevel = data["currentLightLevel"]
		zone.currentMagnitude = data["currentMagnitude"]
		zone.currentDirection = np.array(data["currentDirection"], dtype=np.float32)
		zone.temperatureCyclePeriod = data["temperatureCyclePeriod"]
		zone.temperatureCycleAmplitude = data["temperatureCycleAmplitude"]
		zone.currentCyclePeriod = data["currentCyclePeriod"]
		zone.currentFluctuationAmplitude = data["currentFluctuationAmplitude"]
		
		return zone