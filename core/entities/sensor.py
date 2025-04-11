from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import uuid

from utils.serialization import Serializable


class Sensor(Serializable):
	"""
	Classe représentant un capteur environnemental d'une créature marine.
	Les capteurs permettent aux créatures de percevoir leur environnement.
	"""
	
	def __init__(
		self,
		sensorType: str,  # Types: "vision", "pressure", "temperature", "chemical", "electromagnetic", "proximity"
		position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
		direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
		range: float = 10.0,
		sensitivity: float = 1.0,
		fieldOfView: float = 120.0,  # En degrés (pour les capteurs directionnels)
		resolution: int = 5,  # Nombre de valeurs distinctes retournées
		noiseLevel: float = 0.05  # Niveau de bruit (0.0 = pas de bruit)
	) -> None:
		"""
		Initialise un nouveau capteur.
		
		Args:
			sensorType: Type de capteur
			position: Position relative sur le corps de la créature
			direction: Direction du capteur (pour les capteurs directionnels)
			range: Portée maximale du capteur
			sensitivity: Sensibilité du capteur
			fieldOfView: Champ de vision/détection en degrés
			resolution: Nombre de points de mesure (résolution)
			noiseLevel: Niveau de bruit dans les mesures
		"""
		self.id: str = str(uuid.uuid4())
		self.sensorType: str = sensorType
		self.position: np.ndarray = np.array(position, dtype=np.float32)
		self.direction: np.ndarray = np.array(direction, dtype=np.float32)
		
		# Normaliser la direction
		norm = np.linalg.norm(self.direction)
		if norm > 0:
			self.direction /= norm
			
		self.range: float = range
		self.sensitivity: float = sensitivity
		self.fieldOfView: float = fieldOfView
		self.resolution: int = resolution
		self.noiseLevel: float = noiseLevel
		
		# État actuel
		self.lastReadings: np.ndarray = np.zeros(resolution, dtype=np.float32)
		self.lastReadingTime: float = 0.0
	
	def sense(
		self, 
		environment: Any, 
		creaturePosition: np.ndarray, 
		creatureOrientation: np.ndarray
	) -> List[float]:
		"""
		Effectue une mesure de l'environnement et retourne les valeurs captées.
		
		Args:
			environment: L'environnement marin
			creaturePosition: Position absolue de la créature
			creatureOrientation: Orientation de la créature (angles d'Euler)
			
		Returns:
			Liste des valeurs mesurées
		"""
		# Calculer la position et l'orientation absolues du capteur
		sensorWorldPos = self._calculateWorldPosition(creaturePosition, creatureOrientation)
		sensorWorldDir = self._calculateWorldDirection(creatureOrientation)
		
		# Effectuer la mesure selon le type de capteur
		if self.sensorType == "vision":
			readings = self._senseVision(environment, sensorWorldPos, sensorWorldDir)
		elif self.sensorType == "pressure":
			readings = self._sensePressure(environment, sensorWorldPos)
		elif self.sensorType == "temperature":
			readings = self._senseTemperature(environment, sensorWorldPos)
		elif self.sensorType == "chemical":
			readings = self._senseChemical(environment, sensorWorldPos)
		elif self.sensorType == "electromagnetic":
			readings = self._senseElectromagnetic(environment, sensorWorldPos)
		elif self.sensorType == "proximity":
			readings = self._senseProximity(environment, sensorWorldPos, sensorWorldDir)
		else:
			# Capteur par défaut
			readings = np.zeros(self.resolution, dtype=np.float32)
		
		# Ajouter du bruit aux mesures
		if self.noiseLevel > 0:
			noise = np.random.normal(0, self.noiseLevel, size=len(readings))
			readings = readings + noise
			
		# Limiter les valeurs entre 0 et 1
		readings = np.clip(readings, 0.0, 1.0)
		
		# Stocker les résultats
		self.lastReadings = readings
		self.lastReadingTime = environment.currentTime
		
		return readings.tolist()
	
	def _calculateWorldPosition(self, creaturePosition: np.ndarray, creatureOrientation: np.ndarray) -> np.ndarray:
		"""
		Calcule la position absolue du capteur dans le monde.
		
		Args:
			creaturePosition: Position absolue de la créature
			creatureOrientation: Orientation de la créature (angles d'Euler)
			
		Returns:
			Position absolue du capteur
		"""
		# Créer la matrice de rotation à partir de l'orientation
		rotationMatrix = self._createRotationMatrix(creatureOrientation)
		
		# Appliquer la rotation à la position relative du capteur
		rotatedPosition = np.dot(rotationMatrix, self.position)
		
		# Ajouter la position de la créature pour obtenir la position absolue
		worldPosition = creaturePosition + rotatedPosition
		
		return worldPosition
	
	def _calculateWorldDirection(self, creatureOrientation: np.ndarray) -> np.ndarray:
		"""
		Calcule la direction absolue du capteur dans le monde.
		
		Args:
			creatureOrientation: Orientation de la créature (angles d'Euler)
			
		Returns:
			Direction absolue du capteur
		"""
		# Créer la matrice de rotation à partir de l'orientation
		rotationMatrix = self._createRotationMatrix(creatureOrientation)
		
		# Appliquer la rotation à la direction relative du capteur
		worldDirection = np.dot(rotationMatrix, self.direction)
		
		# Normaliser la direction
		norm = np.linalg.norm(worldDirection)
		if norm > 0:
			worldDirection /= norm
			
		return worldDirection
	
	def _createRotationMatrix(self, orientation: np.ndarray) -> np.ndarray:
		"""
		Crée une matrice de rotation 3D à partir des angles d'Euler.
		
		Args:
			orientation: Angles d'Euler (pitch, yaw, roll) en radians
			
		Returns:
			Matrice de rotation 3x3
		"""
		# Extraire les angles
		pitch, yaw, roll = orientation
		
		# Calcul des sinus et cosinus
		cp, sp = np.cos(pitch), np.sin(pitch)
		cy, sy = np.cos(yaw), np.sin(yaw)
		cr, sr = np.cos(roll), np.sin(roll)
		
		# Construire la matrice de rotation (ordre XYZ)
		rotationMatrix = np.array([
			[cy*cr, -cy*sr, sy],
			[cp*sr + sp*sy*cr, cp*cr - sp*sy*sr, -sp*cy],
			[sp*sr - cp*sy*cr, sp*cr + cp*sy*sr, cp*cy]
		])
		
		return rotationMatrix
	
	def _senseVision(self, environment: Any, position: np.ndarray, direction: np.ndarray) -> np.ndarray:
		"""
		Simule la vision en détectant les objets dans le champ de vision.
		
		Args:
			environment: L'environnement marin
			position: Position absolue du capteur
			direction: Direction absolue du capteur
			
		Returns:
			Tableau des intensités lumineuses détectées
		"""
		# Convertir le champ de vision en radians
		fovRad = np.radians(self.fieldOfView)
		
		# Créer des rayons dans le champ de vision
		readings = np.zeros(self.resolution, dtype=np.float32)
		
		# Pour chaque segment du champ de vision
		for i in range(self.resolution):
			# Calculer l'angle pour ce rayon
			angle = -fovRad/2 + fovRad * i / (self.resolution-1)
			
			# Rotation du rayon autour de l'axe vertical (simplification 2D)
			rayDirection = np.array([
				direction[0] * np.cos(angle) - direction[2] * np.sin(angle),
				direction[1],
				direction[0] * np.sin(angle) + direction[2] * np.cos(angle)
			])
			
			# Normaliser la direction
			rayDirection = rayDirection / np.linalg.norm(rayDirection)
			
			# Lancer le rayon dans l'environnement
			hit, distance, objectType = environment.rayCast(position, rayDirection, self.range)
			
			if hit:
				# Calculer l'intensité en fonction de la distance
				intensity = 1.0 - min(1.0, distance / self.range)
				intensity *= self.sensitivity
				
				# Ajuster selon le type d'objet détecté
				if objectType == "creature":
					intensity *= 0.8  # Créatures moins visibles
				elif objectType == "food":
					intensity *= 1.2  # Nourriture plus visible
				
				readings[i] = intensity
			else:
				# Aucune collision, lire le niveau de lumière ambiant
				lightLevel = environment.getLightLevelAt(position)
				readings[i] = lightLevel * 0.2 * self.sensitivity
		
		return readings
	
	def _sensePressure(self, environment: Any, position: np.ndarray) -> np.ndarray:
		"""
		Détecte la pression de l'eau à la position actuelle.
		
		Args:
			environment: L'environnement marin
			position: Position absolue du capteur
			
		Returns:
			Tableau avec la mesure de pression
		"""
		# Obtenir la pression à la position actuelle
		pressure = environment.getPressureAt(position)
		
		# Normaliser dans la plage [0, 1]
		maxPressure = 100.0  # Pression maximale attendue
		normalizedPressure = min(1.0, pressure / maxPressure)
		
		# Appliquer la sensibilité
		reading = normalizedPressure * self.sensitivity
		
		# Créer un tableau de la taille de résolution spécifiée
		# Pour un capteur de pression simple, toutes les valeurs sont identiques
		readings = np.ones(self.resolution, dtype=np.float32) * reading
		
		return readings
	
	def _senseTemperature(self, environment: Any, position: np.ndarray) -> np.ndarray:
		"""
		Détecte la température de l'eau à la position actuelle.
		
		Args:
			environment: L'environnement marin
			position: Position absolue du capteur
			
		Returns:
			Tableau avec la mesure de température
		"""
		# Obtenir la température à la position actuelle
		temperature = environment.getTemperatureAt(position)
		
		# Normaliser dans la plage [0, 1]
		minTemp, maxTemp = 0.0, 30.0  # Plage de température attendue en °C
		normalizedTemp = max(0.0, min(1.0, (temperature - minTemp) / (maxTemp - minTemp)))
		
		# Appliquer la sensibilité
		reading = normalizedTemp * self.sensitivity
		
		# Créer un tableau de la taille de résolution spécifiée
		readings = np.ones(self.resolution, dtype=np.float32) * reading
		
		return readings
	
	def _senseChemical(self, environment: Any, position: np.ndarray) -> np.ndarray:
		"""
		Détecte les substances chimiques dans l'eau (nourriture, prédateurs).
		
		Args:
			environment: L'environnement marin
			position: Position absolue du capteur
			
		Returns:
			Tableau des concentrations chimiques détectées
		"""
		# Obtenir les concentrations chimiques à proximité
		chemicals = environment.getChemicalsAt(position, self.range)
		
		# Un capteur chimique peut détecter différentes substances
		readings = np.zeros(self.resolution, dtype=np.float32)
		
		# Traiter chaque type de substance (on suppose que l'environnement renvoie
		# un dictionnaire {type -> concentration})
		typeIndex = 0
		for chemType, concentration in chemicals.items():
			if typeIndex < self.resolution:
				# Normaliser et appliquer la sensibilité
				normalizedConc = min(1.0, concentration * self.sensitivity)
				readings[typeIndex] = normalizedConc
				typeIndex += 1
		
		return readings
	
	def _senseElectromagnetic(self, environment: Any, position: np.ndarray) -> np.ndarray:
		"""
		Détecte les champs électromagnétiques (comme certains poissons).
		
		Args:
			environment: L'environnement marin
			position: Position absolue du capteur
			
		Returns:
			Tableau des intensités électromagnétiques détectées
		"""
		# Obtenir les sources électromagnétiques à proximité
		emSources = environment.getElectromagneticSourcesNear(position, self.range)
		
		# Initialiser les lectures
		readings = np.zeros(self.resolution, dtype=np.float32)
		
		# Diviser le champ de détection en secteurs
		sectorAngle = 2 * np.pi / self.resolution
		
		# Pour chaque source EM
		for source in emSources:
			sourcePos, intensity = source
			
			# Calculer la direction de la source
			direction = sourcePos - position
			distance = np.linalg.norm(direction)
			
			if distance <= self.range:
				# Calculer l'angle par rapport au capteur
				angle = np.arctan2(direction[2], direction[0])  # Dans le plan XZ
				
				# Normaliser l'angle entre 0 et 2π
				angle = (angle + 2*np.pi) % (2*np.pi)
				
				# Déterminer le secteur correspondant
				sector = int(angle / sectorAngle)
				sector = min(sector, self.resolution - 1)  # Assurer que l'index est valide
				
				# Calculer l'intensité en fonction de la distance
				sensorIntensity = intensity * (1.0 - min(1.0, distance / self.range))
				sensorIntensity *= self.sensitivity
				
				# Mettre à jour la lecture du secteur (prendre la valeur maximale)
				readings[sector] = max(readings[sector], sensorIntensity)
		
		return readings
	
	def _senseProximity(self, environment: Any, position: np.ndarray, direction: np.ndarray) -> np.ndarray:
		"""
		Détecte la proximité d'objets, comme un sonar simplifié.
		
		Args:
			environment: L'environnement marin
			position: Position absolue du capteur
			direction: Direction absolue du capteur
			
		Returns:
			Tableau des distances détectées
		"""
		# Convertir le champ de détection en radians
		fovRad = np.radians(self.fieldOfView)
		
		# Initialiser les lectures
		readings = np.zeros(self.resolution, dtype=np.float32)
		
		# Pour chaque segment du champ de détection
		for i in range(self.resolution):
			# Calculer l'angle pour ce rayon
			angle = -fovRad/2 + fovRad * i / (self.resolution-1)
			
			# Rotation du rayon autour de l'axe vertical (simplification 2D)
			rayDirection = np.array([
				direction[0] * np.cos(angle) - direction[2] * np.sin(angle),
				direction[1],
				direction[0] * np.sin(angle) + direction[2] * np.cos(angle)
			])
			
			# Normaliser la direction
			rayDirection = rayDirection / np.linalg.norm(rayDirection)
			
			# Lancer le rayon dans l'environnement
			hit, distance, _ = environment.rayCast(position, rayDirection, self.range)
			
			if hit:
				# Normaliser la distance inversée (plus proche = plus élevé)
				normalizedDistance = 1.0 - min(1.0, distance / self.range)
				readings[i] = normalizedDistance * self.sensitivity
			else:
				readings[i] = 0.0  # Aucun obstacle détecté
		
		return readings
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Sensor en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du capteur
		"""
		return {
			"id": self.id,
			"sensorType": self.sensorType,
			"position": self.position.tolist(),
			"direction": self.direction.tolist(),
			"range": self.range,
			"sensitivity": self.sensitivity,
			"fieldOfView": self.fieldOfView,
			"resolution": self.resolution,
			"noiseLevel": self.noiseLevel,
			"lastReadings": self.lastReadings.tolist(),
			"lastReadingTime": self.lastReadingTime
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Sensor':
		"""
		Crée une instance de Sensor à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du capteur
			
		Returns:
			Instance de Sensor reconstruite
		"""
		sensor = cls(
			sensorType=data["sensorType"],
			position=tuple(data["position"]),
			direction=tuple(data["direction"]),
			range=data["range"],
			sensitivity=data["sensitivity"],
			fieldOfView=data["fieldOfView"],
			resolution=data["resolution"],
			noiseLevel=data["noiseLevel"]
		)
		
		sensor.id = data["id"]
		sensor.lastReadings = np.array(data["lastReadings"], dtype=np.float32)
		sensor.lastReadingTime = data["lastReadingTime"]
		
		return sensor