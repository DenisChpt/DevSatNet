# core/environment/marine_world.py
from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np
import uuid
import time
from collections import defaultdict

from core.environment.water_properties import WaterProperties
from core.environment.terrain import Terrain
from core.environment.resources import FoodResource
from core.environment.zones import EnvironmentalZone
from utils.serialization import Serializable


class MarineWorld(Serializable):
	"""
	Classe principale représentant l'environnement marin global.
	Cette classe coordonne tous les aspects de l'environnement.
	"""
	
	def __init__(
		self,
		size: Tuple[float, float, float] = (1000.0, 500.0, 1000.0),
		resolution: Tuple[int, int, int] = (100, 50, 100),
		defaultTemperature: float = 15.0,
		defaultSalinity: float = 35.0,
		defaultLightLevel: float = 0.8,
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise un nouveau monde marin.
		
		Args:
			size: Dimensions du monde (largeur, hauteur, profondeur) en unités de simulation
			resolution: Résolution de la grille pour les calculs physiques
			defaultTemperature: Température par défaut de l'eau en degrés Celsius
			defaultSalinity: Salinité par défaut de l'eau en PSU (Practical Salinity Unit)
			defaultLightLevel: Niveau de lumière par défaut à la surface (0.0 à 1.0)
			seed: Graine pour le générateur de nombres aléatoires
		"""
		self.id: str = str(uuid.uuid4())
		self.size: np.ndarray = np.array(size, dtype=np.float32)
		self.resolution: np.ndarray = np.array(resolution, dtype=np.int32)
		self.defaultTemperature: float = defaultTemperature
		self.defaultSalinity: float = defaultSalinity
		self.defaultLightLevel: float = defaultLightLevel
		
		# Initialiser le générateur de nombres aléatoires
		if seed is None:
			seed = int(time.time())
		self.rng: np.random.Generator = np.random.default_rng(seed)
		self.seed: int = seed
		
		# Temps de simulation
		self.currentTime: float = 0.0
		self.dayLength: float = 24.0  # Longueur d'une journée en unités de temps
		self.yearLength: float = 365.0  # Longueur d'une année en jours
		
		# Composants de l'environnement
		self.terrain: Terrain = Terrain(size, resolution, seed=seed)
		self.waterProperties: WaterProperties = WaterProperties(
			size, 
			resolution, 
			defaultTemperature=defaultTemperature, 
			defaultSalinity=defaultSalinity,
			defaultLightLevel=defaultLightLevel,
			seed=seed
		)
		
		# Zones environnementales
		self.zones: List[EnvironmentalZone] = []
		
		# Ressources
		self.foodResources: List[FoodResource] = []
		self.maxFoodResources: int = 1000  # Nombre maximum de ressources alimentaires
		self.foodRegenerationRate: float = 0.1  # Taux de régénération par unité de temps
		
		# Entités
		self.creatures: Dict[str, Any] = {}  # id -> creature
		self.occupancyGrid: Dict[Tuple[int, int, int], Set[str]] = defaultdict(set)  # Grille d'occupation spatiale
		
		# Statistiques
		self.foodConsumed: int = 0
		self.creaturesBorn: int = 0
		self.creaturesDied: int = 0
		
		# Initialiser le monde
		self._initializeZones()
		self._initializeFoodResources()
	
	def _initializeZones(self) -> None:
		"""
		Initialise les zones environnementales du monde marin.
		Chaque zone peut avoir des caractéristiques différentes (température, courants, etc.).
		"""
		# Zone de surface
		surfaceZone = EnvironmentalZone(
			name="Surface Zone",
			bounds=((0, self.size[0]), (0, 100), (0, self.size[2])),
			temperatureRange=(18.0, 25.0),
			salinityRange=(34.0, 36.0),
			lightRange=(0.7, 1.0),
			currentMagnitudeRange=(0.0, 0.5),
			currentDirectionRange=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
		)
		self.zones.append(surfaceZone)
		
		# Zone intermédiaire
		intermediateZone = EnvironmentalZone(
			name="Intermediate Zone",
			bounds=((0, self.size[0]), (100, 250), (0, self.size[2])),
			temperatureRange=(10.0, 18.0),
			salinityRange=(34.5, 35.5),
			lightRange=(0.2, 0.7),
			currentMagnitudeRange=(0.0, 0.3),
			currentDirectionRange=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
		)
		self.zones.append(intermediateZone)
		
		# Zone profonde
		deepZone = EnvironmentalZone(
			name="Deep Zone",
			bounds=((0, self.size[0]), (250, self.size[1]), (0, self.size[2])),
			temperatureRange=(4.0, 10.0),
			salinityRange=(34.0, 35.0),
			lightRange=(0.0, 0.2),
			currentMagnitudeRange=(0.0, 0.1),
			currentDirectionRange=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
		)
		self.zones.append(deepZone)
		
		# Zone de récif
		reefZone = EnvironmentalZone(
			name="Reef Zone",
			bounds=((200, 400), (50, 150), (200, 400)),
			temperatureRange=(20.0, 28.0),
			salinityRange=(35.0, 36.0),
			lightRange=(0.6, 0.9),
			currentMagnitudeRange=(0.1, 0.4),
			currentDirectionRange=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
		)
		self.zones.append(reefZone)
		
		# Zone de courant marin
		currentZone = EnvironmentalZone(
			name="Current Zone",
			bounds=((400, 600), (50, 200), (0, self.size[2])),
			temperatureRange=(15.0, 22.0),
			salinityRange=(35.0, 35.5),
			lightRange=(0.5, 0.8),
			currentMagnitudeRange=(0.5, 1.0),
			currentDirectionRange=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
		)
		self.zones.append(currentZone)
	
	def _initializeFoodResources(self) -> None:
		"""
		Initialise les ressources alimentaires dans le monde marin.
		"""
		# Vider la liste actuelle de ressources
		self.foodResources.clear()
		
		# Nombre de ressources à créer
		numResources = self.rng.integers(self.maxFoodResources // 2, self.maxFoodResources)
		
		for _ in range(numResources):
			# Position aléatoire dans le monde
			position = np.array([
				self.rng.uniform(0, self.size[0]),
				self.rng.uniform(0, self.size[1]),
				self.rng.uniform(0, self.size[2])
			], dtype=np.float32)
			
			# Déterminer l'énergie en fonction de la profondeur (plus d'énergie près de la surface)
			depthFactor = 1.0 - (position[1] / self.size[1])
			energyValue = self.rng.uniform(10.0, 50.0) * (0.5 + 0.5 * depthFactor)
			
			# Déterminer le type de nourriture
			foodTypes = ["algae", "plankton", "small_fish", "detritus"]
			weights = [0.4, 0.3, 0.2, 0.1]
			foodType = self.rng.choice(foodTypes, p=weights)
			
			# Créer la ressource alimentaire
			food = FoodResource(
				position=tuple(position),
				energyValue=energyValue,
				foodType=foodType,
				size=self.rng.uniform(0.5, 2.0)
			)
			
			self.foodResources.append(food)
	
	def update(self, deltaTime: float) -> None:
		"""
		Met à jour l'état du monde marin pour un pas de temps.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
		"""
		# Mettre à jour le temps courant
		self.currentTime += deltaTime
		
		# Mettre à jour les propriétés de l'eau (courants, température, etc.)
		self.waterProperties.update(deltaTime, self.currentTime)
		
		# Mettre à jour les zones environnementales
		for zone in self.zones:
			zone.update(deltaTime, self.currentTime)
		
		# Regénérer les ressources alimentaires
		self._updateFoodResources(deltaTime)
		
		# Mettre à jour la grille d'occupation spatiale
		self._updateOccupancyGrid()
	
	def _updateFoodResources(self, deltaTime: float) -> None:
		"""
		Met à jour les ressources alimentaires (consommation, régénération).
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
		"""
		# Supprimer les ressources consommées
		self.foodResources = [food for food in self.foodResources if not food.isConsumed]
		
		# Ajouter de nouvelles ressources avec une certaine probabilité
		if len(self.foodResources) < self.maxFoodResources:
			numNewResources = int(self.foodRegenerationRate * deltaTime * (self.maxFoodResources - len(self.foodResources)))
			numNewResources = min(numNewResources, self.maxFoodResources - len(self.foodResources))
			
			for _ in range(numNewResources):
				# Position aléatoire dans le monde
				position = np.array([
					self.rng.uniform(0, self.size[0]),
					self.rng.uniform(0, self.size[1]),
					self.rng.uniform(0, self.size[2])
				], dtype=np.float32)
				
				# Plus de chances près de la surface
				if self.rng.random() > position[1] / self.size[1]:
					# Déterminer l'énergie en fonction de la profondeur
					depthFactor = 1.0 - (position[1] / self.size[1])
					energyValue = self.rng.uniform(10.0, 50.0) * (0.5 + 0.5 * depthFactor)
					
					# Déterminer le type de nourriture
					foodTypes = ["algae", "plankton", "small_fish", "detritus"]
					weights = [0.4, 0.3, 0.2, 0.1]
					foodType = self.rng.choice(foodTypes, p=weights)
					
					# Créer la ressource alimentaire
					food = FoodResource(
						position=tuple(position),
						energyValue=energyValue,
						foodType=foodType,
						size=self.rng.uniform(0.5, 2.0)
					)
					
					self.foodResources.append(food)
	
	def _updateOccupancyGrid(self) -> None:
		"""
		Met à jour la grille d'occupation spatiale pour l'optimisation des collisions.
		"""
		# Vider la grille
		self.occupancyGrid.clear()
		
		# Ajouter les créatures à la grille
		for creatureId, creature in self.creatures.items():
			if creature.isAlive:
				# Convertir la position en indices de cellule
				cellX = int(creature.position[0] / self.size[0] * self.resolution[0])
				cellY = int(creature.position[1] / self.size[1] * self.resolution[1])
				cellZ = int(creature.position[2] / self.size[2] * self.resolution[2])
				
				# Limiter aux indices valides
				cellX = max(0, min(cellX, self.resolution[0] - 1))
				cellY = max(0, min(cellY, self.resolution[1] - 1))
				cellZ = max(0, min(cellZ, self.resolution[2] - 1))
				
				# Ajouter la créature à la cellule
				self.occupancyGrid[(cellX, cellY, cellZ)].add(creatureId)
		
		# Ajouter les ressources alimentaires à la grille
		for i, food in enumerate(self.foodResources):
			if not food.isConsumed:
				# Convertir la position en indices de cellule
				cellX = int(food.position[0] / self.size[0] * self.resolution[0])
				cellY = int(food.position[1] / self.size[1] * self.resolution[1])
				cellZ = int(food.position[2] / self.size[2] * self.resolution[2])
				
				# Limiter aux indices valides
				cellX = max(0, min(cellX, self.resolution[0] - 1))
				cellY = max(0, min(cellY, self.resolution[1] - 1))
				cellZ = max(0, min(cellZ, self.resolution[2] - 1))
				
				# Ajouter l'ID de la ressource à la cellule (avec préfixe "food_")
				self.occupancyGrid[(cellX, cellY, cellZ)].add(f"food_{i}")
	
	def addCreature(self, creature: Any) -> None:
		"""
		Ajoute une créature au monde marin.
		
		Args:
			creature: La créature à ajouter
		"""
		if creature.id in self.creatures:
			return  # Éviter les doublons
			
		self.creatures[creature.id] = creature
		self.creaturesBorn += 1
	
	def removeCreature(self, creatureId: str) -> None:
		"""
		Retire une créature du monde marin.
		
		Args:
			creatureId: ID de la créature à retirer
		"""
		if creatureId in self.creatures:
			self.creatures.pop(creatureId)
			self.creaturesDied += 1
	
	def getCreaturesNear(self, position: np.ndarray, radius: float) -> List[Any]:
		"""
		Retourne les créatures proches d'une position donnée.
		
		Args:
			position: Position centrale pour la recherche
			radius: Rayon de recherche
			
		Returns:
			Liste des créatures dans le rayon spécifié
		"""
		nearbyCreatures = []
		
		# Convertir la position en indices de cellule
		cellX = int(position[0] / self.size[0] * self.resolution[0])
		cellY = int(position[1] / self.size[1] * self.resolution[1])
		cellZ = int(position[2] / self.size[2] * self.resolution[2])
		
		# Déterminer le nombre de cellules à vérifier en fonction du rayon
		cellRadius = int(radius / min(
			self.size[0] / self.resolution[0],
			self.size[1] / self.resolution[1],
			self.size[2] / self.resolution[2]
		)) + 1
		
		# Vérifier les cellules voisines
		for dx in range(-cellRadius, cellRadius + 1):
			for dy in range(-cellRadius, cellRadius + 1):
				for dz in range(-cellRadius, cellRadius + 1):
					# Calculer les indices de la cellule voisine
					nx = cellX + dx
					ny = cellY + dy
					nz = cellZ + dz
					
					# Vérifier si la cellule est valide
					if (0 <= nx < self.resolution[0] and
						0 <= ny < self.resolution[1] and
						0 <= nz < self.resolution[2]):
						
						# Obtenir les créatures dans cette cellule
						creatureIds = [cid for cid in self.occupancyGrid.get((nx, ny, nz), set())
									  if not cid.startswith("food_")]
						
						for creatureId in creatureIds:
							creature = self.creatures.get(creatureId)
							if creature and creature.isAlive:
								# Calculer la distance réelle
								distance = np.linalg.norm(creature.position - position)
								if distance <= radius:
									nearbyCreatures.append(creature)
		
		return nearbyCreatures
	
	def getFoodNear(self, position: np.ndarray, radius: float) -> List[FoodResource]:
		"""
		Retourne les ressources alimentaires proches d'une position donnée.
		
		Args:
			position: Position centrale pour la recherche
			radius: Rayon de recherche
			
		Returns:
			Liste des ressources alimentaires dans le rayon spécifié
		"""
		nearbyFood = []
		
		# Convertir la position en indices de cellule
		cellX = int(position[0] / self.size[0] * self.resolution[0])
		cellY = int(position[1] / self.size[1] * self.resolution[1])
		cellZ = int(position[2] / self.size[2] * self.resolution[2])
		
		# Déterminer le nombre de cellules à vérifier en fonction du rayon
		cellRadius = int(radius / min(
			self.size[0] / self.resolution[0],
			self.size[1] / self.resolution[1],
			self.size[2] / self.resolution[2]
		)) + 1
		
		# Vérifier les cellules voisines
		for dx in range(-cellRadius, cellRadius + 1):
			for dy in range(-cellRadius, cellRadius + 1):
				for dz in range(-cellRadius, cellRadius + 1):
					# Calculer les indices de la cellule voisine
					nx = cellX + dx
					ny = cellY + dy
					nz = cellZ + dz
					
					# Vérifier si la cellule est valide
					if (0 <= nx < self.resolution[0] and
						0 <= ny < self.resolution[1] and
						0 <= nz < self.resolution[2]):
						
						# Obtenir les ressources alimentaires dans cette cellule
						foodIds = [fid for fid in self.occupancyGrid.get((nx, ny, nz), set())
								  if fid.startswith("food_")]
						
						for foodId in foodIds:
							# Extraire l'index de la ressource
							foodIndex = int(foodId.split("_")[1])
							if foodIndex < len(self.foodResources):
								food = self.foodResources[foodIndex]
								if not food.isConsumed:
									# Calculer la distance réelle
									distance = np.linalg.norm(np.array(food.position) - position)
									if distance <= radius:
										nearbyFood.append(food)
		
		return nearbyFood
	
	def consumeFood(self, foodResource: FoodResource, creatureId: str) -> float:
		"""
		Marque une ressource alimentaire comme consommée et retourne sa valeur énergétique.
		
		Args:
			foodResource: La ressource à consommer
			creatureId: ID de la créature qui consomme la ressource
			
		Returns:
			Valeur énergétique de la ressource
		"""
		if not foodResource.isConsumed:
			foodResource.consume(creatureId)
			self.foodConsumed += 1
			return foodResource.energyValue
		
		return 0.0
	
	def getZoneAt(self, position: np.ndarray) -> Optional[EnvironmentalZone]:
		"""
		Retourne la zone environnementale à une position donnée.
		
		Args:
			position: Position à vérifier
			
		Returns:
			Zone environnementale contenant la position, ou None si aucune
		"""
		for zone in self.zones:
			if zone.containsPosition(position):
				return zone
		
		return None
	
	def getTemperatureAt(self, position: np.ndarray) -> float:
		"""
		Retourne la température de l'eau à une position donnée.
		
		Args:
			position: Position à vérifier
			
		Returns:
			Température en degrés Celsius
		"""
		# Vérifier si la position est dans une zone spécifique
		zone = self.getZoneAt(position)
		if zone:
			return zone.getTemperatureAt(position, self.currentTime)
		
		# Sinon, utiliser le modèle global de température
		return self.waterProperties.getTemperatureAt(position, self.currentTime)
	
	def getPressureAt(self, position: np.ndarray) -> float:
		"""
		Retourne la pression de l'eau à une position donnée.
		
		Args:
			position: Position à vérifier
			
		Returns:
			Pression en bars
		"""
		# La pression augmente avec la profondeur (environ 1 bar tous les 10 mètres)
		depth = position[1]  # Profondeur (axe Y)
		atmosphericPressure = 1.0  # Pression atmosphérique en surface (1 bar)
		
		# Calculer la pression hydrostatique
		waterDensity = 1025.0  # kg/m³
		gravitationalAcceleration = 9.81  # m/s²
		hydrostaticPressure = depth * waterDensity * gravitationalAcceleration / 100000.0  # Conversion en bars
		
		# Pression totale
		totalPressure = atmosphericPressure + hydrostaticPressure
		
		return totalPressure
	
	def getLightLevelAt(self, position: np.ndarray) -> float:
		"""
		Retourne le niveau de lumière à une position donnée.
		
		Args:
			position: Position à vérifier
			
		Returns:
			Niveau de lumière entre 0.0 (obscurité) et 1.0 (pleine lumière)
		"""
		# Vérifier si la position est dans une zone spécifique
		zone = self.getZoneAt(position)
		if zone:
			return zone.getLightLevelAt(position, self.currentTime)
		
		# Sinon, utiliser le modèle global de lumière
		return self.waterProperties.getLightLevelAt(position, self.currentTime)
	
	def getCurrentAt(self, position: np.ndarray) -> np.ndarray:
		"""
		Retourne le vecteur de courant d'eau à une position donnée.
		
		Args:
			position: Position à vérifier
			
		Returns:
			Vecteur de vitesse du courant (m/s)
		"""
		# Vérifier si la position est dans une zone spécifique
		zone = self.getZoneAt(position)
		if zone:
			return zone.getCurrentAt(position, self.currentTime)
		
		# Sinon, utiliser le modèle global de courant
		return self.waterProperties.getCurrentAt(position, self.currentTime)
	
	def getChemicalsAt(self, position: np.ndarray, radius: float) -> Dict[str, float]:
		"""
		Retourne les concentrations chimiques dans l'eau à une position donnée.
		
		Args:
			position: Position à vérifier
			radius: Rayon de détection
			
		Returns:
			Dictionnaire des concentrations chimiques {type -> concentration}
		"""
		chemicals = {
			"oxygen": 0.0,
			"nutrients": 0.0,
			"prey_scent": 0.0,
			"predator_scent": 0.0
		}
		
		# Oxygène diminue avec la profondeur
		depthFactor = max(0.0, 1.0 - position[1] / self.size[1])
		chemicals["oxygen"] = 0.2 + 0.8 * depthFactor
		
		# Nutriments augmentent près du fond et près de la surface
		surfaceProximity = max(0.0, 1.0 - position[1] / (0.2 * self.size[1]))
		bottomProximity = max(0.0, 1.0 - (self.size[1] - position[1]) / (0.2 * self.size[1]))
		chemicals["nutrients"] = max(0.2, 0.8 * (surfaceProximity + 0.5 * bottomProximity))
		
		# Odeurs de proies et prédateurs basées sur les créatures à proximité
		nearbyCreatures = self.getCreaturesNear(position, radius)
		for creature in nearbyCreatures:
			distance = np.linalg.norm(creature.position - position)
			if distance > 0:
				# L'intensité de l'odeur diminue avec la distance
				intensity = 1.0 - min(1.0, distance / radius)
				
				# Considérer les carnivores comme prédateurs et les herbivores comme proies
				# (simplification)
				if hasattr(creature, "dietType"):
					if creature.dietType == "carnivore":
						chemicals["predator_scent"] += 0.2 * intensity
					elif creature.dietType == "herbivore":
						chemicals["prey_scent"] += 0.2 * intensity
		
		# Normaliser les valeurs
		for key in chemicals:
			chemicals[key] = min(1.0, chemicals[key])
			
		return chemicals
	
	def getElectromagneticSourcesNear(self, position: np.ndarray, radius: float) -> List[Tuple[np.ndarray, float]]:
		"""
		Retourne les sources électromagnétiques à proximité.
		
		Args:
			position: Position à vérifier
			radius: Rayon de détection
			
		Returns:
			Liste des sources EM [(position, intensité), ...]
		"""
		sources = []
		
		# Les créatures peuvent émettre des signaux électromagnétiques
		nearbyCreatures = self.getCreaturesNear(position, radius)
		for creature in nearbyCreatures:
			# Déterminer si la créature émet un signal EM (simplification)
			emitsEM = False
			if hasattr(creature, "traits"):
				# Vérifier s'il y a un trait d'émission électromagnétique
				if "electromagnetic_emission" in creature.traits:
					emitsEM = creature.traits["electromagnetic_emission"] > 0.5
			
			if emitsEM:
				intensity = 0.8  # Intensité par défaut
				sources.append((creature.position, intensity))
		
		return sources
	
	def rayCast(self, origin: np.ndarray, direction: np.ndarray, maxDistance: float) -> Tuple[bool, float, str]:
		"""
		Lance un rayon dans l'environnement et détecte la première collision.
		
		Args:
			origin: Point d'origine du rayon
			direction: Direction du rayon (normalisée)
			maxDistance: Distance maximale de recherche
			
		Returns:
			Tuple (hit, distance, objectType) indiquant s'il y a eu une collision,
			à quelle distance, et avec quel type d'objet
		"""
		# Vérifier la collision avec le terrain
		terrainHit, terrainDistance = self.terrain.rayCast(origin, direction, maxDistance)
		
		# Vérifier la collision avec les créatures
		nearbyCreatures = self.getCreaturesNear(origin, maxDistance)
		creatureHit = False
		creatureDistance = float('inf')
		for creature in nearbyCreatures:
			# Simplification: on considère chaque créature comme une sphère
			creatureRadius = creature.size if hasattr(creature, "size") else 1.0
			
			# Vecteur de l'origine vers le centre de la créature
			toCreature = creature.position - origin
			
			# Projection de ce vecteur sur la direction du rayon
			projection = np.dot(toCreature, direction)
			
			# Si la projection est négative, la créature est derrière le rayon
			if projection < 0:
				continue
				
			# Distance perpendiculaire entre le centre de la créature et la ligne du rayon
			perpDistance = np.linalg.norm(toCreature - projection * direction)
			
			# Si cette distance est inférieure au rayon de la créature, il y a collision
			if perpDistance <= creatureRadius:
				# Calculer la distance au point d'entrée
				distanceToCreature = projection - np.sqrt(creatureRadius**2 - perpDistance**2)
				
				# Si cette distance est la plus proche jusqu'à présent et dans la portée
				if distanceToCreature < creatureDistance and distanceToCreature <= maxDistance:
					creatureHit = True
					creatureDistance = distanceToCreature
		
		# Vérifier la collision avec les ressources alimentaires
		nearbyFood = self.getFoodNear(origin, maxDistance)
		foodHit = False
		foodDistance = float('inf')
		for food in nearbyFood:
			# Simplification: on considère chaque ressource comme une sphère
			foodRadius = food.size
			
			# Vecteur de l'origine vers le centre de la ressource
			position = np.array(food.position)
			toFood = position - origin
			
			# Projection de ce vecteur sur la direction du rayon
			projection = np.dot(toFood, direction)
			
			# Si la projection est négative, la ressource est derrière le rayon
			if projection < 0:
				continue
				
			# Distance perpendiculaire entre le centre de la ressource et la ligne du rayon
			perpDistance = np.linalg.norm(toFood - projection * direction)
			
			# Si cette distance est inférieure au rayon de la ressource, il y a collision
			if perpDistance <= foodRadius:
				# Calculer la distance au point d'entrée
				distanceToFood = projection - np.sqrt(foodRadius**2 - perpDistance**2)
				
				# Si cette distance est la plus proche jusqu'à présent et dans la portée
				if distanceToFood < foodDistance and distanceToFood <= maxDistance:
					foodHit = True
					foodDistance = distanceToFood
		
		# Déterminer le résultat final (la collision la plus proche)
		hit = terrainHit or creatureHit or foodHit
		
		if not hit:
			return False, maxDistance, "none"
			
		# Trouver la distance minimale parmi les collisions
		distance = maxDistance
		objectType = "none"
		
		if terrainHit and terrainDistance < distance:
			distance = terrainDistance
			objectType = "terrain"
			
		if creatureHit and creatureDistance < distance:
			distance = creatureDistance
			objectType = "creature"
			
		if foodHit and foodDistance < distance:
			distance = foodDistance
			objectType = "food"
			
		return hit, distance, objectType
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet MarineWorld en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du monde marin
		"""
		return {
			"id": self.id,
			"size": self.size.tolist(),
			"resolution": self.resolution.tolist(),
			"defaultTemperature": self.defaultTemperature,
			"defaultSalinity": self.defaultSalinity,
			"defaultLightLevel": self.defaultLightLevel,
			"seed": self.seed,
			"currentTime": self.currentTime,
			"dayLength": self.dayLength,
			"yearLength": self.yearLength,
			"terrain": self.terrain.toDict(),
			"waterProperties": self.waterProperties.toDict(),
			"zones": [zone.toDict() for zone in self.zones],
			"foodResources": [food.toDict() for food in self.foodResources],
			"maxFoodResources": self.maxFoodResources,
			"foodRegenerationRate": self.foodRegenerationRate,
			"foodConsumed": self.foodConsumed,
			"creaturesBorn": self.creaturesBorn,
			"creaturesDied": self.creaturesDied
			# Note: Les créatures et la grille d'occupation ne sont pas sérialisées ici
			# car elles sont généralement gérées séparément
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'MarineWorld':
		"""
		Crée une instance de MarineWorld à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du monde marin
			
		Returns:
			Instance de MarineWorld reconstruite
		"""
		world = cls(
			size=tuple(data["size"]),
			resolution=tuple(data["resolution"]),
			defaultTemperature=data["defaultTemperature"],
			defaultSalinity=data["defaultSalinity"],
			defaultLightLevel=data["defaultLightLevel"],
			seed=data["seed"]
		)
		
		world.id = data["id"]
		world.currentTime = data["currentTime"]
		world.dayLength = data["dayLength"]
		world.yearLength = data["yearLength"]
		
		# Reconstruire le terrain
		world.terrain = Terrain.fromDict(data["terrain"])
		
		# Reconstruire les propriétés de l'eau
		world.waterProperties = WaterProperties.fromDict(data["waterProperties"])
		
		# Reconstruire les zones
		world.zones = [EnvironmentalZone.fromDict(zone_data) for zone_data in data["zones"]]
		
		# Reconstruire les ressources alimentaires
		world.foodResources = [FoodResource.fromDict(food_data) for food_data in data["foodResources"]]
		
		world.maxFoodResources = data["maxFoodResources"]
		world.foodRegenerationRate = data["foodRegenerationRate"]
		world.foodConsumed = data["foodConsumed"]
		world.creaturesBorn = data["creaturesBorn"]
		world.creaturesDied = data["creaturesDied"]
		
		return world