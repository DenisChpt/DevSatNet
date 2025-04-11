# core/environment/terrain.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import uuid
from noise import snoise3

from utils.serialization import Serializable


class Terrain(Serializable):
	"""
	Classe représentant la topographie du fond marin.
	Génère et gère le relief sous-marin, les obstacles, etc.
	"""
	
	def __init__(
		self,
		worldSize: Tuple[float, float, float],
		resolution: Tuple[int, int, int],
		seed: Optional[int] = None,
		roughness: float = 0.5,
		featureSize: float = 100.0
	) -> None:
		"""
		Initialise le terrain marin.
		
		Args:
			worldSize: Dimensions du monde marin (x, y, z)
			resolution: Résolution de la grille pour les calculs (nx, ny, nz)
			seed: Graine pour le générateur de nombres aléatoires
			roughness: Rugosité du terrain (0.0 = lisse, 1.0 = très accidenté)
			featureSize: Taille caractéristique des formations (en unités de simulation)
		"""
		self.id: str = str(uuid.uuid4())
		self.worldSize: np.ndarray = np.array(worldSize, dtype=np.float32)
		self.resolution: np.ndarray = np.array(resolution, dtype=np.int32)
		self.seed: int = seed if seed is not None else np.random.randint(0, 100000)
		self.roughness: float = roughness
		self.featureSize: float = featureSize
		
		# Initialiser le générateur de nombres aléatoires
		self.rng: np.random.Generator = np.random.default_rng(self.seed)
		
		# Carte d'élévation (hauteur du fond marin)
		# On utilise une grille 2D car le terrain est principalement défini par sa hauteur
		self.elevationMap = np.zeros((self.resolution[0], self.resolution[2]), dtype=np.float32)
		
		# Types de terrain (sable, rocher, corail, etc.)
		self.terrainTypeMap = np.zeros((self.resolution[0], self.resolution[2]), dtype=np.int32)
		
		# Obstacles (structures plus complexes que la simple élévation)
		self.obstacles: List[Dict[str, Any]] = []
		
		# Structures spéciales (récifs, épaves, grottes, etc.)
		self.structures: List[Dict[str, Any]] = []
		
		# Générer le terrain
		self._generateTerrain()
	
	def _generateTerrain(self) -> None:
		"""
		Génère le terrain en utilisant des algorithmes de bruit procédural.
		"""
		# Générer la carte d'élévation de base avec du bruit de Perlin
		for x in range(self.resolution[0]):
			for z in range(self.resolution[2]):
				# Coordonnées normalisées
				nx = x / self.resolution[0]
				nz = z / self.resolution[2]
				
				# Bruit à plusieurs échelles (fractale)
				elevation = 0.0
				amplitude = 1.0
				frequency = 1.0
				maxValue = 0.0
				
				# Superposer plusieurs octaves de bruit
				for _ in range(5):
					elevation += amplitude * snoise3(
						nx * frequency, 
						0.5, 
						nz * frequency, 
						octaves=1, 
						persistence=0.5,
						lacunarity=2.0,
						repeatx=1024,
						repeaty=1024,
						repeatz=1024,
						base=self.seed
					)
					maxValue += amplitude
					amplitude *= 0.5
					frequency *= 2.0
				
				# Normaliser l'élévation
				elevation /= maxValue
				
				# Ajuster l'élévation selon la rugosité désirée
				elevation = elevation * self.roughness
				
				# Convertir en profondeur réelle (en pourcentage de la hauteur du monde)
				# Plus la valeur est élevée, plus le fond est profond
				depthPercentage = 0.7 + 0.3 * elevation  # Entre 70% et 100% de la profondeur maximale
				
				# Stocker l'élévation (en valeur absolue par rapport au fond)
				self.elevationMap[x, z] = depthPercentage * self.worldSize[1]
		
		# Générer quelques caractéristiques géologiques supplémentaires
		self._generateGeologicalFeatures()
		
		# Générer la carte des types de terrain
		self._generateTerrainTypes()
		
		# Générer des obstacles
		self._generateObstacles()
		
		# Générer des structures spéciales
		self._generateStructures()
	
	def _generateGeologicalFeatures(self) -> None:
		"""
		Ajoute des caractéristiques géologiques comme des canyons, des montagnes sous-marines, etc.
		"""
		# Ajouter un canyon
		canyonStartX = self.rng.integers(0, self.resolution[0] // 4)
		canyonStartZ = self.rng.integers(0, self.resolution[2])
		canyonEndX = self.rng.integers(3 * self.resolution[0] // 4, self.resolution[0])
		canyonEndZ = self.rng.integers(0, self.resolution[2])
		canyonWidth = self.rng.integers(5, max(6, self.resolution[0] // 20))
		canyonDepth = self.rng.uniform(0.1, 0.3) * self.worldSize[1]
		
		# Tracer le canyon avec un algorithme de ligne de Bresenham
		points = self._bresenhamLine(canyonStartX, canyonStartZ, canyonEndX, canyonEndZ)
		for x, z in points:
			# Appliquer la profondeur du canyon avec un profil gaussien pour la largeur
			for dx in range(-canyonWidth, canyonWidth + 1):
				for dz in range(-canyonWidth, canyonWidth + 1):
					nx = x + dx
					nz = z + dz
					if 0 <= nx < self.resolution[0] and 0 <= nz < self.resolution[2]:
						# Distance au centre du canyon
						distance = np.sqrt(dx**2 + dz**2)
						# Profil gaussien
						depthFactor = np.exp(-(distance**2) / (2 * (canyonWidth / 2)**2))
						# Appliquer la profondeur
						self.elevationMap[nx, nz] += canyonDepth * depthFactor
						# Limiter à la hauteur maximale
						self.elevationMap[nx, nz] = min(self.elevationMap[nx, nz], self.worldSize[1])
		
		# Ajouter quelques montagnes sous-marines
		numMountains = self.rng.integers(2, 5)
		for _ in range(numMountains):
			mountainX = self.rng.integers(0, self.resolution[0])
			mountainZ = self.rng.integers(0, self.resolution[2])
			mountainRadius = self.rng.integers(5, max(6, self.resolution[0] // 15))
			mountainHeight = self.rng.uniform(0.2, 0.5) * self.worldSize[1]
			
			# Appliquer la montagne avec un profil gaussien
			for dx in range(-mountainRadius * 2, mountainRadius * 2 + 1):
				for dz in range(-mountainRadius * 2, mountainRadius * 2 + 1):
					nx = mountainX + dx
					nz = mountainZ + dz
					if 0 <= nx < self.resolution[0] and 0 <= nz < self.resolution[2]:
						# Distance au centre de la montagne
						distance = np.sqrt(dx**2 + dz**2)
						# Profil gaussien
						heightFactor = np.exp(-(distance**2) / (2 * mountainRadius**2))
						# Soustraire la hauteur (car l'élévation est en fait la profondeur)
						self.elevationMap[nx, nz] -= mountainHeight * heightFactor
						# Limiter à la hauteur minimale (surface)
						self.elevationMap[nx, nz] = max(self.elevationMap[nx, nz], 0.0)
	
	def _generateTerrainTypes(self) -> None:
		"""
		Génère la carte des types de terrain.
		Types: 0 = sable, 1 = vase, 2 = rocher, 3 = corail, 4 = gravier
		"""
		# Génération basée sur la profondeur et des variations aléatoires
		for x in range(self.resolution[0]):
			for z in range(self.resolution[2]):
				# Profondeur normalisée
				depth = self.elevationMap[x, z] / self.worldSize[1]
				
				# Ajouter un bruit pour créer des variations
				noise = snoise3(
					x / (self.resolution[0] / 10), 
					0.5, 
					z / (self.resolution[2] / 10),
					octaves=2,
					persistence=0.5,
					lacunarity=2.0,
					base=self.seed + 1  # Différent du bruit d'élévation
				)
				
				# Déterminer le type de terrain
				if depth < 0.3:  # Peu profond
					if noise > 0.2:
						self.terrainTypeMap[x, z] = 3  # Corail
					else:
						self.terrainTypeMap[x, z] = 0  # Sable
				elif depth < 0.6:  # Profondeur moyenne
					if noise > 0.3:
						self.terrainTypeMap[x, z] = 2  # Rocher
					elif noise > -0.3:
						self.terrainTypeMap[x, z] = 4  # Gravier
					else:
						self.terrainTypeMap[x, z] = 0  # Sable
				else:  # Profond
					if noise > 0.4:
						self.terrainTypeMap[x, z] = 2  # Rocher
					else:
						self.terrainTypeMap[x, z] = 1  # Vase
	
	def _generateObstacles(self) -> None:
		"""
		Génère des obstacles comme des rochers, des formations coralliennes, etc.
		"""
		# Nombre d'obstacles à générer
		numObstacles = int(0.001 * self.resolution[0] * self.resolution[2])
		
		for _ in range(numObstacles):
			# Position aléatoire
			x = self.rng.uniform(0, self.worldSize[0])
			z = self.rng.uniform(0, self.worldSize[2])
			
			# Déterminer la profondeur à cette position
			ix = int(x / self.worldSize[0] * self.resolution[0])
			iz = int(z / self.worldSize[2] * self.resolution[2])
			ix = max(0, min(ix, self.resolution[0] - 1))
			iz = max(0, min(iz, self.resolution[2] - 1))
			
			depth = self.elevationMap[ix, iz]
			y = depth  # L'obstacle est posé sur le fond
			
			# Type d'obstacle selon la profondeur
			obstacleType = ""
			size = 1.0
			
			if depth < 0.3 * self.worldSize[1]:  # Peu profond
				if self.rng.random() < 0.7:
					obstacleType = "coral_formation"
					size = self.rng.uniform(1.0, 5.0)
				else:
					obstacleType = "rock"
					size = self.rng.uniform(0.5, 3.0)
			elif depth < 0.6 * self.worldSize[1]:  # Profondeur moyenne
				if self.rng.random() < 0.4:
					obstacleType = "rock_formation"
					size = self.rng.uniform(2.0, 8.0)
				else:
					obstacleType = "rock"
					size = self.rng.uniform(1.0, 4.0)
			else:  # Profond
				obstacleType = "deep_sea_vent"
				size = self.rng.uniform(3.0, 10.0)
			
			# Ajouter l'obstacle
			obstacle = {
				"id": str(uuid.uuid4()),
				"type": obstacleType,
				"position": (x, y, z),
				"size": size,
				"orientation": self.rng.uniform(0, 2 * np.pi)
			}
			
			self.obstacles.append(obstacle)
	
	def _generateStructures(self) -> None:
		"""
		Génère des structures spéciales comme des récifs, des épaves, des grottes, etc.
		"""
		# Nombre de structures à générer
		numStructures = self.rng.integers(3, 8)
		
		structureTypes = ["reef", "shipwreck", "underwater_cave", "kelp_forest", "abyss"]
		
		for _ in range(numStructures):
			# Position aléatoire
			x = self.rng.uniform(0, self.worldSize[0])
			z = self.rng.uniform(0, self.worldSize[2])
			
			# Déterminer la profondeur à cette position
			ix = int(x / self.worldSize[0] * self.resolution[0])
			iz = int(z / self.worldSize[2] * self.resolution[2])
			ix = max(0, min(ix, self.resolution[0] - 1))
			iz = max(0, min(iz, self.resolution[2] - 1))
			
			depth = self.elevationMap[ix, iz]
			y = depth  # La structure est posée sur le fond
			
			# Type de structure selon la profondeur
			structureType = ""
			size = 1.0
			
			if depth < 0.2 * self.worldSize[1]:  # Très peu profond
				structureType = "reef"
				size = self.rng.uniform(10.0, 30.0)
			elif depth < 0.4 * self.worldSize[1]:  # Peu profond
				if self.rng.random() < 0.6:
					structureType = "kelp_forest"
					size = self.rng.uniform(15.0, 40.0)
				else:
					structureType = "shipwreck"
					size = self.rng.uniform(5.0, 15.0)
			elif depth < 0.7 * self.worldSize[1]:  # Profondeur moyenne
				if self.rng.random() < 0.7:
					structureType = "underwater_cave"
					size = self.rng.uniform(10.0, 25.0)
				else:
					structureType = "shipwreck"
					size = self.rng.uniform(5.0, 15.0)
			else:  # Profond
				structureType = "abyss"
				size = self.rng.uniform(20.0, 50.0)
			
			# Ajouter la structure
			structure = {
				"id": str(uuid.uuid4()),
				"type": structureType,
				"position": (x, y, z),
				"size": size,
				"orientation": self.rng.uniform(0, 2 * np.pi),
				"properties": {}  # Propriétés spécifiques selon le type
			}
			
			# Ajouter des propriétés spécifiques selon le type
			if structureType == "reef":
				structure["properties"]["coral_coverage"] = self.rng.uniform(0.5, 1.0)
				structure["properties"]["biodiversity"] = self.rng.uniform(0.7, 1.0)
			elif structureType == "shipwreck":
				structure["properties"]["age"] = self.rng.uniform(10, 200)  # Âge en années
				structure["properties"]["decay"] = self.rng.uniform(0.2, 0.9)
			elif structureType == "underwater_cave":
				structure["properties"]["depth"] = self.rng.uniform(10, 50)  # Profondeur de la grotte
				structure["properties"]["complexity"] = self.rng.uniform(0.3, 1.0)
			elif structureType == "kelp_forest":
				structure["properties"]["density"] = self.rng.uniform(0.5, 1.0)
				structure["properties"]["height"] = self.rng.uniform(5, 20)  # Hauteur en mètres
			elif structureType == "abyss":
				structure["properties"]["pressure"] = self.rng.uniform(100, 300)  # Pression en bars
				structure["properties"]["thermal_activity"] = self.rng.uniform(0, 1.0)
			
			self.structures.append(structure)
	
	def getElevationAt(self, x: float, z: float) -> float:
		"""
		Retourne l'élévation du terrain à une position donnée.
		
		Args:
			x: Coordonnée X dans le monde
			z: Coordonnée Z dans le monde
			
		Returns:
			Élévation (profondeur) en unités absolues
		"""
		# Convertir les coordonnées mondiales en indices de grille
		ix = int(x / self.worldSize[0] * self.resolution[0])
		iz = int(z / self.worldSize[2] * self.resolution[2])
		
		# Limiter aux indices valides
		ix = max(0, min(ix, self.resolution[0] - 1))
		iz = max(0, min(iz, self.resolution[2] - 1))
		
		return self.elevationMap[ix, iz]
	
	def getTerrainTypeAt(self, x: float, z: float) -> int:
		"""
		Retourne le type de terrain à une position donnée.
		
		Args:
			x: Coordonnée X dans le monde
			z: Coordonnée Z dans le monde
			
		Returns:
			Type de terrain (0 = sable, 1 = vase, 2 = rocher, 3 = corail, 4 = gravier)
		"""
		# Convertir les coordonnées mondiales en indices de grille
		ix = int(x / self.worldSize[0] * self.resolution[0])
		iz = int(z / self.worldSize[2] * self.resolution[2])
		
		# Limiter aux indices valides
		ix = max(0, min(ix, self.resolution[0] - 1))
		iz = max(0, min(iz, self.resolution[2] - 1))
		
		return self.terrainTypeMap[ix, iz]
	
	def getObstaclesNear(self, position: np.ndarray, radius: float) -> List[Dict[str, Any]]:
		"""
		Retourne les obstacles proches d'une position donnée.
		
		Args:
			position: Position centrale pour la recherche
			radius: Rayon de recherche
			
		Returns:
			Liste des obstacles dans le rayon spécifié
		"""
		nearbyObstacles = []
		
		for obstacle in self.obstacles:
			obstaclePos = np.array(obstacle["position"])
			distance = np.linalg.norm(obstaclePos - position)
			
			if distance <= radius + obstacle["size"]:
				nearbyObstacles.append(obstacle)
		
		return nearbyObstacles
	
	def getStructuresNear(self, position: np.ndarray, radius: float) -> List[Dict[str, Any]]:
		"""
		Retourne les structures spéciales proches d'une position donnée.
		
		Args:
			position: Position centrale pour la recherche
			radius: Rayon de recherche
			
		Returns:
			Liste des structures dans le rayon spécifié
		"""
		nearbyStructures = []
		
		for structure in self.structures:
			structurePos = np.array(structure["position"])
			distance = np.linalg.norm(structurePos - position)
			
			if distance <= radius + structure["size"]:
				nearbyStructures.append(structure)
		
		return nearbyStructures
	
	def rayCast(self, origin: np.ndarray, direction: np.ndarray, maxDistance: float) -> Tuple[bool, float]:
		"""
		Lance un rayon et détecte l'intersection avec le terrain.
		
		Args:
			origin: Point d'origine du rayon
			direction: Direction du rayon (normalisée)
			maxDistance: Distance maximale de recherche
			
		Returns:
			Tuple (hit, distance) indiquant s'il y a eu une intersection et à quelle distance
		"""
		# Pas d'échantillonnage
		steps = 100
		stepSize = maxDistance / steps
		
		# Point courant
		point = origin.copy()
		
		for _ in range(steps):
			# Vérifier si le point est sous le niveau du terrain
			x, y, z = point
			
			# Si le point est hors des limites du monde, pas d'intersection
			if (x < 0 or x >= self.worldSize[0] or
				y < 0 or y >= self.worldSize[1] or
				z < 0 or z >= self.worldSize[2]):
				return False, maxDistance
			
			# Obtenir l'élévation (profondeur) du terrain à cette position
			elevation = self.getElevationAt(x, z)
			
			# Si le point est sous le fond marin, il y a intersection
			if y >= elevation:
				# Calculer la distance parcourue
				distance = np.linalg.norm(point - origin)
				return True, distance
			
			# Avancer le long du rayon
			point += direction * stepSize
		
		# Aucune intersection trouvée
		return False, maxDistance
	
	def _bresenhamLine(self, x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
		"""
		Algorithme de Bresenham pour tracer une ligne entre deux points.
		
		Args:
			x0, y0: Coordonnées du point de départ
			x1, y1: Coordonnées du point d'arrivée
			
		Returns:
			Liste des points sur la ligne
		"""
		points = []
		dx = abs(x1 - x0)
		dy = abs(y1 - y0)
		sx = 1 if x0 < x1 else -1
		sy = 1 if y0 < y1 else -1
		err = dx - dy
		
		while True:
			points.append((x0, y0))
			if x0 == x1 and y0 == y1:
				break
			e2 = 2 * err
			if e2 > -dy:
				err -= dy
				x0 += sx
			if e2 < dx:
				err += dx
				y0 += sy
		
		return points
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Terrain en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du terrain
		"""
		return {
			"id": self.id,
			"worldSize": self.worldSize.tolist(),
			"resolution": self.resolution.tolist(),
			"seed": self.seed,
			"roughness": self.roughness,
			"featureSize": self.featureSize,
			"elevationMap": self.elevationMap.tolist(),
			"terrainTypeMap": self.terrainTypeMap.tolist(),
			"obstacles": self.obstacles,
			"structures": self.structures
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Terrain':
		"""
		Crée une instance de Terrain à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du terrain
			
		Returns:
			Instance de Terrain reconstruite
		"""
		terrain = cls(
			worldSize=tuple(data["worldSize"]),
			resolution=tuple(data["resolution"]),
			seed=data["seed"],
			roughness=data["roughness"],
			featureSize=data["featureSize"]
		)
		
		terrain.id = data["id"]
		terrain.elevationMap = np.array(data["elevationMap"], dtype=np.float32)
		terrain.terrainTypeMap = np.array(data["terrainTypeMap"], dtype=np.int32)
		terrain.obstacles = data["obstacles"]
		terrain.structures = data["structures"]
		
		return terrain