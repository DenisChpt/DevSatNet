# core/environment/resources.py
from typing import Dict, Tuple, Any, Optional
import numpy as np
import uuid
import time

from utils.serialization import Serializable


class FoodResource(Serializable):
	"""
	Classe représentant une ressource alimentaire dans l'environnement marin.
	"""
	
	def __init__(
		self,
		position: Tuple[float, float, float],
		energyValue: float = 10.0,
		foodType: str = "plankton",  # Types: "algae", "plankton", "small_fish", "detritus"
		size: float = 1.0,
		respawnTime: float = 100.0,
		movable: bool = False,
		nutritionalValue: Dict[str, float] = None
	) -> None:
		"""
		Initialise une nouvelle ressource alimentaire.
		
		Args:
			position: Position (x, y, z) dans le monde
			energyValue: Valeur énergétique de la ressource
			foodType: Type de nourriture
			size: Taille de la ressource
			respawnTime: Temps avant réapparition après consommation
			movable: Indique si la ressource peut se déplacer (pour les petits poissons, etc.)
			nutritionalValue: Valeurs nutritionnelles spécifiques
		"""
		self.id: str = str(uuid.uuid4())
		self.position: Tuple[float, float, float] = position
		self.energyValue: float = energyValue
		self.foodType: str = foodType
		self.size: float = size
		self.respawnTime: float = respawnTime
		self.movable: bool = movable
		
		# État actuel
		self.isConsumed: bool = False
		self.consumedBy: Optional[str] = None
		self.consumedTime: float = 0.0
		self.velocity: np.ndarray = np.zeros(3, dtype=np.float32)
		
		# Valeurs nutritionnelles
		if nutritionalValue is None:
			self.nutritionalValue = self._defaultNutritionalValue()
		else:
			self.nutritionalValue = nutritionalValue
	
	def _defaultNutritionalValue(self) -> Dict[str, float]:
		"""
		Définit les valeurs nutritionnelles par défaut selon le type de nourriture.
		
		Returns:
			Dictionnaire des valeurs nutritionnelles
		"""
		if self.foodType == "algae":
			return {
				"protein": 0.2,
				"fat": 0.1,
				"carbohydrate": 0.7,
				"minerals": 0.4,
				"vitamins": 0.5
			}
		elif self.foodType == "plankton":
			return {
				"protein": 0.6,
				"fat": 0.3,
				"carbohydrate": 0.3,
				"minerals": 0.5,
				"vitamins": 0.7
			}
		elif self.foodType == "small_fish":
			return {
				"protein": 0.8,
				"fat": 0.5,
				"carbohydrate": 0.1,
				"minerals": 0.3,
				"vitamins": 0.4
			}
		elif self.foodType == "detritus":
			return {
				"protein": 0.3,
				"fat": 0.2,
				"carbohydrate": 0.2,
				"minerals": 0.8,
				"vitamins": 0.1
			}
		else:
			# Valeurs par défaut pour les types inconnus
			return {
				"protein": 0.4,
				"fat": 0.3,
				"carbohydrate": 0.3,
				"minerals": 0.4,
				"vitamins": 0.4
			}
	
	def consume(self, consumerId: str) -> float:
		"""
		Consomme la ressource alimentaire.
		
		Args:
			consumerId: ID de la créature qui consomme la ressource
			
		Returns:
			Valeur énergétique obtenue
		"""
		if self.isConsumed:
			return 0.0
			
		self.isConsumed = True
		self.consumedBy = consumerId
		self.consumedTime = time.time()
		
		return self.energyValue
	
	def update(self, deltaTime: float, worldSize: np.ndarray, waterCurrent: np.ndarray) -> None:
		"""
		Met à jour l'état de la ressource pour un pas de temps.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
			worldSize: Dimensions du monde marin
			waterCurrent: Vecteur du courant d'eau à la position actuelle
		"""
		if self.isConsumed:
			return
			
		if self.movable:
			# Ajouter l'influence du courant
			self.velocity += waterCurrent * 0.1 * deltaTime
			
			# Amortissement naturel
			self.velocity *= (1.0 - 0.05 * deltaTime)
			
			# Pour les petits poissons, ajouter un mouvement aléatoire
			if self.foodType == "small_fish":
				randomMovement = np.random.normal(0, 0.1, 3)
				self.velocity += randomMovement * deltaTime
				
				# Limiter la vitesse maximale
				speedLimit = 2.0
				speed = np.linalg.norm(self.velocity)
				if speed > speedLimit:
					self.velocity = self.velocity / speed * speedLimit
			
			# Mettre à jour la position
			newPosition = np.array(self.position) + self.velocity * deltaTime
			
			# Limites du monde
			newPosition[0] = max(0, min(newPosition[0], worldSize[0]))
			newPosition[1] = max(0, min(newPosition[1], worldSize[1]))
			newPosition[2] = max(0, min(newPosition[2], worldSize[2]))
			
			self.position = tuple(newPosition)
	
	def canRespawn(self, currentTime: float) -> bool:
		"""
		Vérifie si la ressource peut réapparaître.
		
		Args:
			currentTime: Temps actuel
			
		Returns:
			True si la ressource peut réapparaître, False sinon
		"""
		if not self.isConsumed:
			return False
			
		return (currentTime - self.consumedTime) >= self.respawnTime
	
	def respawn(self, newPosition: Optional[Tuple[float, float, float]] = None) -> None:
		"""
		Fait réapparaître la ressource alimentaire.
		
		Args:
			newPosition: Nouvelle position (optionnelle)
		"""
		self.isConsumed = False
		self.consumedBy = None
		
		if newPosition is not None:
			self.position = newPosition
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet FoodResource en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de la ressource alimentaire
		"""
		return {
			"id": self.id,
			"position": self.position,
			"energyValue": self.energyValue,
			"foodType": self.foodType,
			"size": self.size,
			"respawnTime": self.respawnTime,
			"movable": self.movable,
			"isConsumed": self.isConsumed,
			"consumedBy": self.consumedBy,
			"consumedTime": self.consumedTime,
			"velocity": self.velocity.tolist() if isinstance(self.velocity, np.ndarray) else self.velocity,
			"nutritionalValue": self.nutritionalValue
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'FoodResource':
		"""
		Crée une instance de FoodResource à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de la ressource alimentaire
			
		Returns:
			Instance de FoodResource reconstruite
		"""
		food = cls(
			position=data["position"],
			energyValue=data["energyValue"],
			foodType=data["foodType"],
			size=data["size"],
			respawnTime=data["respawnTime"],
			movable=data["movable"],
			nutritionalValue=data["nutritionalValue"]
		)
		
		food.id = data["id"]
		food.isConsumed = data["isConsumed"]
		food.consumedBy = data["consumedBy"]
		food.consumedTime = data["consumedTime"]
		
		# Reconstruire la vélocité
		if "velocity" in data:
			if isinstance(data["velocity"], list):
				food.velocity = np.array(data["velocity"], dtype=np.float32)
			else:
				food.velocity = data["velocity"]
		
		return food