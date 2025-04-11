from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import uuid

from utils.serialization import Serializable


class Limb(Serializable):
	"""
	Classe représentant un membre ou appendice d'une créature marine.
	Les membres connectent les articulations et peuvent avoir différentes propriétés
	comme la forme, la rigidité et la flottabilité.
	"""
	
	def __init__(
		self,
		startJointId: str,
		endJointId: Optional[str] = None,
		shape: str = "cylinder",  # Types: "cylinder", "fin", "tentacle", "paddle"
		length: float = 1.0,
		width: float = 0.2,
		density: float = 1.0,  # Densité par rapport à l'eau (1.0 = neutre)
		dragCoefficient: float = 0.5,
		rigidity: float = 0.8,
		color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
	) -> None:
		"""
		Initialise un nouveau membre.
		
		Args:
			startJointId: ID de l'articulation de départ
			endJointId: ID de l'articulation d'arrivée (None si c'est un appendice terminal)
			shape: Forme du membre
			length: Longueur du membre
			width: Largeur/diamètre du membre
			density: Densité relative à l'eau (flottabilité)
			dragCoefficient: Coefficient de traînée dans l'eau
			rigidity: Rigidité du membre (0.0 = très flexible, 1.0 = rigide)
			color: Couleur RGB (valeurs entre 0 et 1)
		"""
		self.id: str = str(uuid.uuid4())
		self.startJointId: str = startJointId
		self.endJointId: Optional[str] = endJointId
		self.shape: str = shape
		self.length: float = length
		self.width: float = width
		self.density: float = density
		self.dragCoefficient: float = dragCoefficient
		self.rigidity: float = rigidity
		self.color: np.ndarray = np.array(color, dtype=np.float32)
		
		# Propriétés calculées
		self.volume: float = self.calculateVolume()
		self.mass: float = self.volume * self.density
		self.crossSectionalArea: float = self.calculateCrossSectionalArea()
		
		# État actuel
		self.deformation: float = 0.0  # Déformation actuelle (0.0 = forme normale)
		self.velocity: np.ndarray = np.zeros(3, dtype=np.float32)
		self.force: np.ndarray = np.zeros(3, dtype=np.float32)
	
	def calculateVolume(self) -> float:
		"""
		Calcule le volume du membre en fonction de sa forme.
		
		Returns:
			Volume en unités cubiques
		"""
		if self.shape == "cylinder":
			return np.pi * (self.width/2)**2 * self.length
		elif self.shape == "fin" or self.shape == "paddle":
			# Approximation simplifiée pour une nageoire plate
			return self.length * self.width * 0.1  # Épaisseur de 10% de la largeur
		elif self.shape == "tentacle":
			# Tentacule conique
			return np.pi * (self.width/2)**2 * self.length / 3
		else:
			# Forme par défaut
			return np.pi * (self.width/2)**2 * self.length
	
	def calculateCrossSectionalArea(self) -> float:
		"""
		Calcule l'aire de la section transversale pour le calcul de la traînée.
		
		Returns:
			Aire de la section en unités carrées
		"""
		if self.shape == "cylinder" or self.shape == "tentacle":
			return np.pi * (self.width/2)**2
		elif self.shape == "fin" or self.shape == "paddle":
			return self.length * self.width
		else:
			# Forme par défaut
			return np.pi * (self.width/2)**2
	
	def calculateBuoyancy(self, waterDensity: float = 1.0) -> float:
		"""
		Calcule la force de flottabilité du membre.
		
		Args:
			waterDensity: Densité de l'eau à la position actuelle
			
		Returns:
			Force de flottabilité en unités de force
		"""
		# Formule: Fb = ρ * g * V
		# où ρ est la densité du fluide, g l'accélération due à la gravité, et V le volume immergé
		gravityAcceleration = 9.81
		displacedVolume = self.volume
		buoyancyForce = waterDensity * gravityAcceleration * displacedVolume
		
		# Force nette = flottabilité - poids
		netBuoyancy = buoyancyForce - (self.mass * gravityAcceleration)
		
		return netBuoyancy
	
	def calculateDrag(self, relativeVelocity: np.ndarray, waterDensity: float = 1.0) -> np.ndarray:
		"""
		Calcule la force de traînée hydrodynamique.
		
		Args:
			relativeVelocity: Vitesse relative du membre par rapport à l'eau
			waterDensity: Densité de l'eau à la position actuelle
			
		Returns:
			Vecteur de force de traînée
		"""
		# Formule: Fd = 0.5 * ρ * v² * Cd * A
		# où ρ est la densité du fluide, v la vitesse, Cd le coefficient de traînée et A l'aire de la section
		
		velocityMagnitude = np.linalg.norm(relativeVelocity)
		if velocityMagnitude < 1e-6:  # Éviter la division par zéro
			return np.zeros(3, dtype=np.float32)
			
		dragMagnitude = 0.5 * waterDensity * velocityMagnitude**2 * self.dragCoefficient * self.crossSectionalArea
		
		# La direction de la traînée est opposée à celle du mouvement
		dragDirection = -relativeVelocity / velocityMagnitude
		dragForce = dragDirection * dragMagnitude
		
		return dragForce
	
	def calculatePropulsionForce(self, muscleActivation: float, waterDensity: float = 1.0) -> np.ndarray:
		"""
		Calcule la force de propulsion générée par le mouvement du membre.
		
		Args:
			muscleActivation: Niveau d'activation du muscle (-1.0 à 1.0)
			waterDensity: Densité de l'eau à la position actuelle
			
		Returns:
			Vecteur de force de propulsion
		"""
		# La propulsion dépend de la forme et de l'efficacité du membre
		
		# Efficacité de propulsion selon la forme
		propulsionEfficiency = {
			"cylinder": 0.3,
			"fin": 0.8,
			"tentacle": 0.5,
			"paddle": 0.7
		}.get(self.shape, 0.3)
		
		# Calcul de la force de base
		baseForceMagnitude = abs(muscleActivation) * self.crossSectionalArea * waterDensity * propulsionEfficiency
		
		# La direction dépend du signe de l'activation
		forceDirection = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # Direction par défaut (z positif)
		if muscleActivation < 0:
			forceDirection = -forceDirection
			
		propulsionForce = forceDirection * baseForceMagnitude
		
		return propulsionForce
	
	def update(self, deltaTime: float, waterDensity: float, waterVelocity: np.ndarray) -> None:
		"""
		Met à jour l'état du membre pour un pas de temps.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
			waterDensity: Densité de l'eau à la position actuelle
			waterVelocity: Vitesse de l'eau à la position actuelle
		"""
		# Calculer la flottabilité
		buoyancy = self.calculateBuoyancy(waterDensity)
		buoyancyForce = np.array([0.0, buoyancy, 0.0])  # Force vers le haut
		
		# Calculer la traînée
		relativeVelocity = self.velocity - waterVelocity
		dragForce = self.calculateDrag(relativeVelocity, waterDensity)
		
		# Calculer la force totale
		self.force = buoyancyForce + dragForce
		
		# Mise à jour de la vitesse (F = ma)
		acceleration = self.force / max(0.001, self.mass)  # Éviter la division par zéro
		self.velocity += acceleration * deltaTime
		
		# Amortissement de la déformation
		self.deformation *= (1.0 - self.rigidity * deltaTime)
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Limb en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du membre
		"""
		return {
			"id": self.id,
			"startJointId": self.startJointId,
			"endJointId": self.endJointId,
			"shape": self.shape,
			"length": self.length,
			"width": self.width,
			"density": self.density,
			"dragCoefficient": self.dragCoefficient,
			"rigidity": self.rigidity,
			"color": self.color.tolist(),
			"volume": self.volume,
			"mass": self.mass,
			"crossSectionalArea": self.crossSectionalArea,
			"deformation": self.deformation,
			"velocity": self.velocity.tolist(),
			"force": self.force.tolist()
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Limb':
		"""
		Crée une instance de Limb à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du membre
			
		Returns:
			Instance de Limb reconstruite
		"""
		limb = cls(
			startJointId=data["startJointId"],
			endJointId=data["endJointId"],
			shape=data["shape"],
			length=data["length"],
			width=data["width"],
			density=data["density"],
			dragCoefficient=data["dragCoefficient"],
			rigidity=data["rigidity"],
			color=tuple(data["color"])
		)
		
		limb.id = data["id"]
		limb.deformation = data["deformation"]
		limb.velocity = np.array(data["velocity"], dtype=np.float32)
		limb.force = np.array(data["force"], dtype=np.float32)
		
		return limb
