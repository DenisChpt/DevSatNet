from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import uuid

from utils.serialization import Serializable


class Joint(Serializable):
	"""
	Classe représentant une articulation d'une créature marine.
	Les articulations sont les points de connexion entre les membres.
	"""
	
	def __init__(
		self, 
		position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
		parentId: Optional[str] = None,
		jointType: str = "ball",  # Types: "fixed", "hinge", "ball"
		minAngles: Optional[List[float]] = None,
		maxAngles: Optional[List[float]] = None,
		friction: float = 0.1,
		damping: float = 0.1,
		mass: float = 1.0
	) -> None:
		"""
		Initialise une nouvelle articulation.
		
		Args:
			position: Position relative au parent ou absolue si pas de parent
			parentId: ID de l'articulation parente (None si c'est la racine)
			jointType: Type d'articulation (détermine les degrés de liberté)
			minAngles: Angles minimaux pour chaque degré de liberté
			maxAngles: Angles maximaux pour chaque degré de liberté
			friction: Coefficient de friction de l'articulation
			damping: Coefficient d'amortissement de l'articulation
			mass: Masse de l'articulation
		"""
		self.id: str = str(uuid.uuid4())
		self.position: np.ndarray = np.array(position, dtype=np.float32)
		self.parentId: Optional[str] = parentId
		self.jointType: str = jointType
		
		# Initialisation des limites d'angles selon le type d'articulation
		if minAngles is None or maxAngles is None:
			if jointType == "fixed":
				self.minAngles = np.zeros(0, dtype=np.float32)
				self.maxAngles = np.zeros(0, dtype=np.float32)
			elif jointType == "hinge":
				self.minAngles = np.array([-np.pi/2], dtype=np.float32)
				self.maxAngles = np.array([np.pi/2], dtype=np.float32)
			elif jointType == "ball":
				self.minAngles = np.array([-np.pi/2, -np.pi/2, -np.pi/2], dtype=np.float32)
				self.maxAngles = np.array([np.pi/2, np.pi/2, np.pi/2], dtype=np.float32)
		else:
			self.minAngles = np.array(minAngles, dtype=np.float32)
			self.maxAngles = np.array(maxAngles, dtype=np.float32)
			
		self.friction: float = friction
		self.damping: float = damping
		self.mass: float = mass
		
		# État actuel de l'articulation
		self.currentAngles: np.ndarray = np.zeros_like(self.minAngles)
		self.angularVelocity: np.ndarray = np.zeros_like(self.minAngles)
		
		# Connexions
		self.connectedJointIds: List[str] = []
		self.connectedLimbIds: List[str] = []
		self.connectedMuscleIds: List[str] = []
		
	def addConnection(self, jointId: str) -> None:
		"""Ajoute une connexion avec une autre articulation."""
		if jointId not in self.connectedJointIds:
			self.connectedJointIds.append(jointId)
	
	def addLimb(self, limbId: str) -> None:
		"""Ajoute une connexion avec un membre."""
		if limbId not in self.connectedLimbIds:
			self.connectedLimbIds.append(limbId)
	
	def addMuscle(self, muscleId: str) -> None:
		"""Ajoute une connexion avec un muscle."""
		if muscleId not in self.connectedMuscleIds:
			self.connectedMuscleIds.append(muscleId)
			
	def setAngles(self, angles: np.ndarray) -> None:
		"""
		Définit les angles de l'articulation, en respectant les limites.
		
		Args:
			angles: Nouvelles valeurs d'angle
		"""
		if len(angles) != len(self.currentAngles):
			raise ValueError(f"Nombre d'angles incorrect: {len(angles)} ≠ {len(self.currentAngles)}")
			
		# Appliquer les limites
		clippedAngles = np.clip(angles, self.minAngles, self.maxAngles)
		self.currentAngles = clippedAngles
	
	def applyTorque(self, torques: np.ndarray, deltaTime: float) -> None:
		"""
		Applique des couples de forces à l'articulation, mettant à jour les vitesses angulaires.
		
		Args:
			torques: Couples à appliquer à chaque degré de liberté
			deltaTime: Intervalle de temps depuis la dernière mise à jour
		"""
		if len(torques) != len(self.angularVelocity):
			raise ValueError(f"Dimensions de torque incorrectes: {len(torques)} ≠ {len(self.angularVelocity)}")
			
		# Mise à jour de la vitesse angulaire (F = ma)
		# On divise par la masse pour obtenir l'accélération
		angularAcceleration = torques / self.mass
		
		# Appliquer l'amortissement (résistance au mouvement)
		self.angularVelocity = (1.0 - self.damping * deltaTime) * self.angularVelocity
		
		# Ajouter l'accélération
		self.angularVelocity += angularAcceleration * deltaTime
		
		# Mise à jour des angles
		newAngles = self.currentAngles + self.angularVelocity * deltaTime
		
		# Appliquer les limites
		self.setAngles(newAngles)
		
	def getDegreesOfFreedom(self) -> int:
		"""
		Retourne le nombre de degrés de liberté de l'articulation.
		
		Returns:
			Nombre de degrés de liberté
		"""
		return len(self.currentAngles)
		
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Joint en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de l'articulation
		"""
		return {
			"id": self.id,
			"position": self.position.tolist(),
			"parentId": self.parentId,
			"jointType": self.jointType,
			"minAngles": self.minAngles.tolist(),
			"maxAngles": self.maxAngles.tolist(),
			"currentAngles": self.currentAngles.tolist(),
			"angularVelocity": self.angularVelocity.tolist(),
			"friction": self.friction,
			"damping": self.damping,
			"mass": self.mass,
			"connectedJointIds": self.connectedJointIds,
			"connectedLimbIds": self.connectedLimbIds,
			"connectedMuscleIds": self.connectedMuscleIds
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Joint':
		"""
		Crée une instance de Joint à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de l'articulation
			
		Returns:
			Instance de Joint reconstruite
		"""
		joint = cls(
			position=tuple(data["position"]),
			parentId=data["parentId"],
			jointType=data["jointType"],
			minAngles=data["minAngles"],
			maxAngles=data["maxAngles"],
			friction=data["friction"],
			damping=data["damping"],
			mass=data["mass"]
		)
		
		joint.id = data["id"]
		joint.currentAngles = np.array(data["currentAngles"], dtype=np.float32)
		joint.angularVelocity = np.array(data["angularVelocity"], dtype=np.float32)
		joint.connectedJointIds = data["connectedJointIds"]
		joint.connectedLimbIds = data["connectedLimbIds"]
		joint.connectedMuscleIds = data["connectedMuscleIds"]
		
		return joint