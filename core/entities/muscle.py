from typing import Dict, Any, List, Optional
import numpy as np
import uuid

from utils.serialization import Serializable


class Muscle(Serializable):
	"""
	Classe représentant un muscle connectant des articulations d'une créature marine.
	Les muscles génèrent des forces qui permettent aux créatures de se déplacer.
	"""
	
	def __init__(
		self,
		jointIds: List[str],
		muscleType: str = "linear",  # Types: "linear", "circular", "spring"
		strength: float = 1.0,
		contractionRate: float = 1.0,
		efficiency: float = 1.0,
		maxForce: float = 10.0,
		recoveryRate: float = 0.5
	) -> None:
		"""
		Initialise un nouveau muscle.
		
		Args:
			jointIds: Liste des IDs d'articulations connectées par ce muscle
			muscleType: Type de muscle (détermine le modèle de force)
			strength: Force relative du muscle
			contractionRate: Vitesse de contraction du muscle
			efficiency: Efficacité énergétique du muscle
			maxForce: Force maximale que le muscle peut exercer
			recoveryRate: Taux de récupération après une contraction
		"""
		self.id: str = str(uuid.uuid4())
		self.jointIds: List[str] = jointIds
		self.muscleType: str = muscleType
		self.strength: float = strength
		self.contractionRate: float = contractionRate
		self.efficiency: float = efficiency
		self.maxForce: float = maxForce
		self.recoveryRate: float = recoveryRate
		
		# État actuel
		self.activation: float = 0.0  # Niveau d'activation (-1.0 à 1.0)
		self.fatigue: float = 0.0  # Niveau de fatigue (0.0 à 1.0)
		self.length: float = 1.0  # Longueur actuelle (unités arbitraires)
		self.baseLength: float = 1.0  # Longueur au repos
		
		# Configuration supplémentaire
		self.activationCurve: str = "linear"  # Types: "linear", "sigmoid", "exponential"
		self.fatigueRate: float = 0.1  # Taux d'accumulation de la fatigue
	
	def activate(self, activation: float) -> float:
		"""
		Active le muscle avec un niveau spécifié et retourne la consommation d'énergie.
		
		Args:
			activation: Niveau d'activation désiré (-1.0 à 1.0)
			
		Returns:
			Quantité d'énergie consommée par cette activation
		"""
		# Normaliser l'activation entre -1 et 1
		normalizedActivation = np.clip(activation, -1.0, 1.0)
		
		# Calculer l'activation effective en tenant compte de la fatigue
		effectiveStrength = self.strength * (1.0 - self.fatigue)
		self.activation = normalizedActivation * effectiveStrength
		
		# Calculer la consommation d'énergie
		# Plus d'énergie est consommée pour une activation plus élevée et une efficacité plus faible
		energyConsumption = abs(self.activation) * (2.0 - self.efficiency) * (1.0 + self.fatigue * 0.5)
		
		# Augmenter la fatigue proportionnellement à l'activation
		self.fatigue += abs(self.activation) * self.fatigueRate
		self.fatigue = min(1.0, self.fatigue)  # Limiter à 1.0
		
		return energyConsumption
	
	def update(self, deltaTime: float) -> None:
		"""
		Met à jour l'état du muscle pour un pas de temps.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
		"""
		# Récupération de la fatigue
		self.fatigue = max(0.0, self.fatigue - self.recoveryRate * deltaTime)
		
		# Décroissance naturelle de l'activation
		self.activation *= max(0.0, 1.0 - deltaTime)
	
	def calculateForce(self) -> float:
		"""
		Calcule la force générée par le muscle en fonction de son type et de son activation.
		
		Returns:
			Force générée par le muscle
		"""
		# Force de base proportionnelle à l'activation
		baseForce = self.activation * self.maxForce
		
		# Ajuster la force selon le type de muscle et sa longueur
		if self.muscleType == "linear":
			# Force constante quelle que soit la longueur
			force = baseForce
		elif self.muscleType == "spring":
			# Force proportionnelle à l'étirement (loi de Hooke)
			stretch = self.length / self.baseLength - 1.0
			force = baseForce + stretch * self.strength * self.maxForce * 0.5
		elif self.muscleType == "circular":
			# Force qui varie sinusoïdalement avec la longueur
			normalizedLength = self.length / self.baseLength
			forceFactor = np.sin(normalizedLength * np.pi)
			force = baseForce * forceFactor
		else:
			force = baseForce
			
		return force
	
	def applyForceToJoints(self, joints: Dict[str, Any]) -> None:
		"""
		Applique la force générée par le muscle aux articulations connectées.
		
		Args:
			joints: Dictionnaire des articulations (id -> objet Joint)
		"""
		if len(self.jointIds) < 2:
			return  # Un muscle a besoin d'au moins deux articulations
			
		# Calculer la force totale
		force = self.calculateForce()
		
		# Distribuer la force entre les articulations
		forcePerJoint = force / (len(self.jointIds) - 1)
		
		# Appliquer la force aux articulations (sauf la première comme point d'ancrage)
		for i in range(1, len(self.jointIds)):
			jointId = self.jointIds[i]
			if jointId in joints:
				# La direction et l'ampleur de la force dépendent de la configuration
				# Ici, on simplifie en appliquant un couple dans une seule dimension
				torque = np.array([forcePerJoint, 0.0, 0.0])
				joints[jointId].applyTorque(torque, 0.01)  # Petit pas de temps
	
	def updateLength(self, joints: Dict[str, Any]) -> None:
		"""
		Met à jour la longueur du muscle en fonction de la position des articulations.
		
		Args:
			joints: Dictionnaire des articulations (id -> objet Joint)
		"""
		if len(self.jointIds) < 2:
			return
			
		# Calculer la longueur en sommant les distances entre les articulations consécutives
		totalLength = 0.0
		for i in range(len(self.jointIds) - 1):
			if self.jointIds[i] in joints and self.jointIds[i+1] in joints:
				jointA = joints[self.jointIds[i]]
				jointB = joints[self.jointIds[i+1]]
				
				# Calculer la distance entre les deux articulations
				distance = np.linalg.norm(jointB.position - jointA.position)
				totalLength += distance
				
		self.length = totalLength
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Muscle en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du muscle
		"""
		return {
			"id": self.id,
			"jointIds": self.jointIds,
			"muscleType": self.muscleType,
			"strength": self.strength,
			"contractionRate": self.contractionRate,
			"efficiency": self.efficiency,
			"maxForce": self.maxForce,
			"recoveryRate": self.recoveryRate,
			"activation": self.activation,
			"fatigue": self.fatigue,
			"length": self.length,
			"baseLength": self.baseLength,
			"activationCurve": self.activationCurve,
			"fatigueRate": self.fatigueRate
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Muscle':
		"""
		Crée une instance de Muscle à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du muscle
			
		Returns:
			Instance de Muscle reconstruite
		"""
		muscle = cls(
			jointIds=data["jointIds"],
			muscleType=data["muscleType"],
			strength=data["strength"],
			contractionRate=data["contractionRate"],
			efficiency=data["efficiency"],
			maxForce=data["maxForce"],
			recoveryRate=data["recoveryRate"]
		)
		
		muscle.id = data["id"]
		muscle.activation = data["activation"]
		muscle.fatigue = data["fatigue"]
		muscle.length = data["length"]
		muscle.baseLength = data["baseLength"]
		muscle.activationCurve = data["activationCurve"]
		muscle.fatigueRate = data["fatigueRate"]
		
		return muscle