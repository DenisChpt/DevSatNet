# core/physics/energy_model.py
from typing import Dict, Any, Optional
import numpy as np

from utils.serialization import Serializable


class EnergyModel(Serializable):
	"""
	Modèle d'énergie pour les créatures marines.
	Calcule la consommation d'énergie en fonction des actions et mouvements.
	"""
	
	def __init__(
		self,
		basalRate: float = 1.0,
		swimmingEfficiency: float = 1.0,
		metabolicMultiplier: float = 1.0,
		sizeScalingExponent: float = 0.75,  # Exposant de la loi de Kleiber
		temperatureCoefficient: float = 0.08,  # Coefficient Q10 approximatif
		environmentalInfluence: Dict[str, float] = None
	) -> None:
		"""
		Initialise un nouveau modèle d'énergie.
		
		Args:
			basalRate: Taux métabolique de base (énergie consommée au repos)
			swimmingEfficiency: Efficacité de nage (ratio énergie utile / consommée)
			metabolicMultiplier: Multiplicateur global du métabolisme
			sizeScalingExponent: Exposant pour l'échelle métabolique selon la taille
			temperatureCoefficient: Coefficient d'influence de la température
			environmentalInfluence: Dictionnaire des facteurs d'influence environnementale
		"""
		self.basalRate: float = basalRate
		self.swimmingEfficiency: float = swimmingEfficiency
		self.metabolicMultiplier: float = metabolicMultiplier
		self.sizeScalingExponent: float = sizeScalingExponent
		self.temperatureCoefficient: float = temperatureCoefficient
		
		# Facteurs d'influence environnementale
		if environmentalInfluence is None:
			self.environmentalInfluence = {
				"pressure": 0.05,        # Influence de la pression
				"oxygen": 0.2,           # Influence du niveau d'oxygène
				"salinity": 0.05,        # Influence de la salinité
				"toxicity": 0.1,         # Influence de la toxicité
				"current_speed": 0.15    # Influence de la vitesse du courant
			}
		else:
			self.environmentalInfluence = environmentalInfluence
	
	def calculateBasalMetabolicRate(self, size: float, temperature: float) -> float:
		"""
		Calcule le taux métabolique de base selon la taille et la température.
		
		Args:
			size: Taille de la créature
			temperature: Température de l'environnement en °C
			
		Returns:
			Taux métabolique de base en unités d'énergie par unité de temps
		"""
		# Loi de Kleiber: le métabolisme est proportionnel à la masse^0.75
		# On utilise la taille comme approximation de la masse
		sizeComponent = size ** self.sizeScalingExponent
		
		# Règle du Q10: le métabolisme augmente d'un facteur Q10 tous les 10°C
		# On utilise 20°C comme température de référence
		temperatureComponent = 2.0 ** ((temperature - 20.0) * self.temperatureCoefficient)
		
		# Calculer le taux métabolique de base
		bmr = self.basalRate * sizeComponent * temperatureComponent * self.metabolicMultiplier
		
		return bmr
	
	def calculateMovementEnergyCost(
		self,
		velocity: np.ndarray,
		acceleration: np.ndarray,
		size: float,
		dragCoefficient: float,
		waterDensity: float
	) -> float:
		"""
		Calcule le coût énergétique du mouvement.
		
		Args:
			velocity: Vecteur de vitesse
			acceleration: Vecteur d'accélération
			size: Taille de la créature
			dragCoefficient: Coefficient de traînée
			waterDensity: Densité de l'eau
			
		Returns:
			Coût énergétique du mouvement en unités d'énergie
		"""
		# Magnitude de la vitesse et de l'accélération
		speed = np.linalg.norm(velocity)
		accelMagnitude = np.linalg.norm(acceleration)
		
		# Surface frontale approximative (proportionnelle au carré de la taille)
		frontalArea = np.pi * (size/2)**2
		
		# Calcul de la traînée
		drag = 0.5 * waterDensity * speed**2 * dragCoefficient * frontalArea
		
		# Puissance nécessaire pour vaincre la traînée
		powerAgainstDrag = drag * speed
		
		# Puissance nécessaire pour l'accélération
		# P = F * v = m * a * v
		mass = size**3 * waterDensity  # Approximation de la masse
		powerForAcceleration = mass * accelMagnitude * speed
		
		# Puissance totale
		totalPower = powerAgainstDrag + powerForAcceleration
		
		# Tenir compte de l'efficacité de nage
		if self.swimmingEfficiency > 0:
			energyCost = totalPower / self.swimmingEfficiency
		else:
			energyCost = totalPower
		
		return energyCost
	
	def calculateEnvironmentalEnergyCost(
		self,
		basalRate: float,
		depth: float,
		temperature: float,
		oxygenLevel: float,
		salinityDelta: float,
		toxicityLevel: float,
		currentSpeed: float
	) -> float:
		"""
		Calcule le coût énergétique supplémentaire dû aux conditions environnementales.
		
		Args:
			basalRate: Taux métabolique de base
			depth: Profondeur (pour calculer la pression)
			temperature: Température de l'eau
			oxygenLevel: Niveau d'oxygène (0-1)
			salinityDelta: Écart de salinité par rapport à l'optimum
			toxicityLevel: Niveau de toxicité de l'environnement (0-1)
			currentSpeed: Vitesse du courant
			
		Returns:
			Coût énergétique supplémentaire en unités d'énergie
		"""
		# Calcul de la pression (approximativement 1 bar tous les 10m)
		pressure = 1.0 + depth / 10.0
		
		# Coût énergétique supplémentaire dû à la pression
		pressureCost = basalRate * self.environmentalInfluence["pressure"] * (pressure - 1.0)
		
		# Coût énergétique dû au manque d'oxygène
		oxygenCost = 0.0
		if oxygenLevel < 0.8:
			oxygenCost = basalRate * self.environmentalInfluence["oxygen"] * (1.0 - oxygenLevel)
		
		# Coût énergétique dû à l'écart de salinité
		salinityCost = basalRate * self.environmentalInfluence["salinity"] * abs(salinityDelta)
		
		# Coût énergétique dû à la toxicité
		toxicityCost = basalRate * self.environmentalInfluence["toxicity"] * toxicityLevel
		
		# Coût énergétique dû au courant
		currentCost = basalRate * self.environmentalInfluence["current_speed"] * currentSpeed
		
		# Coût énergétique total
		totalEnvironmentalCost = pressureCost + oxygenCost + salinityCost + toxicityCost + currentCost
		
		return totalEnvironmentalCost
	
	def calculateMuscleEnergyCost(self, activationLevel: float, muscleSize: float, efficiency: float) -> float:
		"""
		Calcule le coût énergétique de l'activation musculaire.
		
		Args:
			activationLevel: Niveau d'activation du muscle (0-1)
			muscleSize: Taille relative du muscle
			efficiency: Efficacité énergétique du muscle
			
		Returns:
			Coût énergétique en unités d'énergie
		"""
		# Le coût est proportionnel au carré de l'activation (effort) et à la taille du muscle
		baseCost = activationLevel**2 * muscleSize
		
		# Tenir compte de l'efficacité
		if efficiency > 0:
			energyCost = baseCost / efficiency
		else:
			energyCost = baseCost
			
		return energyCost
	
	def calculateNeuralEnergyCost(self, brainActivity: float, brainSize: float) -> float:
		"""
		Calcule le coût énergétique de l'activité neurale.
		
		Args:
			brainActivity: Niveau d'activité du cerveau (0-1)
			brainSize: Taille relative du cerveau
			
		Returns:
			Coût énergétique en unités d'énergie
		"""
		# Le cerveau consomme beaucoup d'énergie même au repos
		restingBrainCost = 0.2 * brainSize
		
		# Coût supplémentaire dû à l'activité
		activityCost = 0.8 * brainSize * brainActivity
		
		return restingBrainCost + activityCost
	
	def calculateReproductionEnergyCost(self, size: float) -> float:
		"""
		Calcule le coût énergétique de la reproduction.
		
		Args:
			size: Taille de la créature
			
		Returns:
			Coût énergétique en unités d'énergie
		"""
		# Le coût est proportionnel au cube de la taille (approx. de la masse)
		return 50.0 * size**3
	
	def calculateHealingEnergyCost(self, damageLevel: float, healingRate: float, size: float) -> float:
		"""
		Calcule le coût énergétique de la guérison.
		
		Args:
			damageLevel: Niveau de dommage (0-1)
			healingRate: Taux de guérison
			size: Taille de la créature
			
		Returns:
			Coût énergétique en unités d'énergie
		"""
		return 10.0 * damageLevel * healingRate * size**2
	
	def calculateGrowthEnergyCost(self, growthRate: float, currentSize: float) -> float:
		"""
		Calcule le coût énergétique de la croissance.
		
		Args:
			growthRate: Taux de croissance (variation de taille par unité de temps)
			currentSize: Taille actuelle de la créature
			
		Returns:
			Coût énergétique en unités d'énergie
		"""
		# Le coût augmente avec la taille actuelle et le taux de croissance
		return 20.0 * growthRate * currentSize**2
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet EnergyModel en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du modèle d'énergie
		"""
		return {
			"basalRate": self.basalRate,
			"swimmingEfficiency": self.swimmingEfficiency,
			"metabolicMultiplier": self.metabolicMultiplier,
			"sizeScalingExponent": self.sizeScalingExponent,
			"temperatureCoefficient": self.temperatureCoefficient,
			"environmentalInfluence": self.environmentalInfluence
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'EnergyModel':
		"""
		Crée une instance de EnergyModel à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du modèle d'énergie
			
		Returns:
			Instance de EnergyModel reconstruite
		"""
		model = cls(
			basalRate=data["basalRate"],
			swimmingEfficiency=data["swimmingEfficiency"],
			metabolicMultiplier=data["metabolicMultiplier"],
			sizeScalingExponent=data["sizeScalingExponent"],
			temperatureCoefficient=data["temperatureCoefficient"],
			environmentalInfluence=data["environmentalInfluence"]
		)
		
		return model