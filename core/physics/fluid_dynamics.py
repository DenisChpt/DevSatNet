# core/physics/fluid_dynamics.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from utils.serialization import Serializable


class FluidDynamics(Serializable):
	"""
	Classe gérant la dynamique des fluides pour la simulation de l'environnement marin.
	Calcule les forces de traînée, la flottabilité et d'autres effets hydrodynamiques.
	"""
	
	def __init__(
		self,
		waterDensity: float = 1025.0,  # kg/m³
		waterViscosity: float = 1.0e-3,  # Pa·s
		gravitationalAcceleration: float = 9.81,  # m/s²
		turbulenceIntensity: float = 0.1,
		enableBuoyancy: bool = True,
		enableDrag: bool = True,
		enableLift: bool = True,
		enableAddedMass: bool = True
	) -> None:
		"""
		Initialise le modèle de dynamique des fluides.
		
		Args:
			waterDensity: Densité de l'eau en kg/m³
			waterViscosity: Viscosité de l'eau en Pa·s
			gravitationalAcceleration: Accélération due à la gravité en m/s²
			turbulenceIntensity: Intensité de la turbulence (0-1)
			enableBuoyancy: Activer les forces de flottabilité
			enableDrag: Activer les forces de traînée
			enableLift: Activer les forces de portance
			enableAddedMass: Activer l'effet de masse ajoutée
		"""
		self.waterDensity: float = waterDensity
		self.waterViscosity: float = waterViscosity
		self.gravitationalAcceleration: float = gravitationalAcceleration
		self.turbulenceIntensity: float = turbulenceIntensity
		
		self.enableBuoyancy: bool = enableBuoyancy
		self.enableDrag: bool = enableDrag
		self.enableLift: bool = enableLift
		self.enableAddedMass: bool = enableAddedMass
	
	def calculateBuoyancyForce(
		self,
		volume: float,
		density: float,
		position: np.ndarray
	) -> np.ndarray:
		"""
		Calcule la force de flottabilité.
		
		Args:
			volume: Volume de l'objet en m³
			density: Densité de l'objet en kg/m³
			position: Position (pour déterminer la profondeur)
			
		Returns:
			Vecteur de force de flottabilité en newtons
		"""
		if not self.enableBuoyancy:
			return np.zeros(3, dtype=np.float32)
		
		# Principe d'Archimède: la force de flottabilité est égale au poids du fluide déplacé
		displacedMass = volume * self.waterDensity  # Masse du fluide déplacé
		objectMass = volume * density  # Masse de l'objet
		
		# Force nette (flottabilité - poids)
		netForce = displacedMass - objectMass
		
		# La force est dirigée vers le haut (axe y positif)
		buoyancyForce = np.array([0.0, netForce * self.gravitationalAcceleration, 0.0], dtype=np.float32)
		
		return buoyancyForce
	
	def calculateDragForce(
		self,
		velocity: np.ndarray,
		frontalArea: float,
		dragCoefficient: float,
		waterVelocity: np.ndarray = None
	) -> np.ndarray:
		"""
		Calcule la force de traînée.
		
		Args:
			velocity: Vecteur de vitesse de l'objet
			frontalArea: Surface frontale de l'objet en m²
			dragCoefficient: Coefficient de traînée
			waterVelocity: Vecteur de vitesse de l'eau (courant)
			
		Returns:
			Vecteur de force de traînée en newtons
		"""
		if not self.enableDrag:
			return np.zeros(3, dtype=np.float32)
		
		# Si la vitesse de l'eau n'est pas spécifiée, on la considère nulle
		if waterVelocity is None:
			waterVelocity = np.zeros(3, dtype=np.float32)
		
		# Vitesse relative de l'objet par rapport à l'eau
		relativeVelocity = velocity - waterVelocity
		
		# Magnitude de la vitesse relative
		relativeSpeed = np.linalg.norm(relativeVelocity)
		
		# Si la vitesse est trop faible, pas de traînée
		if relativeSpeed < 1e-6:
			return np.zeros(3, dtype=np.float32)
		
		# Formule de traînée: F_drag = 0.5 * ρ * v² * C_d * A
		dragMagnitude = 0.5 * self.waterDensity * relativeSpeed**2 * dragCoefficient * frontalArea
		
		# Direction de la traînée (opposée à la direction du mouvement relatif)
		dragDirection = -relativeVelocity / relativeSpeed
		
		# Vecteur de force de traînée
		dragForce = dragDirection * dragMagnitude
		
		return dragForce
	
	def calculateLiftForce(
		self,
		velocity: np.ndarray,
		liftArea: float,
		liftCoefficient: float,
		normal: np.ndarray,
		waterVelocity: np.ndarray = None
	) -> np.ndarray:
		"""
		Calcule la force de portance.
		
		Args:
			velocity: Vecteur de vitesse de l'objet
			liftArea: Surface portante de l'objet en m²
			liftCoefficient: Coefficient de portance
			normal: Vecteur normal à la surface portante
			waterVelocity: Vecteur de vitesse de l'eau (courant)
			
		Returns:
			Vecteur de force de portance en newtons
		"""
		if not self.enableLift:
			return np.zeros(3, dtype=np.float32)
		
		# Si la vitesse de l'eau n'est pas spécifiée, on la considère nulle
		if waterVelocity is None:
			waterVelocity = np.zeros(3, dtype=np.float32)
		
		# Vitesse relative de l'objet par rapport à l'eau
		relativeVelocity = velocity - waterVelocity
		
		# Magnitude de la vitesse relative
		relativeSpeed = np.linalg.norm(relativeVelocity)
		
		# Si la vitesse est trop faible, pas de portance
		if relativeSpeed < 1e-6:
			return np.zeros(3, dtype=np.float32)
		
		# Normaliser le vecteur normal
		normalNorm = np.linalg.norm(normal)
		if normalNorm < 1e-6:
			return np.zeros(3, dtype=np.float32)
			
		normalizedNormal = normal / normalNorm
		
		# Direction de la vitesse relative
		relativeDirection = relativeVelocity / relativeSpeed
		
		# Produit scalaire pour déterminer l'angle d'attaque
		dotProduct = np.dot(normalizedNormal, relativeDirection)
		
		# L'angle d'attaque influence le coefficient de portance
		# Modèle simplifié: la portance est maximale à un angle d'attaque de 45°
		angleEffect = abs(dotProduct) * (1.0 - abs(dotProduct))  # Maximum à 45°
		
		# Formule de portance: F_lift = 0.5 * ρ * v² * C_l * A * angle_effect
		liftMagnitude = 0.5 * self.waterDensity * relativeSpeed**2 * liftCoefficient * liftArea * angleEffect
		
		# Direction de la portance (perpendiculaire à la direction du mouvement et au vecteur normal)
		liftDirection = np.cross(relativeDirection, np.cross(normalizedNormal, relativeDirection))
		liftDirectionNorm = np.linalg.norm(liftDirection)
		
		if liftDirectionNorm < 1e-6:
			return np.zeros(3, dtype=np.float32)
			
		liftDirection = liftDirection / liftDirectionNorm
		
		# Vecteur de force de portance
		liftForce = liftDirection * liftMagnitude
		
		return liftForce
	
	def calculateAddedMassForce(
		self,
		acceleration: np.ndarray,
		volume: float,
		addedMassCoefficient: float
	) -> np.ndarray:
		"""
		Calcule la force due à l'effet de masse ajoutée.
		
		Args:
			acceleration: Vecteur d'accélération de l'objet
			volume: Volume de l'objet en m³
			addedMassCoefficient: Coefficient de masse ajoutée
			
		Returns:
			Vecteur de force due à la masse ajoutée en newtons
		"""
		if not self.enableAddedMass:
			return np.zeros(3, dtype=np.float32)
		
		# La masse ajoutée est proportionnelle au volume de fluide déplacé
		addedMass = self.waterDensity * volume * addedMassCoefficient
		
		# Formule: F = -m_a * a
		addedMassForce = -addedMass * acceleration
		
		return addedMassForce
	
	def calculatePressureGradientForce(
		self,
		volume: float,
		position: np.ndarray,
		depthGradient: float = 0.1  # Variation de la pression avec la profondeur
	) -> np.ndarray:
		"""
		Calcule la force due au gradient de pression.
		
		Args:
			volume: Volume de l'objet en m³
			position: Position (pour déterminer la profondeur)
			depthGradient: Variation de la pression avec la profondeur
			
		Returns:
			Vecteur de force due au gradient de pression en newtons
		"""
		# La pression augmente avec la profondeur
		depth = position[1]  # Profondeur (axe y)
		
		# Gradient de pression (simplifié)
		pressureGradient = np.array([0.0, -self.waterDensity * self.gravitationalAcceleration * depthGradient, 0.0])
		
		# Force due au gradient de pression
		pressureForce = volume * pressureGradient
		
		return pressureForce
	
	def calculateTurbulenceForce(
		self,
		velocity: np.ndarray,
		frontalArea: float,
		time: float,
		frequency: float = 1.0
	) -> np.ndarray:
		"""
		Calcule une force aléatoire due à la turbulence.
		
		Args:
			velocity: Vecteur de vitesse de l'objet
			frontalArea: Surface frontale de l'objet en m²
			time: Temps actuel (pour les variations temporelles)
			frequency: Fréquence des variations de turbulence
			
		Returns:
			Vecteur de force de turbulence en newtons
		"""
		# La turbulence dépend de la vitesse et de la surface frontale
		speedMagnitude = np.linalg.norm(velocity)
		
		# Base pour le calcul de la turbulence
		turbulenceBase = 0.5 * self.waterDensity * speedMagnitude**2 * frontalArea * self.turbulenceIntensity
		
		# Composante temporelle (variation sinusoïdale)
		timeFactor = np.sin(time * frequency * 2 * np.pi)
		
		# Direction aléatoire mais cohérente dans le temps
		# On utilise des fonctions sinusoïdales déphasées pour chaque composante
		dirX = np.sin(time * frequency * 2 * np.pi)
		dirY = np.sin(time * frequency * 2 * np.pi + 2*np.pi/3)
		dirZ = np.sin(time * frequency * 2 * np.pi + 4*np.pi/3)
		
		direction = np.array([dirX, dirY, dirZ], dtype=np.float32)
		directionNorm = np.linalg.norm(direction)
		
		if directionNorm > 0:
			direction /= directionNorm
		
		# Vecteur de force de turbulence
		turbulenceForce = direction * turbulenceBase * (0.5 + 0.5 * timeFactor)
		
		return turbulenceForce
	
	def calculateTotalFluidForces(
		self,
		bodyProperties: Dict[str, Any],
		dynamics: Dict[str, Any],
		environmentProperties: Dict[str, Any],
		time: float
	) -> Dict[str, np.ndarray]:
		"""
		Calcule toutes les forces dues à la dynamique des fluides.
		
		Args:
			bodyProperties: Propriétés physiques de l'objet
			dynamics: Propriétés dynamiques de l'objet (vitesse, accélération)
			environmentProperties: Propriétés de l'environnement
			time: Temps actuel
			
		Returns:
			Dictionnaire des forces calculées
		"""
		# Extraire les propriétés nécessaires
		volume = bodyProperties.get("volume", 1.0)
		density = bodyProperties.get("density", 1000.0)
		frontalArea = bodyProperties.get("frontalArea", 1.0)
		dragCoefficient = bodyProperties.get("dragCoefficient", 0.5)
		liftArea = bodyProperties.get("liftArea", 1.0)
		liftCoefficient = bodyProperties.get("liftCoefficient", 0.3)
		normal = bodyProperties.get("normal", np.array([0.0, 1.0, 0.0], dtype=np.float32))
		addedMassCoefficient = bodyProperties.get("addedMassCoefficient", 0.5)
		
		position = dynamics.get("position", np.zeros(3, dtype=np.float32))
		velocity = dynamics.get("velocity", np.zeros(3, dtype=np.float32))
		acceleration = dynamics.get("acceleration", np.zeros(3, dtype=np.float32))
		
		waterVelocity = environmentProperties.get("waterVelocity", np.zeros(3, dtype=np.float32))
		
		# Calculer les différentes forces
		buoyancyForce = self.calculateBuoyancyForce(volume, density, position)
		dragForce = self.calculateDragForce(velocity, frontalArea, dragCoefficient, waterVelocity)
		liftForce = self.calculateLiftForce(velocity, liftArea, liftCoefficient, normal, waterVelocity)
		addedMassForce = self.calculateAddedMassForce(acceleration, volume, addedMassCoefficient)
		pressureGradientForce = self.calculatePressureGradientForce(volume, position)
		turbulenceForce = self.calculateTurbulenceForce(velocity, frontalArea, time)
		
		# Force totale
		totalForce = buoyancyForce + dragForce + liftForce + addedMassForce + pressureGradientForce + turbulenceForce
		
		# Retourner les forces individuelles et la force totale
		forces = {
			"buoyancy": buoyancyForce,
			"drag": dragForce,
			"lift": liftForce,
			"addedMass": addedMassForce,
			"pressureGradient": pressureGradientForce,
			"turbulence": turbulenceForce,
			"total": totalForce
		}
		
		return forces
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet FluidDynamics en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du modèle de dynamique des fluides
		"""
		return {
			"waterDensity": self.waterDensity,
			"waterViscosity": self.waterViscosity,
			"gravitationalAcceleration": self.gravitationalAcceleration,
			"turbulenceIntensity": self.turbulenceIntensity,
			"enableBuoyancy": self.enableBuoyancy,
			"enableDrag": self.enableDrag,
			"enableLift": self.enableLift,
			"enableAddedMass": self.enableAddedMass
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'FluidDynamics':
		"""
		Crée une instance de FluidDynamics à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du modèle de dynamique des fluides
			
		Returns:
			Instance de FluidDynamics reconstruite
		"""
		model = cls(
			waterDensity=data["waterDensity"],
			waterViscosity=data["waterViscosity"],
			gravitationalAcceleration=data["gravitationalAcceleration"],
			turbulenceIntensity=data["turbulenceIntensity"],
			enableBuoyancy=data["enableBuoyancy"],
			enableDrag=data["enableDrag"],
			enableLift=data["enableLift"],
			enableAddedMass=data["enableAddedMass"]
		)
		
		return model