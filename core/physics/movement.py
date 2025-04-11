# core/physics/movement.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from utils.serialization import Serializable


class MovementSystem(Serializable):
	"""
	Système gérant le mouvement et la cinématique des créatures.
	Intègre les forces appliquées pour calculer les nouvelles positions et orientations.
	"""
	
	def __init__(
		self,
		enablePhysics: bool = True,
		damping: float = 0.1,  # Facteur d'amortissement pour simuler la résistance du milieu
		angularDamping: float = 0.2,  # Amortissement des rotations
		maxSpeed: float = 50.0,  # Vitesse maximale en unités/s
		maxAcceleration: float = 100.0,  # Accélération maximale en unités/s²
		maxAngularSpeed: float = 5.0,  # Vitesse angulaire maximale en rad/s
		maxAngularAcceleration: float = 10.0,  # Accélération angulaire maximale en rad/s²
		gravity: np.ndarray = np.array([0.0, -9.81, 0.0], dtype=np.float32),  # Gravité
		timeScale: float = 1.0,  # Échelle de temps (pour ralentir/accélérer la simulation)
		subSteps: int = 1  # Nombre de sous-étapes pour l'intégration
	) -> None:
		"""
		Initialise le système de mouvement.
		
		Args:
			enablePhysics: Activer/désactiver la physique
			damping: Facteur d'amortissement linéaire
			angularDamping: Facteur d'amortissement angulaire
			maxSpeed: Vitesse maximale autorisée
			maxAcceleration: Accélération maximale autorisée
			maxAngularSpeed: Vitesse angulaire maximale autorisée
			maxAngularAcceleration: Accélération angulaire maximale autorisée
			gravity: Vecteur de gravité
			timeScale: Échelle de temps pour la simulation
			subSteps: Nombre de sous-étapes pour améliorer la stabilité
		"""
		self.enablePhysics: bool = enablePhysics
		self.damping: float = damping
		self.angularDamping: float = angularDamping
		self.maxSpeed: float = maxSpeed
		self.maxAcceleration: float = maxAcceleration
		self.maxAngularSpeed: float = maxAngularSpeed
		self.maxAngularAcceleration: float = maxAngularAcceleration
		self.gravity: np.ndarray = gravity
		self.timeScale: float = timeScale
		self.subSteps: int = subSteps
	
	def updatePosition(
		self,
		position: np.ndarray,
		velocity: np.ndarray,
		acceleration: np.ndarray,
		deltaTime: float
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Met à jour la position basée sur la vitesse et l'accélération (intégration semi-implicite d'Euler).
		
		Args:
			position: Position actuelle
			velocity: Vitesse actuelle
			acceleration: Accélération actuelle
			deltaTime: Intervalle de temps
			
		Returns:
			Tuple (newPosition, newVelocity, newAcceleration) des valeurs mises à jour
		"""
		if not self.enablePhysics:
			return position, velocity, acceleration
		
		# Appliquer l'échelle de temps
		scaledDeltaTime = deltaTime * self.timeScale
		
		# Limiter l'accélération
		accelerationMagnitude = np.linalg.norm(acceleration)
		if accelerationMagnitude > self.maxAcceleration:
			acceleration = acceleration * self.maxAcceleration / accelerationMagnitude
		
		# Mettre à jour la vitesse avec l'accélération
		newVelocity = velocity + acceleration * scaledDeltaTime
		
		# Appliquer l'amortissement
		newVelocity *= (1.0 - self.damping * scaledDeltaTime)
		
		# Limiter la vitesse
		speedMagnitude = np.linalg.norm(newVelocity)
		if speedMagnitude > self.maxSpeed:
			newVelocity = newVelocity * self.maxSpeed / speedMagnitude
		
		# Mettre à jour la position avec la nouvelle vitesse
		newPosition = position + newVelocity * scaledDeltaTime
		
		# L'accélération est réinitialisée à chaque pas de temps (elle sera recalculée)
		newAcceleration = np.zeros_like(acceleration)
		
		return newPosition, newVelocity, newAcceleration
	
	def updateOrientation(
		self,
		orientation: np.ndarray,
		angularVelocity: np.ndarray,
		angularAcceleration: np.ndarray,
		deltaTime: float
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Met à jour l'orientation basée sur la vitesse angulaire et l'accélération angulaire.
		
		Args:
			orientation: Orientation actuelle (angles d'Euler)
			angularVelocity: Vitesse angulaire actuelle
			angularAcceleration: Accélération angulaire actuelle
			deltaTime: Intervalle de temps
			
		Returns:
			Tuple (newOrientation, newAngularVelocity, newAngularAcceleration) des valeurs mises à jour
		"""
		if not self.enablePhysics:
			return orientation, angularVelocity, angularAcceleration
		
		# Appliquer l'échelle de temps
		scaledDeltaTime = deltaTime * self.timeScale
		
		# Limiter l'accélération angulaire
		angularAccelerationMagnitude = np.linalg.norm(angularAcceleration)
		if angularAccelerationMagnitude > self.maxAngularAcceleration:
			angularAcceleration = angularAcceleration * self.maxAngularAcceleration / angularAccelerationMagnitude
		
		# Mettre à jour la vitesse angulaire avec l'accélération angulaire
		newAngularVelocity = angularVelocity + angularAcceleration * scaledDeltaTime
		
		# Appliquer l'amortissement angulaire
		newAngularVelocity *= (1.0 - self.angularDamping * scaledDeltaTime)
		
		# Limiter la vitesse angulaire
		angularSpeedMagnitude = np.linalg.norm(newAngularVelocity)
		if angularSpeedMagnitude > self.maxAngularSpeed:
			newAngularVelocity = newAngularVelocity * self.maxAngularSpeed / angularSpeedMagnitude
		
		# Mettre à jour l'orientation avec la nouvelle vitesse angulaire
		# Pour les angles d'Euler, on ajoute simplement la vitesse angulaire
		newOrientation = orientation + newAngularVelocity * scaledDeltaTime
		
		# Normaliser les angles entre -π et π
		newOrientation = np.mod(newOrientation + np.pi, 2 * np.pi) - np.pi
		
		# L'accélération angulaire est réinitialisée à chaque pas de temps
		newAngularAcceleration = np.zeros_like(angularAcceleration)
		
		return newOrientation, newAngularVelocity, newAngularAcceleration
	
	def calculateAccelerationFromForce(
		self,
		force: np.ndarray,
		mass: float
	) -> np.ndarray:
		"""
		Calcule l'accélération à partir d'une force (F = ma).
		
		Args:
			force: Force appliquée
			mass: Masse de l'objet
			
		Returns:
			Accélération résultante
		"""
		if mass <= 0.0:
			return np.zeros_like(force)
			
		# F = ma, donc a = F/m
		acceleration = force / mass
		
		# Limiter l'accélération
		accelerationMagnitude = np.linalg.norm(acceleration)
		if accelerationMagnitude > self.maxAcceleration:
			acceleration = acceleration * self.maxAcceleration / accelerationMagnitude
			
		return acceleration
	
	def calculateAngularAccelerationFromTorque(
		self,
		torque: np.ndarray,
		inertia: np.ndarray
	) -> np.ndarray:
		"""
		Calcule l'accélération angulaire à partir d'un couple et d'un tenseur d'inertie.
		
		Args:
			torque: Couple appliqué
			inertia: Tenseur d'inertie ou moments d'inertie principaux
			
		Returns:
			Accélération angulaire résultante
		"""
		# Simplification: on considère que inertia est un vecteur des moments d'inertie principaux
		if np.any(inertia <= 0.0):
			return np.zeros_like(torque)
			
		# τ = Iα, donc α = τ/I
		# Dans le cas général, on aurait α = I⁻¹τ avec I le tenseur d'inertie complet
		# Ici on simplifie en supposant que I est diagonal (pas de produits d'inertie)
		angularAcceleration = torque / inertia
		
		# Limiter l'accélération angulaire
		angularAccelerationMagnitude = np.linalg.norm(angularAcceleration)
		if angularAccelerationMagnitude > self.maxAngularAcceleration:
			angularAcceleration = angularAcceleration * self.maxAngularAcceleration / angularAccelerationMagnitude
			
		return angularAcceleration
	
	def applyForce(
		self,
		force: np.ndarray,
		position: np.ndarray,
		velocity: np.ndarray,
		acceleration: np.ndarray,
		mass: float,
		deltaTime: float,
		pointOfApplication: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
		"""
		Applique une force à un objet et met à jour son mouvement.
		
		Args:
			force: Force à appliquer
			position: Position actuelle
			velocity: Vitesse actuelle
			acceleration: Accélération actuelle
			mass: Masse de l'objet
			deltaTime: Intervalle de temps
			pointOfApplication: Point d'application de la force (pour le calcul du couple)
			
		Returns:
			Tuple (newPosition, newVelocity, newAcceleration, torque) des valeurs mises à jour
			et du couple généré (si le point d'application est spécifié)
		"""
		if not self.enablePhysics:
			return position, velocity, acceleration, None
			
		# Calculer l'accélération supplémentaire due à la force
		additionalAcceleration = self.calculateAccelerationFromForce(force, mass)
		
		# Ajouter à l'accélération actuelle
		totalAcceleration = acceleration + additionalAcceleration
		
		# Mettre à jour la position, la vitesse et l'accélération
		newPosition, newVelocity, newAcceleration = self.updatePosition(
			position, velocity, totalAcceleration, deltaTime
		)
		
		# Calculer le couple si le point d'application est spécifié
		torque = None
		if pointOfApplication is not None:
			# Vecteur du centre de masse au point d'application
			r = pointOfApplication - position
			
			# Couple = r × F
			torque = np.cross(r, force)
			
		return newPosition, newVelocity, newAcceleration, torque
	
	def applyImpulse(
		self,
		impulse: np.ndarray,
		position: np.ndarray,
		velocity: np.ndarray,
		mass: float,
		pointOfApplication: Optional[np.ndarray] = None
	) -> Tuple[np.ndarray, Optional[np.ndarray]]:
		"""
		Applique une impulsion à un objet (changement instantané de quantité de mouvement).
		
		Args:
			impulse: Impulsion à appliquer
			position: Position actuelle
			velocity: Vitesse actuelle
			mass: Masse de l'objet
			pointOfApplication: Point d'application de l'impulsion (pour le calcul du couple)
			
		Returns:
			Tuple (newVelocity, angularImpulse) de la vitesse mise à jour
			et de l'impulsion angulaire générée (si le point d'application est spécifié)
		"""
		if not self.enablePhysics or mass <= 0.0:
			return velocity, None
			
		# p = mv, donc Δv = Δp/m = impulse/m
		velocityChange = impulse / mass
		
		# Mettre à jour la vitesse
		newVelocity = velocity + velocityChange
		
		# Limiter la vitesse
		speedMagnitude = np.linalg.norm(newVelocity)
		if speedMagnitude > self.maxSpeed:
			newVelocity = newVelocity * self.maxSpeed / speedMagnitude
		
		# Calculer l'impulsion angulaire si le point d'application est spécifié
		angularImpulse = None
		if pointOfApplication is not None:
			# Vecteur du centre de masse au point d'application
			r = pointOfApplication - position
			
			# Impulsion angulaire = r × impulse
			angularImpulse = np.cross(r, impulse)
			
		return newVelocity, angularImpulse
	
	def integrateMotion(
		self,
		body: Dict[str, Any],
		forces: Dict[str, np.ndarray],
		deltaTime: float
	) -> Dict[str, Any]:
		"""
		Intègre le mouvement complet d'un corps basé sur les forces appliquées.
		
		Args:
			body: Dictionnaire contenant les propriétés du corps
			forces: Dictionnaire des forces appliquées
			deltaTime: Intervalle de temps
			
		Returns:
			Dictionnaire mis à jour avec les nouvelles propriétés du corps
		"""
		if not self.enablePhysics:
			return body
			
		# Extraire les propriétés nécessaires
		position = body.get("position", np.zeros(3, dtype=np.float32))
		velocity = body.get("velocity", np.zeros(3, dtype=np.float32))
		acceleration = body.get("acceleration", np.zeros(3, dtype=np.float32))
		orientation = body.get("orientation", np.zeros(3, dtype=np.float32))
		angularVelocity = body.get("angularVelocity", np.zeros(3, dtype=np.float32))
		angularAcceleration = body.get("angularAcceleration", np.zeros(3, dtype=np.float32))
		mass = body.get("mass", 1.0)
		inertia = body.get("inertia", np.ones(3, dtype=np.float32))
		isStatic = body.get("isStatic", False)
		
		# Si le corps est statique, pas de mouvement
		if isStatic:
			return body
			
		# Diviser le pas de temps en sous-étapes pour améliorer la stabilité
		subDeltaTime = deltaTime / self.subSteps
		
		# Initialiser les forces et couples totaux
		totalForce = np.zeros(3, dtype=np.float32)
		totalTorque = np.zeros(3, dtype=np.float32)
		
		# Ajouter toutes les forces
		for forceName, force in forces.items():
			totalForce += force
			
			# Si une force a un couple associé, l'ajouter
			if f"{forceName}_torque" in forces:
				totalTorque += forces[f"{forceName}_torque"]
		
		# Ajouter la gravité
		if body.get("enableGravity", True):
			totalForce += self.gravity * mass
		
		for _ in range(self.subSteps):
			# Calculer les accélérations
			newAcceleration = self.calculateAccelerationFromForce(totalForce, mass)
			newAngularAcceleration = self.calculateAngularAccelerationFromTorque(totalTorque, inertia)
			
			# Intégrer le mouvement linéaire
			position, velocity, acceleration = self.updatePosition(
				position, velocity, newAcceleration, subDeltaTime
			)
			
			# Intégrer le mouvement angulaire
			orientation, angularVelocity, angularAcceleration = self.updateOrientation(
				orientation, angularVelocity, newAngularAcceleration, subDeltaTime
			)
		
		# Mettre à jour le dictionnaire du corps
		updatedBody = body.copy()
		updatedBody["position"] = position
		updatedBody["velocity"] = velocity
		updatedBody["acceleration"] = acceleration
		updatedBody["orientation"] = orientation
		updatedBody["angularVelocity"] = angularVelocity
		updatedBody["angularAcceleration"] = angularAcceleration
		
		return updatedBody
	
	def calculateSwimmingForces(
		self,
		body: Dict[str, Any],
		muscleActivations: np.ndarray,
		muscles: List[Dict[str, Any]],
		joints: List[Dict[str, Any]],
		waterProperties: Any
	) -> Dict[str, np.ndarray]:
		"""
		Calcule les forces de nage générées par les muscles et l'interaction avec l'eau.
		
		Args:
			body: Dictionnaire contenant les propriétés du corps
			muscleActivations: Activations des muscles (-1.0 à 1.0)
			muscles: Liste des propriétés des muscles
			joints: Liste des propriétés des articulations
			waterProperties: Propriétés de l'eau à la position actuelle
			
		Returns:
			Dictionnaire des forces générées par la nage
		"""
		if not self.enablePhysics:
			return {"swimming": np.zeros(3, dtype=np.float32)}
			
		# Position et orientation du corps
		position = body.get("position", np.zeros(3, dtype=np.float32))
		orientation = body.get("orientation", np.zeros(3, dtype=np.float32))
		
		# Propriétés de l'eau
		waterDensity = waterProperties.get("density", 1000.0)
		waterVelocity = waterProperties.get("velocity", np.zeros(3, dtype=np.float32))
		
		# Force totale de nage
		swimmingForce = np.zeros(3, dtype=np.float32)
		
		# Couple total
		swimmingTorque = np.zeros(3, dtype=np.float32)
		
		# Pour chaque muscle, calculer la force générée
		for i, (muscle, activation) in enumerate(zip(muscles, muscleActivations)):
			if i >= len(muscleActivations):
				break
				
			# Propriétés du muscle
			strength = muscle.get("strength", 1.0)
			efficiency = muscle.get("efficiency", 1.0)
			
			# Articulations connectées par ce muscle
			jointIds = muscle.get("jointIds", [])
			
			# Si le muscle ne connecte pas d'articulations, continuer
			if len(jointIds) < 2:
				continue
				
			# Obtenir les articulations
			connectedJoints = [joint for joint in joints if joint.get("id") in jointIds]
			
			# Si moins de deux articulations sont trouvées, continuer
			if len(connectedJoints) < 2:
				continue
				
			# Calculer la force générée par l'activation du muscle
			# L'intensité dépend de l'activation et des propriétés du muscle
			activationIntensity = abs(activation) * strength * efficiency
			
			# Direction de base (du premier au deuxième joint)
			joint1Pos = connectedJoints[0].get("position", np.zeros(3, dtype=np.float32))
			joint2Pos = connectedJoints[1].get("position", np.zeros(3, dtype=np.float32))
			
			# Convertir les positions locales en positions mondiales
			# Simplification: on considère que les positions sont déjà dans le système mondial
			
			# Direction du muscle
			muscleDirection = joint2Pos - joint1Pos
			muscleMagnitude = np.linalg.norm(muscleDirection)
			
			if muscleMagnitude < 1e-6:
				continue
				
			muscleDirection /= muscleMagnitude
			
			# Direction de la force (perpendiculaire au muscle pour créer un mouvement de rame)
			# On considère que le joint1 est plus proche de la colonne vertébrale
			# La force est appliquée perpendiculairement selon l'activation
			
			# Vecteur up approximatif (en fonction de l'orientation du corps)
			upVector = np.array([
				np.sin(orientation[1]) * np.sin(orientation[0]),
				np.cos(orientation[0]),
				np.cos(orientation[1]) * np.sin(orientation[0])
			], dtype=np.float32)
			
			# Vecteur perpendiculaire au muscle et à l'axe vertical
			perpendicularDirection = np.cross(muscleDirection, upVector)
			perpMagnitude = np.linalg.norm(perpendicularDirection)
			
			if perpMagnitude < 1e-6:
				# Fallback: utiliser simplement une perpendiculaire arbitraire
				perpendicularDirection = np.array([0.0, 0.0, 1.0])
				if abs(np.dot(muscleDirection, perpendicularDirection)) > 0.9:
					perpendicularDirection = np.array([1.0, 0.0, 0.0])
				perpendicularDirection = np.cross(muscleDirection, perpendicularDirection)
				perpMagnitude = np.linalg.norm(perpendicularDirection)
				if perpMagnitude < 1e-6:
					continue
					
			perpendicularDirection /= perpMagnitude
			
			# Ajuster la direction selon le signe de l'activation
			if activation < 0:
				perpendicularDirection = -perpendicularDirection
			
			# Force de propulsion (dépend de l'activation, de la surface et de la densité de l'eau)
			# Modèle simplifié: F = 0.5 * ρ * v² * A * Cd
			# Où ρ est la densité de l'eau, v la vitesse relative, A la surface et Cd le coefficient de traînée
			
			# Surface approximative du muscle (simplifié)
			muscleArea = muscleMagnitude * 0.1  # 10% de la longueur comme largeur
			
			# Vitesse relative (simulée en fonction de l'activation)
			relativeSpeed = activationIntensity * 2.0  # Facteur arbitraire
			
			# Coefficient de traînée (simulé)
			dragCoefficient = 0.5
			
			# Force de propulsion
			propulsionMagnitude = 0.5 * waterDensity * relativeSpeed**2 * muscleArea * dragCoefficient
			propulsionForce = perpendicularDirection * propulsionMagnitude
			
			# Point d'application (milieu du muscle)
			applicationPoint = (joint1Pos + joint2Pos) / 2.0
			
			# Ajouter à la force totale
			swimmingForce += propulsionForce
			
			# Calculer le couple
			r = applicationPoint - position
			muscleTorque = np.cross(r, propulsionForce)
			swimmingTorque += muscleTorque
		
		return {
			"swimming": swimmingForce,
			"swimming_torque": swimmingTorque
		}
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet MovementSystem en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du système de mouvement
		"""
		return {
			"enablePhysics": self.enablePhysics,
			"damping": self.damping,
			"angularDamping": self.angularDamping,
			"maxSpeed": self.maxSpeed,
			"maxAcceleration": self.maxAcceleration,
			"maxAngularSpeed": self.maxAngularSpeed,
			"maxAngularAcceleration": self.maxAngularAcceleration,
			"gravity": self.gravity.tolist(),
			"timeScale": self.timeScale,
			"subSteps": self.subSteps
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'MovementSystem':
		"""
		Crée une instance de MovementSystem à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du système de mouvement
			
		Returns:
			Instance de MovementSystem reconstruite
		"""
		return cls(
			enablePhysics=data["enablePhysics"],
			damping=data["damping"],
			angularDamping=data["angularDamping"],
			maxSpeed=data["maxSpeed"],
			maxAcceleration=data["maxAcceleration"],
			maxAngularSpeed=data["maxAngularSpeed"],
			maxAngularAcceleration=data["maxAngularAcceleration"],
			gravity=np.array(data["gravity"], dtype=np.float32),
			timeScale=data["timeScale"],
			subSteps=data["subSteps"]
		)