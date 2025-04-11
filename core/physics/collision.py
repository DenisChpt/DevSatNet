# core/physics/collision.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from utils.serialization import Serializable


class CollisionSystem(Serializable):
	"""
	Système de détection et de résolution des collisions entre les objets de la simulation.
	"""
	
	def __init__(
		self,
		enableCollisions: bool = True,
		restitutionCoefficient: float = 0.5,  # Coefficient de restitution (élasticité)
		frictionCoefficient: float = 0.3,     # Coefficient de friction
		collisionDamping: float = 0.2,        # Amortissement des collisions
		enableSelfCollisions: bool = False,    # Collisions entre parties d'un même objet
		simplifiedCollisions: bool = True      # Utiliser un modèle de collision simplifié
	) -> None:
		"""
		Initialise le système de collision.
		
		Args:
			enableCollisions: Activer/désactiver toutes les collisions
			restitutionCoefficient: Coefficient de restitution (0 = totalement inélastique, 1 = totalement élastique)
			frictionCoefficient: Coefficient de friction entre les objets
			collisionDamping: Facteur d'amortissement des collisions
			enableSelfCollisions: Activer les collisions entre parties d'un même objet
			simplifiedCollisions: Utiliser un modèle de collision simplifié pour les performances
		"""
		self.enableCollisions: bool = enableCollisions
		self.restitutionCoefficient: float = restitutionCoefficient
		self.frictionCoefficient: float = frictionCoefficient
		self.collisionDamping: float = collisionDamping
		self.enableSelfCollisions: bool = enableSelfCollisions
		self.simplifiedCollisions: bool = simplifiedCollisions
		
		# Statistiques de collisions
		self.totalCollisions: int = 0
		self.lastCollisionTime: float = 0.0
		self.collisionsThisFrame: int = 0
	
	def detectSphereCollision(
		self,
		position1: np.ndarray,
		radius1: float,
		position2: np.ndarray,
		radius2: float
	) -> Tuple[bool, np.ndarray, float]:
		"""
		Détecte une collision entre deux sphères.
		
		Args:
			position1: Position du centre de la première sphère
			radius1: Rayon de la première sphère
			position2: Position du centre de la seconde sphère
			radius2: Rayon de la seconde sphère
			
		Returns:
			Tuple (collision, normal, penetrationDepth) indiquant s'il y a collision,
			la normale de collision et la profondeur de pénétration
		"""
		if not self.enableCollisions:
			return False, np.zeros(3, dtype=np.float32), 0.0
		
		# Vecteur de séparation entre les centres
		separation = position2 - position1
		distance = np.linalg.norm(separation)
		
		# Somme des rayons
		sumRadii = radius1 + radius2
		
		# S'il y a collision, distance < somme des rayons
		if distance < sumRadii:
			# Calculer la normale de collision (direction de la séparation)
			normal = separation / distance if distance > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
			
			# Profondeur de pénétration
			penetrationDepth = sumRadii - distance
			
			return True, normal, penetrationDepth
		
		return False, np.zeros(3, dtype=np.float32), 0.0
	
	def detectAABBCollision(
		self,
		min1: np.ndarray,
		max1: np.ndarray,
		min2: np.ndarray,
		max2: np.ndarray
	) -> Tuple[bool, np.ndarray, float]:
		"""
		Détecte une collision entre deux boîtes englobantes alignées sur les axes (AABB).
		
		Args:
			min1: Coin minimum de la première AABB
			max1: Coin maximum de la première AABB
			min2: Coin minimum de la seconde AABB
			max2: Coin maximum de la seconde AABB
			
		Returns:
			Tuple (collision, normal, penetrationDepth) indiquant s'il y a collision,
			la normale de collision et la profondeur de pénétration
		"""
		if not self.enableCollisions:
			return False, np.zeros(3, dtype=np.float32), 0.0
		
		# Vérifier s'il y a chevauchement sur tous les axes
		overlap = True
		for i in range(3):
			overlap = overlap and (min1[i] <= max2[i] and max1[i] >= min2[i])
			
		if not overlap:
			return False, np.zeros(3, dtype=np.float32), 0.0
		
		# Calculer les profondeurs de pénétration sur chaque axe
		depths = np.zeros(3, dtype=np.float32)
		for i in range(3):
			# Deux cas possibles: min1 est dans l'intervalle 2 ou max1 est dans l'intervalle 2
			if min1[i] >= min2[i] and min1[i] <= max2[i]:
				depths[i] = max2[i] - min1[i]
			else:
				depths[i] = max1[i] - min2[i]
		
		# Trouver l'axe de pénétration minimale
		minDepthIndex = np.argmin(depths)
		penetrationDepth = depths[minDepthIndex]
		
		# Déterminer la normale
		normal = np.zeros(3, dtype=np.float32)
		normal[minDepthIndex] = 1.0
		
		# Ajuster la direction de la normale
		center1 = (min1 + max1) / 2
		center2 = (min2 + max2) / 2
		if center1[minDepthIndex] > center2[minDepthIndex]:
			normal[minDepthIndex] = -1.0
		
		return True, normal, penetrationDepth
	
	def detectCapsuleCollision(
		self,
		positionA1: np.ndarray,
		positionA2: np.ndarray,
		radiusA: float,
		positionB1: np.ndarray,
		positionB2: np.ndarray,
		radiusB: float
	) -> Tuple[bool, np.ndarray, float]:
		"""
		Détecte une collision entre deux capsules (segments avec des extrémités sphériques).
		
		Args:
			positionA1: Première extrémité de la capsule A
			positionA2: Seconde extrémité de la capsule A
			radiusA: Rayon de la capsule A
			positionB1: Première extrémité de la capsule B
			positionB2: Seconde extrémité de la capsule B
			radiusB: Rayon de la capsule B
			
		Returns:
			Tuple (collision, normal, penetrationDepth) indiquant s'il y a collision,
			la normale de collision et la profondeur de pénétration
		"""
		if not self.enableCollisions:
			return False, np.zeros(3, dtype=np.float32), 0.0
		
		# Vecteurs des segments
		segmentA = positionA2 - positionA1
		segmentB = positionB2 - positionB1
		
		# Direction des segments
		lengthA = np.linalg.norm(segmentA)
		lengthB = np.linalg.norm(segmentB)
		
		if lengthA < 1e-6 or lengthB < 1e-6:
			# Un des segments est dégénéré, traiter comme une collision de sphères
			return self.detectSphereCollision(positionA1, radiusA, positionB1, radiusB)
		
		dirA = segmentA / lengthA
		dirB = segmentB / lengthB
		
		# Vecteur entre les points de départ des segments
		r = positionA1 - positionB1
		
		# Produits scalaires
		a = np.dot(dirA, dirA)  # Toujours 1 pour des vecteurs normalisés
		b = np.dot(dirA, dirB)
		c = np.dot(dirB, dirB)  # Toujours 1 pour des vecteurs normalisés
		d = np.dot(dirA, r)
		e = np.dot(dirB, r)
		
		# Déterminant
		det = a * c - b * b  # Toujours positif
		
		# Paramètres du point le plus proche sur chaque segment
		s, t = 0.0, 0.0
		
		if det > 1e-6:
			# Les segments ne sont pas parallèles
			s = (b * e - c * d) / det
			t = (a * e - b * d) / det
			
			# Limiter s et t aux segments
			s = max(0.0, min(lengthA, s))
			t = max(0.0, min(lengthB, t))
		else:
			# Les segments sont parallèles, chercher la projection de A1 sur B
			t = max(0.0, min(lengthB, e))
			s = max(0.0, min(lengthA, 0.0 - d))
		
		# Points les plus proches sur chaque segment
		closestA = positionA1 + dirA * s
		closestB = positionB1 + dirB * t
		
		# Vérifier la collision entre les points les plus proches
		return self.detectSphereCollision(closestA, radiusA, closestB, radiusB)
	
	def detectRayCollision(
		self,
		rayOrigin: np.ndarray,
		rayDirection: np.ndarray,
		position: np.ndarray,
		radius: float
	) -> Tuple[bool, float, np.ndarray]:
		"""
		Détecte une collision entre un rayon et une sphère.
		
		Args:
			rayOrigin: Origine du rayon
			rayDirection: Direction du rayon (normalisée)
			position: Position du centre de la sphère
			radius: Rayon de la sphère
			
		Returns:
			Tuple (collision, distance, hitPoint) indiquant s'il y a collision,
			la distance le long du rayon et le point d'impact
		"""
		if not self.enableCollisions:
			return False, float('inf'), np.zeros(3, dtype=np.float32)
		
		# Vecteur de l'origine du rayon au centre de la sphère
		oc = rayOrigin - position
		
		# Coefficients de l'équation quadratique
		a = np.dot(rayDirection, rayDirection)  # Toujours 1 pour une direction normalisée
		b = 2.0 * np.dot(oc, rayDirection)
		c = np.dot(oc, oc) - radius * radius
		
		# Discriminant
		discriminant = b * b - 4 * a * c
		
		if discriminant < 0:
			# Pas d'intersection
			return False, float('inf'), np.zeros(3, dtype=np.float32)
		
		# Distance le long du rayon
		sqrtDiscriminant = np.sqrt(discriminant)
		distance = (-b - sqrtDiscriminant) / (2.0 * a)
		
		if distance < 0:
			# L'intersection est derrière l'origine du rayon
			distance = (-b + sqrtDiscriminant) / (2.0 * a)
			
			if distance < 0:
				# Les deux intersections sont derrière
				return False, float('inf'), np.zeros(3, dtype=np.float32)
		
		# Point d'impact
		hitPoint = rayOrigin + rayDirection * distance
		
		return True, distance, hitPoint
	
	def resolveCollision(
		self,
		position1: np.ndarray,
		velocity1: np.ndarray,
		mass1: float,
		position2: np.ndarray,
		velocity2: np.ndarray,
		mass2: float,
		normal: np.ndarray,
		penetrationDepth: float
	) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Résout une collision entre deux objets.
		
		Args:
			position1: Position du premier objet
			velocity1: Vitesse du premier objet
			mass1: Masse du premier objet
			position2: Position du second objet
			velocity2: Vitesse du second objet
			mass2: Masse du second objet
			normal: Normale de collision (pointant du premier vers le second objet)
			penetrationDepth: Profondeur de pénétration
			
		Returns:
			Tuple (newPosition1, newVelocity1, newPosition2, newVelocity2) avec les positions
			et vitesses mises à jour après résolution de la collision
		"""
		if not self.enableCollisions:
			return position1, velocity1, position2, velocity2
		
		# Vitesse relative
		relativeVelocity = velocity2 - velocity1
		
		# Vitesse normale relative
		normalVelocity = np.dot(relativeVelocity, normal)
		
		# Ne résoudre la collision que si les objets se rapprochent
		if normalVelocity > 0:
			return position1, velocity1, position2, velocity2
		
		# Impulsion normale
		restitution = self.restitutionCoefficient
		j = -(1.0 + restitution) * normalVelocity
		j /= 1.0 / mass1 + 1.0 / mass2
		
		# Appliquer l'impulsion
		impulse = j * normal
		
		# Nouvelles vitesses
		newVelocity1 = velocity1 - impulse / mass1
		newVelocity2 = velocity2 + impulse / mass2
		
		# Amortissement
		newVelocity1 *= (1.0 - self.collisionDamping)
		newVelocity2 *= (1.0 - self.collisionDamping)
		
		# Résoudre la pénétration (correction de position)
		correction = normal * penetrationDepth * 0.5  # Répartir entre les deux objets
		newPosition1 = position1 - correction * (mass2 / (mass1 + mass2))
		newPosition2 = position2 + correction * (mass1 / (mass1 + mass2))
		
		return newPosition1, newVelocity1, newPosition2, newVelocity2
	
	def handleTerrainCollision(
		self,
		position: np.ndarray,
		velocity: np.ndarray,
		radius: float,
		terrain: Any,
		restitution: Optional[float] = None
	) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Gère la collision avec le terrain.
		
		Args:
			position: Position de l'objet
			velocity: Vitesse de l'objet
			radius: Rayon de l'objet (pour la détection de collision)
			terrain: Objet terrain pour obtenir l'élévation
			restitution: Coefficient de restitution spécifique (optionnel)
			
		Returns:
			Tuple (newPosition, newVelocity) après résolution de la collision
		"""
		if not self.enableCollisions:
			return position, velocity
		
		# Obtenir l'élévation du terrain à la position horizontale de l'objet
		terrainElevation = terrain.getElevationAt(position[0], position[2])
		
		# Distance verticale entre l'objet et le terrain
		distanceToTerrain = position[1] - terrainElevation
		
		# S'il y a collision (objet sous le terrain ou à moins de son rayon)
		if distanceToTerrain < radius:
			# Normale du terrain (simplifiée à l'axe vertical)
			normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
			
			# Profondeur de pénétration
			penetrationDepth = radius - distanceToTerrain
			
			# Coefficient de restitution
			if restitution is None:
				restitution = self.restitutionCoefficient
			
			# Composante normale de la vitesse
			normalVelocity = np.dot(velocity, normal)
			
			# Ne résoudre que si l'objet se dirige vers le terrain
			if normalVelocity < 0:
				# Calculer la nouvelle vitesse avec la réflexion et la restitution
				reflectionVelocity = velocity - (1.0 + restitution) * normalVelocity * normal
				
				# Appliquer l'amortissement
				newVelocity = reflectionVelocity * (1.0 - self.collisionDamping)
				
				# Appliquer la friction (composante tangentielle)
				tangentialVelocity = velocity - normalVelocity * normal
				tangentialSpeed = np.linalg.norm(tangentialVelocity)
				
				if tangentialSpeed > 1e-6:
					frictionDirection = tangentialVelocity / tangentialSpeed
					frictionMagnitude = self.frictionCoefficient * abs(normalVelocity)
					
					# Limiter la friction à la vitesse tangentielle
					frictionMagnitude = min(frictionMagnitude, tangentialSpeed)
					
					newVelocity -= frictionDirection * frictionMagnitude
			else:
				# L'objet remonte déjà, pas besoin de changer la vitesse
				newVelocity = velocity
			
			# Corriger la position pour éliminer la pénétration
			newPosition = position.copy()
			newPosition[1] = terrainElevation + radius
			
			return newPosition, newVelocity
		
		# Pas de collision
		return position, velocity
	
	def updateCollisionStatistics(self, time: float, numCollisions: int) -> None:
		"""
		Met à jour les statistiques de collision.
		
		Args:
			time: Temps actuel
			numCollisions: Nombre de collisions détectées
		"""
		self.totalCollisions += numCollisions
		self.lastCollisionTime = time if numCollisions > 0 else self.lastCollisionTime
		self.collisionsThisFrame = numCollisions
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet CollisionSystem en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du système de collision
		"""
		return {
			"enableCollisions": self.enableCollisions,
			"restitutionCoefficient": self.restitutionCoefficient,
			"frictionCoefficient": self.frictionCoefficient,
			"collisionDamping": self.collisionDamping,
			"enableSelfCollisions": self.enableSelfCollisions,
			"simplifiedCollisions": self.simplifiedCollisions,
			"totalCollisions": self.totalCollisions,
			"lastCollisionTime": self.lastCollisionTime
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'CollisionSystem':
		"""
		Crée une instance de CollisionSystem à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du système de collision
			
		Returns:
			Instance de CollisionSystem reconstruite
		"""
		system = cls(
			enableCollisions=data["enableCollisions"],
			restitutionCoefficient=data["restitutionCoefficient"],
			frictionCoefficient=data["frictionCoefficient"],
			collisionDamping=data["collisionDamping"],
			enableSelfCollisions=data["enableSelfCollisions"],
			simplifiedCollisions=data["simplifiedCollisions"]
		)
		
		system.totalCollisions = data["totalCollisions"]
		system.lastCollisionTime = data["lastCollisionTime"]
		
		return system