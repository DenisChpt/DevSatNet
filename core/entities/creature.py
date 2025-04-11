from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
from torch import Tensor
import uuid

from core.entities.joint import Joint
from core.entities.limb import Limb
from core.entities.muscle import Muscle
from core.entities.sensor import Sensor
from core.physics.energy_model import EnergyModel
from learning.models.creature_brain import CreatureBrain
from utils.serialization import Serializable


class Creature(Serializable):
	"""
	Classe représentant une créature marine avec morphologie et capacités comportementales.
	"""
	
	def __init__(
		self, 
		speciesId: str,
		genomeId: str,
		position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
		age: float = 0.0,
		energy: float = 100.0,
		maxEnergy: float = 100.0,
		size: float = 1.0
	) -> None:
		"""
		Initialise une nouvelle créature.
		
		Args:
			speciesId: Identifiant de l'espèce à laquelle appartient la créature
			genomeId: Identifiant du génome de la créature
			position: Position (x,y,z) initiale dans l'environnement
			age: Âge initial de la créature en unités de temps de simulation
			energy: Énergie initiale de la créature
			maxEnergy: Capacité énergétique maximale
			size: Taille globale de la créature (facteur d'échelle)
		"""
		self.id: str = str(uuid.uuid4())
		self.speciesId: str = speciesId
		self.genomeId: str = genomeId
		self.position: np.ndarray = np.array(position, dtype=np.float32)
		self.velocity: np.ndarray = np.zeros(3, dtype=np.float32)
		self.orientation: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # pitch, yaw, roll
		self.angularVelocity: np.ndarray = np.zeros(3, dtype=np.float32)
		
		# État interne
		self.age: float = age
		self.energy: float = energy
		self.maxEnergy: float = maxEnergy
		self.size: float = size
		self.isAlive: bool = True
		self.fitness: float = 0.0
		self.reproductionReadiness: float = 0.0
		
		# Morphologie
		self.joints: List[Joint] = []
		self.limbs: List[Limb] = []
		self.muscles: List[Muscle] = []
		self.sensors: List[Sensor] = []
		
		# Caractéristiques biologiques
		self.metabolicRate: float = 1.0  # Taux métabolique de base
		self.energyEfficiency: float = 1.0  # Efficacité dans l'utilisation de l'énergie
		self.requiredDepthRange: Tuple[float, float] = (-100.0, 0.0)  # Plage de profondeur optimale
		self.temperatureTolerance: Tuple[float, float] = (5.0, 30.0)  # Plage de température supportée
		self.pressureTolerance: Tuple[float, float] = (1.0, 10.0)  # Plage de pression supportée
		self.lightSensitivity: float = 1.0  # Sensibilité à la lumière
		
		# Système de contrôle
		self.brain: Optional[CreatureBrain] = None
		self.energyModel: EnergyModel = EnergyModel()
		
		# Mémoire d'état
		self.lastAction: Optional[np.ndarray] = None
		self.lastReward: float = 0.0
		self.lastObservation: Optional[np.ndarray] = None
		
		# Statistiques de vie
		self.totalDistanceTraveled: float = 0.0
		self.totalEnergyConsumed: float = 0.0
		self.totalFoodEaten: int = 0
		self.offspring: int = 0
		
	def initializeBrain(self, inputDim: int, outputDim: int, hiddenDim: int = 64) -> None:
		"""
		Initialise le réseau neuronal contrôlant la créature.
		
		Args:
			inputDim: Dimension de l'espace d'observation
			outputDim: Dimension de l'espace d'action
			hiddenDim: Dimension des couches cachées
		"""
		self.brain = CreatureBrain(inputDim, outputDim, hiddenDim)
		
	def addJoint(self, joint: Joint) -> None:
		"""Ajoute une articulation à la créature."""
		self.joints.append(joint)
		
	def addLimb(self, limb: Limb) -> None:
		"""Ajoute un membre à la créature."""
		self.limbs.append(limb)
		
	def addMuscle(self, muscle: Muscle) -> None:
		"""Ajoute un muscle à la créature."""
		self.muscles.append(muscle)
		
	def addSensor(self, sensor: Sensor) -> None:
		"""Ajoute un capteur à la créature."""
		self.sensors.append(sensor)
		
	def getObservation(self, environment: Any) -> np.ndarray:
		"""
		Collecte les observations de l'environnement via les capteurs de la créature.
		
		Args:
			environment: L'environnement marin actuel
			
		Returns:
			Un vecteur d'observation combinant les données de tous les capteurs
		"""
		# Collecter les données des capteurs
		sensorData = []
		for sensor in self.sensors:
			sensorData.extend(sensor.sense(environment, self.position, self.orientation))
			
		# État interne
		internalState = [
			self.energy / self.maxEnergy,  # Niveau d'énergie normalisé
			self.age,
			self.reproductionReadiness
		]
		
		# Position et mouvement
		positionData = [
			*self.position,  # Position x, y, z
			*self.velocity,  # Vélocité vx, vy, vz
			*self.orientation,  # Orientation
		]
		
		# Assembler l'observation complète
		observation = np.concatenate([
			np.array(sensorData, dtype=np.float32),
			np.array(internalState, dtype=np.float32),
			np.array(positionData, dtype=np.float32)
		])
		
		self.lastObservation = observation
		return observation
	
	def act(self, observation: np.ndarray) -> np.ndarray:
		"""
		Sélectionne une action en fonction de l'observation actuelle.
		
		Args:
			observation: Le vecteur d'observation actuel
			
		Returns:
			Un vecteur d'action pour contrôler les muscles
		"""
		if self.brain is None:
			raise ValueError("Le cerveau de la créature n'a pas été initialisé.")
			
		# Convertir en tensor PyTorch
		obsTensor = torch.FloatTensor(observation)
		
		# Obtenir l'action du cerveau de la créature
		with torch.no_grad():
			action = self.brain.selectAction(obsTensor)
		
		self.lastAction = action.numpy() if isinstance(action, Tensor) else action
		return self.lastAction
	
	def applyAction(self, action: np.ndarray) -> None:
		"""
		Applique l'action choisie aux muscles de la créature.
		
		Args:
			action: Vecteur d'action contenant les signaux musculaires
		"""
		if len(action) != len(self.muscles):
			raise ValueError(f"Dimensions d'action incorrectes: {len(action)} ≠ {len(self.muscles)}")
			
		# Appliquer chaque composante d'action au muscle correspondant
		energyUsed = 0.0
		for i, muscle in enumerate(self.muscles):
			# Normaliser l'action entre -1 et 1
			normalizedAction = np.clip(action[i], -1.0, 1.0)
			
			# Calculer la consommation d'énergie pour cette action
			energyCost = muscle.activate(normalizedAction) * self.metabolicRate / self.energyEfficiency
			energyUsed += energyCost
		
		# Soustraire l'énergie utilisée
		self.energy -= energyUsed
		self.totalEnergyConsumed += energyUsed
		
		# Vérifier si la créature est à court d'énergie
		if self.energy <= 0:
			self.energy = 0
			self.isAlive = False
	
	def update(self, deltaTime: float, environment: Any) -> None:
		"""
		Met à jour l'état de la créature pour un pas de temps.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
			environment: L'environnement marin actuel
		"""
		if not self.isAlive:
			return
			
		# Vieillissement
		self.age += deltaTime
		
		# Métabolisme de base
		baseMetabolicCost = self.metabolicRate * deltaTime
		self.energy -= baseMetabolicCost
		self.totalEnergyConsumed += baseMetabolicCost
		
		# Vérifier les conditions environnementales
		self.checkEnvironmentalConditions(environment)
		
		# Mettre à jour la position en fonction de la physique
		self.updatePosition(deltaTime)
		
		# Mettre à jour les statistiques
		distanceTraveled = np.linalg.norm(self.velocity) * deltaTime
		self.totalDistanceTraveled += distanceTraveled
		
		# Mise à jour de la capacité reproductive
		if self.energy > 0.7 * self.maxEnergy:  # Reproduction possible si énergie suffisante
			self.reproductionReadiness += 0.01 * deltaTime
			self.reproductionReadiness = min(1.0, self.reproductionReadiness)
		else:
			self.reproductionReadiness = max(0.0, self.reproductionReadiness - 0.005 * deltaTime)
			
		# Vérification de survie
		if self.energy <= 0 or self.age > self.getMaxAge():
			self.isAlive = False
	
	def updatePosition(self, deltaTime: float) -> None:
		"""
		Met à jour la position et l'orientation de la créature en fonction des forces appliquées.
		
		Args:
			deltaTime: Intervalle de temps depuis la dernière mise à jour
		"""
		# Mise à jour de la position
		self.position += self.velocity * deltaTime
		
		# Mise à jour de l'orientation
		self.orientation += self.angularVelocity * deltaTime
		
		# Normalisation des angles (entre -π et π)
		self.orientation = np.mod(self.orientation + np.pi, 2 * np.pi) - np.pi
	
	def checkEnvironmentalConditions(self, environment: Any) -> None:
		"""
		Vérifie si les conditions environnementales sont favorables à la créature.
		Applique des pénalités d'énergie si les conditions sont défavorables.
		
		Args:
			environment: L'environnement marin actuel
		"""
		# Obtenir les conditions environnementales à la position actuelle
		depth = -self.position[1]  # Profondeur (y vers le haut en général)
		temperature = environment.getTemperatureAt(self.position)
		pressure = environment.getPressureAt(self.position)
		light = environment.getLightLevelAt(self.position)
		
		# Pénalités pour conditions défavorables
		penalties = 0.0
		
		# Pénalité de profondeur
		minDepth, maxDepth = self.requiredDepthRange
		if depth < minDepth or depth > maxDepth:
			depthDiff = min(abs(depth - minDepth), abs(depth - maxDepth))
			penalties += 0.01 * depthDiff * self.metabolicRate
		
		# Pénalité de température
		minTemp, maxTemp = self.temperatureTolerance
		if temperature < minTemp or temperature > maxTemp:
			tempDiff = min(abs(temperature - minTemp), abs(temperature - maxTemp))
			penalties += 0.02 * tempDiff * self.metabolicRate
		
		# Pénalité de pression
		minPressure, maxPressure = self.pressureTolerance
		if pressure < minPressure or pressure > maxPressure:
			pressureDiff = min(abs(pressure - minPressure), abs(pressure - maxPressure))
			penalties += 0.02 * pressureDiff * self.metabolicRate
		
		# Pénalité/bonus de lumière selon la sensibilité
		optimalLight = 0.5  # Niveau de lumière moyen comme référence
		lightDiff = abs(light - optimalLight) * self.lightSensitivity
		penalties += 0.01 * lightDiff * self.metabolicRate
		
		# Appliquer les pénalités
		self.energy -= penalties
		self.totalEnergyConsumed += penalties
	
	def consumeFood(self, energyValue: float) -> None:
		"""
		Consomme de la nourriture et augmente le niveau d'énergie.
		
		Args:
			energyValue: Valeur énergétique de la nourriture
		"""
		self.energy += energyValue
		self.energy = min(self.energy, self.maxEnergy)
		self.totalFoodEaten += 1
	
	def canReproduce(self) -> bool:
		"""
		Détermine si la créature peut se reproduire.
		
		Returns:
			True si la créature peut se reproduire, False sinon
		"""
		return (self.energy > 0.7 * self.maxEnergy and 
				self.reproductionReadiness > 0.8 and 
				self.age > self.getMinReproductionAge())
	
	def reproduce(self) -> None:
		"""
		Effectue les changements d'état nécessaires suite à la reproduction.
		"""
		# Coût énergétique de la reproduction
		reproductionCost = 0.3 * self.maxEnergy
		self.energy -= reproductionCost
		self.totalEnergyConsumed += reproductionCost
		
		# Réinitialiser la préparation à la reproduction
		self.reproductionReadiness = 0.0
		
		# Incrémenter le compteur de descendants
		self.offspring += 1
	
	def getMinReproductionAge(self) -> float:
		"""
		Retourne l'âge minimum de reproduction pour cette créature.
		
		Returns:
			Âge minimum pour la reproduction
		"""
		return 20.0  # Valeur arbitraire, à ajuster selon les besoins
	
	def getMaxAge(self) -> float:
		"""
		Retourne l'âge maximal de la créature.
		
		Returns:
			Âge maximal en unités de temps de simulation
		"""
		return 100.0  # Valeur arbitraire, à ajuster selon les besoins
	
	def calculateFitness(self) -> float:
		"""
		Calcule la valeur d'adaptation (fitness) de la créature.
		
		Returns:
			Score de fitness basé sur les performances de la créature
		"""
		# Facteurs de fitness
		survivalFactor = min(1.0, self.age / 50.0)  # Récompense pour la survie
		efficiencyFactor = self.totalDistanceTraveled / (max(1.0, self.totalEnergyConsumed))  # Efficacité énergétique
		offspringFactor = self.offspring * 10.0  # Récompense pour la reproduction
		adaptationFactor = self.energy / self.maxEnergy  # Niveau d'énergie actuel
		
		# Calcul du fitness global
		self.fitness = (
			0.3 * survivalFactor +
			0.3 * efficiencyFactor +
			0.3 * offspringFactor +
			0.1 * adaptationFactor
		)
		
		return self.fitness
	
	def getMorphologyDescription(self) -> Dict[str, Any]:
		"""
		Génère une description structurée de la morphologie de la créature.
		
		Returns:
			Dictionnaire contenant la description de la morphologie
		"""
		return {
			"size": self.size,
			"joints": [joint.toDict() for joint in self.joints],
			"limbs": [limb.toDict() for limb in self.limbs],
			"muscles": [muscle.toDict() for muscle in self.muscles],
			"sensors": [sensor.toDict() for sensor in self.sensors]
		}
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Creature en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de la créature
		"""
		return {
			"id": self.id,
			"speciesId": self.speciesId,
			"genomeId": self.genomeId,
			"position": self.position.tolist(),
			"velocity": self.velocity.tolist(),
			"orientation": self.orientation.tolist(),
			"angularVelocity": self.angularVelocity.tolist(),
			"age": self.age,
			"energy": self.energy,
			"maxEnergy": self.maxEnergy,
			"size": self.size,
			"isAlive": self.isAlive,
			"fitness": self.fitness,
			"reproductionReadiness": self.reproductionReadiness,
			"metabolicRate": self.metabolicRate,
			"energyEfficiency": self.energyEfficiency,
			"requiredDepthRange": self.requiredDepthRange,
			"temperatureTolerance": self.temperatureTolerance,
			"pressureTolerance": self.pressureTolerance,
			"lightSensitivity": self.lightSensitivity,
			"totalDistanceTraveled": self.totalDistanceTraveled,
			"totalEnergyConsumed": self.totalEnergyConsumed,
			"totalFoodEaten": self.totalFoodEaten,
			"offspring": self.offspring,
			"morphology": self.getMorphologyDescription()
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Creature':
		"""
		Crée une instance de Creature à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de la créature
			
		Returns:
			Instance de Creature reconstruite
		"""
		creature = cls(
			speciesId=data["speciesId"],
			genomeId=data["genomeId"],
			position=tuple(data["position"]),
			age=data["age"],
			energy=data["energy"],
			maxEnergy=data["maxEnergy"],
			size=data["size"]
		)
		
		creature.id = data["id"]
		creature.velocity = np.array(data["velocity"], dtype=np.float32)
		creature.orientation = np.array(data["orientation"], dtype=np.float32)
		creature.angularVelocity = np.array(data["angularVelocity"], dtype=np.float32)
		creature.isAlive = data["isAlive"]
		creature.fitness = data["fitness"]
		creature.reproductionReadiness = data["reproductionReadiness"]
		creature.metabolicRate = data["metabolicRate"]
		creature.energyEfficiency = data["energyEfficiency"]
		creature.requiredDepthRange = tuple(data["requiredDepthRange"])
		creature.temperatureTolerance = tuple(data["temperatureTolerance"])
		creature.pressureTolerance = tuple(data["pressureTolerance"])
		creature.lightSensitivity = data["lightSensitivity"]
		creature.totalDistanceTraveled = data["totalDistanceTraveled"]
		creature.totalEnergyConsumed = data["totalEnergyConsumed"]
		creature.totalFoodEaten = data["totalFoodEaten"]
		creature.offspring = data["offspring"]
		
		# Reconstruction de la morphologie
		morphology = data.get("morphology", {})
		
		# Reconstruction des articulations
		for jointData in morphology.get("joints", []):
			joint = Joint.fromDict(jointData)
			creature.addJoint(joint)
			
		# Reconstruction des membres
		for limbData in morphology.get("limbs", []):
			limb = Limb.fromDict(limbData)
			creature.addLimb(limb)
			
		# Reconstruction des muscles
		for muscleData in morphology.get("muscles", []):
			muscle = Muscle.fromDict(muscleData)
			creature.addMuscle(muscle)
			
		# Reconstruction des capteurs
		for sensorData in morphology.get("sensors", []):
			sensor = Sensor.fromDict(sensorData)
			creature.addSensor(sensor)
		
		return creature