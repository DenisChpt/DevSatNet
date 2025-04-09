from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import torch
from dataclasses import dataclass
from environment.utils.CoordinateTransforms import CoordinateTransforms
from environment.utils.OrbitalElements import OrbitalElements
from environment.utils.TimeUtils import TimeUtils

@dataclass
class SatelliteState:
	"""Classe pour stocker l'état actuel d'un satellite"""
	position: np.ndarray  # Position en coordonnées cartésiennes [x, y, z] en km
	velocity: np.ndarray  # Vitesse en coordonnées cartésiennes [vx, vy, vz] en km/s
	attitude: np.ndarray  # Quaternion représentant l'attitude [qw, qx, qy, qz]
	angularVelocity: np.ndarray  # Vitesse angulaire [wx, wy, wz] en rad/s
	batteryLevel: float  # Niveau de batterie entre 0.0 et 1.0
	solarPanelOutput: float  # Production d'énergie solaire actuelle en W
	temperature: float  # Température en Kelvin
	fuelRemaining: float  # Carburant restant en kg
	timeStamp: float  # Temps en secondes depuis le début de la simulation
	
	def toTensor(self) -> torch.Tensor:
		"""Convertit l'état en tensor pour l'agent d'apprentissage"""
		return torch.tensor([
			*self.position, 
			*self.velocity, 
			*self.attitude,
			*self.angularVelocity,
			self.batteryLevel,
			self.solarPanelOutput,
			self.temperature,
			self.fuelRemaining
		], dtype=torch.float32)
	
	@classmethod
	def fromTensor(cls, tensor: torch.Tensor, timeStamp: float) -> 'SatelliteState':
		"""Crée un état de satellite à partir d'un tensor"""
		return cls(
			position=tensor[0:3].numpy(),
			velocity=tensor[3:6].numpy(),
			attitude=tensor[6:10].numpy(),
			angularVelocity=tensor[10:13].numpy(),
			batteryLevel=tensor[13].item(),
			solarPanelOutput=tensor[14].item(),
			temperature=tensor[15].item(),
			fuelRemaining=tensor[16].item(),
			timeStamp=timeStamp
		)


class Satellite:
	"""
	Classe représentant un satellite dans la constellation.
	Gère l'état, la dynamique et les ressources du satellite.
	"""
	
	def __init__(
		self, 
		satelliteId: int,
		initialOrbitalElements: OrbitalElements,
		mass: float,
		maxBatteryCapacity: float,
		maxFuel: float,
		solarPanelArea: float,
		solarPanelEfficiency: float,
		dragCoefficient: float,
		crossSectionalArea: float,
		maxThrust: float,
		specificImpulse: float,
		powerConsumptionIdle: float,
		powerConsumptionTransmit: float,
		powerConsumptionCompute: float,
		dataStorageCapacity: float,
		communicationBandwidth: float,
		computationalCapacity: float,
		thermalProperties: Dict[str, float],
		faultProbabilities: Dict[str, float],
		attitudeControlProperties: Dict[str, Any]
	):
		"""
		Initialise un nouveau satellite avec ses paramètres physiques.
		
		Args:
			satelliteId: Identifiant unique du satellite
			initialOrbitalElements: Éléments orbitaux initiaux
			mass: Masse du satellite en kg
			maxBatteryCapacity: Capacité maximale de la batterie en Wh
			maxFuel: Capacité maximale de carburant en kg
			solarPanelArea: Surface des panneaux solaires en m²
			solarPanelEfficiency: Efficacité des panneaux solaires (0-1)
			dragCoefficient: Coefficient de traînée du satellite
			crossSectionalArea: Surface exposée au flux en m²
			maxThrust: Poussée maximale des propulseurs en N
			specificImpulse: Impulsion spécifique des propulseurs en s
			powerConsumptionIdle: Consommation électrique au repos en W
			powerConsumptionTransmit: Consommation électrique en transmission en W
			powerConsumptionCompute: Consommation électrique en calcul en W
			dataStorageCapacity: Capacité de stockage de données en GB
			communicationBandwidth: Bande passante de communication en Mbps
			computationalCapacity: Capacité de calcul en FLOPS
			thermalProperties: Propriétés thermiques du satellite
			faultProbabilities: Probabilités de défaillance des composants
			attitudeControlProperties: Propriétés du système de contrôle d'attitude
		"""
		self.satelliteId: int = satelliteId
		self.mass: float = mass
		self.maxBatteryCapacity: float = maxBatteryCapacity
		self.maxFuel: float = maxFuel
		self.solarPanelArea: float = solarPanelArea
		self.solarPanelEfficiency: float = solarPanelEfficiency
		self.dragCoefficient: float = dragCoefficient
		self.crossSectionalArea: float = crossSectionalArea
		self.maxThrust: float = maxThrust
		self.specificImpulse: float = specificImpulse
		self.powerConsumptionIdle: float = powerConsumptionIdle
		self.powerConsumptionTransmit: float = powerConsumptionTransmit
		self.powerConsumptionCompute: float = powerConsumptionCompute
		self.dataStorageCapacity: float = dataStorageCapacity
		self.communicationBandwidth: float = communicationBandwidth
		self.computationalCapacity: float = computationalCapacity
		self.thermalProperties: Dict[str, float] = thermalProperties
		self.faultProbabilities: Dict[str, float] = faultProbabilities
		self.attitudeControlProperties: Dict[str, Any] = attitudeControlProperties
		
		# Propriétés dérivées
		self.gravitationalParameter: float = 3.986004418e14  # m³/s² (Terre)
		self.earthRadius: float = 6371.0  # km
		self.solarConstant: float = 1361.0  # W/m²
		
		# État actuel
		self.currentOrbitalElements: OrbitalElements = initialOrbitalElements
		
		# Initialiser l'état
		self.state: SatelliteState = self._initializeState()
		
		# Paramètres opérationnels actuels
		self.currentDataStored: float = 0.0  # GB
		self.currentTransmissionPower: float = 0.0  # W
		self.currentComputingPower: float = 0.0  # FLOPS
		self.isEclipsed: bool = False
		self.visibleSatellites: List[int] = []
		self.visibleGroundStations: List[int] = []
		self.activeLinks: List[Tuple[int, float]] = []  # (id, bande passante allouée)
		self.taskQueue: List[Dict[str, Any]] = []
		self.activeFaults: List[str] = []
		
		# Historique des données pour analyse
		self.stateHistory: List[SatelliteState] = []
		self.actionHistory: List[Dict[str, Any]] = []
		self.rewardHistory: List[float] = []
		
	def _initializeState(self) -> SatelliteState:
		"""Initialise l'état du satellite à partir des éléments orbitaux"""
		# Calculer la position et la vitesse à partir des éléments orbitaux
		position, velocity = self.currentOrbitalElements.toPosVel()
		
		# Initialiser une attitude aléatoire mais normalisée
		attitude = np.random.normal(0, 1, 4)
		attitude = attitude / np.linalg.norm(attitude)
		
		return SatelliteState(
			position=position,
			velocity=velocity,
			attitude=attitude,
			angularVelocity=np.zeros(3),
			batteryLevel=0.8,  # Initialisation à 80% de charge
			solarPanelOutput=0.0,
			temperature=293.0,  # 20°C en K
			fuelRemaining=self.maxFuel,
			timeStamp=0.0
		)
	
	def updateState(self, deltaTime: float, earthPosition: np.ndarray, sunPosition: np.ndarray, 
				   atmosphericDensity: float, action: Dict[str, Any]) -> SatelliteState:
		"""
		Met à jour l'état du satellite en fonction du temps écoulé et des actions.
		
		Args:
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
			earthPosition: Position de la Terre dans le référentiel inertiel
			sunPosition: Position du Soleil dans le référentiel inertiel
			atmosphericDensity: Densité atmosphérique locale en kg/m³
			action: Dictionnaire des actions à appliquer
			
		Returns:
			Le nouvel état du satellite
		"""
		# Extraire l'état actuel
		oldPosition = self.state.position
		oldVelocity = self.state.velocity
		oldAttitude = self.state.attitude
		oldAngularVelocity = self.state.angularVelocity
		oldBatteryLevel = self.state.batteryLevel
		oldFuelRemaining = self.state.fuelRemaining
		
		# Appliquer les actions de propulsion
		thrustVector = np.zeros(3)
		fuelConsumed = 0.0
		
		if action.get("thrustMagnitude", 0.0) > 0.0:
			thrustDirection = action.get("thrustDirection", np.zeros(3))
			thrustDirection = thrustDirection / np.linalg.norm(thrustDirection)
			thrustMagnitude = min(action["thrustMagnitude"], self.maxThrust)
			
			thrustVector = thrustDirection * thrustMagnitude
			
			# Calculer la consommation de carburant (équation de la fusée de Tsiolkovsky)
			exhaustVelocity = self.specificImpulse * 9.81  # Ve = Isp * g0
			fuelConsumed = (thrustMagnitude * deltaTime) / exhaustVelocity
			fuelConsumed = min(fuelConsumed, oldFuelRemaining)
			
			# Réduire la masse du satellite
			effectiveMass = self.mass - (oldFuelRemaining - fuelConsumed) / 2.0  # masse moyenne sur l'intervalle
			acceleration = thrustVector / effectiveMass
			
			# Mettre à jour la vitesse
			newVelocity = oldVelocity + acceleration * deltaTime
		else:
			newVelocity = oldVelocity
		
		# Calculer les forces gravitationnelles
		distanceToEarth = np.linalg.norm(oldPosition - earthPosition)
		gravityDirection = (earthPosition - oldPosition) / distanceToEarth
		gravityAcceleration = (self.gravitationalParameter / (distanceToEarth ** 2)) * gravityDirection
		
		# Calculer la traînée atmosphérique
		if atmosphericDensity > 0:
			velocityMagnitude = np.linalg.norm(oldVelocity)
			if velocityMagnitude > 0:
				dragDirection = -oldVelocity / velocityMagnitude
				dragForce = 0.5 * atmosphericDensity * velocityMagnitude**2 * self.dragCoefficient * self.crossSectionalArea
				dragAcceleration = (dragForce / self.mass) * dragDirection
				newVelocity += dragAcceleration * deltaTime
		
		# Mettre à jour la position
		newPosition = oldPosition + newVelocity * deltaTime
		
		# Vérifier si le satellite est éclipsé par la Terre
		satelliteToSunVector = sunPosition - oldPosition
		satelliteToEarthVector = earthPosition - oldPosition
		
		# Angle entre les vecteurs
		dotProduct = np.dot(satelliteToSunVector, satelliteToEarthVector)
		magnitudeProduct = np.linalg.norm(satelliteToSunVector) * np.linalg.norm(satelliteToEarthVector)
		
		if magnitudeProduct > 0:
			angle = np.arccos(dotProduct / magnitudeProduct)
			
			# Si l'angle est inférieur à un certain seuil et que la distance à la Terre est inférieure à la distance au Soleil
			if (angle < 0.01 and np.linalg.norm(satelliteToEarthVector) < np.linalg.norm(satelliteToSunVector)):
				self.isEclipsed = True
			else:
				self.isEclipsed = False
		
		# Calculer la production d'énergie solaire
		solarPanelOutput = 0.0
		if not self.isEclipsed:
			# Calculer l'angle d'incidence de la lumière solaire sur les panneaux solaires
			panelNormal = self._getPanelNormalVector(oldAttitude)
			sunDirection = satelliteToSunVector / np.linalg.norm(satelliteToSunVector)
			incidenceAngle = np.arccos(np.clip(np.dot(panelNormal, sunDirection), -1.0, 1.0))
			
			# La production est maximale quand l'angle d'incidence est de 0 (normale du panneau orientée vers le soleil)
			# et nulle quand l'angle est de π/2 ou plus
			if incidenceAngle < np.pi/2:
				effectiveArea = self.solarPanelArea * np.cos(incidenceAngle)
				solarPanelOutput = effectiveArea * self.solarPanelEfficiency * self.solarConstant
		
		# Calculer la consommation d'énergie
		powerConsumption = self.powerConsumptionIdle
		
		# Ajouter la consommation pour la transmission
		transmitPower = action.get("transmitPower", 0.0)
		transmitPower = min(transmitPower, 1.0) * self.powerConsumptionTransmit
		powerConsumption += transmitPower
		
		# Ajouter la consommation pour le calcul
		computePower = action.get("computePower", 0.0)
		computePower = min(computePower, 1.0) * self.powerConsumptionCompute
		powerConsumption += computePower
		
		# Mettre à jour le niveau de batterie
		energyDelta = (solarPanelOutput - powerConsumption) * deltaTime / 3600.0  # convertir en Wh
		newBatteryLevel = oldBatteryLevel + energyDelta / self.maxBatteryCapacity
		newBatteryLevel = np.clip(newBatteryLevel, 0.0, 1.0)
		
		# Appliquer les actions de contrôle d'attitude
		attitudeTorque = action.get("attitudeTorque", np.zeros(3))
		
		# Gain de contrôle d'attitude depuis les propriétés
		attitudeControlGain = self.attitudeControlProperties.get("controlGain", 0.1)
		attitudeDamping = self.attitudeControlProperties.get("damping", 0.05)
		
		# Calculer la nouvelle vitesse angulaire avec les couples appliqués et l'amortissement
		newAngularVelocity = oldAngularVelocity + attitudeTorque * attitudeControlGain * deltaTime
		newAngularVelocity *= (1.0 - attitudeDamping * deltaTime)  # Appliquer l'amortissement
		
		# Mettre à jour l'attitude (quaternion) en fonction de la vitesse angulaire
		angularSpeed = np.linalg.norm(newAngularVelocity)
		
		if angularSpeed > 1e-6:  # Éviter la division par zéro
			axis = newAngularVelocity / angularSpeed
			angle = angularSpeed * deltaTime
			
			# Quaternion de rotation pour l'incrément d'attitude
			cosHalfAngle = np.cos(angle / 2.0)
			sinHalfAngle = np.sin(angle / 2.0)
			
			rotQuat = np.array([
				cosHalfAngle,
				sinHalfAngle * axis[0],
				sinHalfAngle * axis[1],
				sinHalfAngle * axis[2]
			])
			
			# Multiplier les quaternions
			newAttitude = self._multiplyQuaternions(oldAttitude, rotQuat)
			# Normaliser pour éviter les erreurs d'accumulation
			newAttitude = newAttitude / np.linalg.norm(newAttitude)
		else:
			newAttitude = oldAttitude
		
		# Mettre à jour les paramètres opérationnels
		self.currentTransmissionPower = transmitPower
		self.currentComputingPower = computePower
		
		# Créer et retourner le nouvel état
		newState = SatelliteState(
			position=newPosition,
			velocity=newVelocity,
			attitude=newAttitude,
			angularVelocity=newAngularVelocity,
			batteryLevel=newBatteryLevel,
			solarPanelOutput=solarPanelOutput,
			temperature=self._calculateTemperature(
				self.state.temperature, 
				solarPanelOutput, 
				powerConsumption, 
				self.isEclipsed, 
				deltaTime
			),
			fuelRemaining=oldFuelRemaining - fuelConsumed,
			timeStamp=self.state.timeStamp + deltaTime
		)
		
		# Mettre à jour l'état courant et l'historique
		self.stateHistory.append(self.state)
		self.state = newState
		
		# Mettre à jour les éléments orbitaux à partir de la nouvelle position et vitesse
		self.currentOrbitalElements = OrbitalElements.fromPosVel(
			newPosition, newVelocity, self.gravitationalParameter)
		
		# Enregistrer l'action
		self.actionHistory.append(action)
		
		return newState
	
	def _calculateTemperature(self, currentTemp: float, solarInput: float, powerConsumption: float, 
							 isEclipsed: bool, deltaTime: float) -> float:
		"""Calcule la température du satellite en fonction des flux d'énergie"""
		# Récupérer les propriétés thermiques
		heatCapacity = self.thermalProperties.get("heatCapacity", 800.0)  # J/kg·K
		emissivity = self.thermalProperties.get("emissivity", 0.8)
		absorptivity = self.thermalProperties.get("absorptivity", 0.3)
		radiatingArea = self.thermalProperties.get("radiatingArea", self.crossSectionalArea * 6)  # m²
		
		# Constante de Stefan-Boltzmann
		stefanBoltzmann = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
		
		# Chaleur reçue du Soleil (W)
		solarHeatInput = solarInput * absorptivity / self.solarPanelEfficiency if not isEclipsed else 0.0
		
		# Chaleur générée par l'électronique (W)
		internalHeatGeneration = powerConsumption * 0.15  # 15% de la puissance devient de la chaleur
		
		# Chaleur rayonnée vers l'espace (W)
		radiatedHeat = emissivity * stefanBoltzmann * radiatingArea * (currentTemp ** 4)
		
		# Bilan thermique (W)
		netHeatFlow = solarHeatInput + internalHeatGeneration - radiatedHeat
		
		# Variation de température (K)
		tempChange = netHeatFlow * deltaTime / (self.mass * heatCapacity)
		
		# Nouvelle température (K)
		newTemp = currentTemp + tempChange
		
		return newTemp
	
	def _getPanelNormalVector(self, attitude: np.ndarray) -> np.ndarray:
		"""Calcule la normale aux panneaux solaires en fonction de l'attitude"""
		# On suppose que les panneaux solaires sont orientés selon l'axe +Y du satellite
		# Convertir le quaternion d'attitude pour obtenir la matrice de rotation
		qw, qx, qy, qz = attitude
		
		# Matrice de rotation à partir du quaternion
		rotationMatrix = np.array([
			[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
			[2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
			[2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
		])
		
		# Vecteur normal aux panneaux dans le référentiel du satellite
		panelNormalBodyFrame = np.array([0.0, 1.0, 0.0])
		
		# Transformer dans le référentiel inertiel
		panelNormalInertialFrame = rotationMatrix @ panelNormalBodyFrame
		
		return panelNormalInertialFrame
	
	def _multiplyQuaternions(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
		"""Multiplie deux quaternions"""
		w1, x1, y1, z1 = q1
		w2, x2, y2, z2 = q2
		
		return np.array([
			w1*w2 - x1*x2 - y1*y2 - z1*z2,
			w1*x2 + x1*w2 + y1*z2 - z1*y2,
			w1*y2 - x1*z2 + y1*w2 + z1*x2,
			w1*z2 + x1*y2 - y1*x2 + z1*w2
		])
	
	def computeVisibility(self, satellites: List['Satellite'], groundStations: List['GroundStation']) -> None:
		"""
		Calcule les satellites et stations au sol visibles depuis ce satellite.
		
		Args:
			satellites: Liste des autres satellites de la constellation
			groundStations: Liste des stations au sol
		"""
		self.visibleSatellites = []
		self.visibleGroundStations = []
		
		# Position du satellite
		satPos = self.state.position
		
		# Vérifier la visibilité des autres satellites
		for otherSat in satellites:
			if otherSat.satelliteId == self.satelliteId:
				continue  # Ignorer le satellite lui-même
				
			otherPos = otherSat.state.position
			distance = np.linalg.norm(satPos - otherPos)
			
			# Vérifier si le satellite est dans la plage de communication
			maxCommRange = 2000.0  # km, à paramétrer selon le type de satellite
			
			if distance <= maxCommRange:
				# Vérifier s'il y a une ligne de vue directe (pas bloquée par la Terre)
				if self._hasLineOfSight(otherPos):
					self.visibleSatellites.append(otherSat.satelliteId)
		
		# Vérifier la visibilité des stations au sol
		for station in groundStations:
			stationPos = station.getPosition()
			
			# Vérifier si la station est dans le champ de vue du satellite
			if self._isGroundStationVisible(stationPos):
				self.visibleGroundStations.append(station.stationId)
	
	def _hasLineOfSight(self, targetPosition: np.ndarray) -> bool:
		"""
		Vérifie s'il y a une ligne de vue directe entre ce satellite et une position cible.
		
		Args:
			targetPosition: Position de la cible en coordonnées cartésiennes
			
		Returns:
			True s'il y a une ligne de vue directe, False sinon
		"""
		# Position du satellite
		satellitePosition = self.state.position
		
		# Vecteur de la ligne de vue
		lineOfSightVector = targetPosition - satellitePosition
		distance = np.linalg.norm(lineOfSightVector)
		
		if distance < 1e-6:  # Éviter la division par zéro
			return True
			
		# Normaliser le vecteur
		lineOfSightDir = lineOfSightVector / distance
		
		# Calculer la distance la plus proche de la ligne à la Terre (origine)
		# Projection du vecteur de l'origine au satellite sur la ligne de vue
		projectionLength = np.dot(-satellitePosition, lineOfSightDir)
		
		# Point le plus proche de l'origine sur la ligne de vue
		closestPoint = satellitePosition + projectionLength * lineOfSightDir
		
		# Distance de ce point à l'origine
		closestDistance = np.linalg.norm(closestPoint)
		
		# Si cette distance est inférieure au rayon de la Terre, la ligne de vue est obstruée
		return closestDistance > self.earthRadius
	
	def _isGroundStationVisible(self, stationPosition: np.ndarray) -> bool:
		"""
		Vérifie si une station au sol est visible depuis le satellite.
		
		Args:
			stationPosition: Position de la station au sol en coordonnées cartésiennes
			
		Returns:
			True si la station est visible, False sinon
		"""
		# Calcul du vecteur entre la station et le satellite
		vectorToSatellite = self.state.position - stationPosition
		distance = np.linalg.norm(vectorToSatellite)
		
		# Normaliser le vecteur
		directionToSatellite = vectorToSatellite / distance
		
		# Calculer l'angle d'élévation
		# D'abord, normaliser la position de la station
		stationNorm = np.linalg.norm(stationPosition)
		if stationNorm < 1e-6:  # Éviter la division par zéro
			return False
			
		stationDir = stationPosition / stationNorm
		
		# L'angle d'élévation est l'angle entre la direction locale "up" (stationDir)
		# et la direction vers le satellite
		dotProduct = np.dot(stationDir, directionToSatellite)
		elevationAngle = np.arcsin(np.clip(dotProduct, -1.0, 1.0))
		
		# Convertir en degrés pour la lisibilité
		elevationDegrees = np.degrees(elevationAngle)
		
		# Typiquement, les satellites ne sont visibles que si l'angle d'élévation est supérieur à 10°
		minimumElevation = 10.0  # degrés
		
		return elevationDegrees >= minimumElevation and self._hasLineOfSight(stationPosition)
	
	def allocateResources(self, action: Dict[str, Any]) -> Dict[str, float]:
		"""
		Alloue les ressources du satellite selon les actions spécifiées.
		
		Args:
			action: Dictionnaire contenant les actions d'allocation de ressources
			
		Returns:
			Dictionnaire contenant les métriques d'allocation
		"""
		# Extraire les actions d'allocation
		powerDistribution = action.get("powerDistribution", {})
		bandwidthAllocation = action.get("bandwidthAllocation", {})
		computeAllocation = action.get("computeAllocation", {})
		dataProcessingPriority = action.get("dataProcessingPriority", [])
		
		# Vérifier si le niveau de batterie est suffisant
		if self.state.batteryLevel < 0.1:  # Mode de survie si la batterie est trop basse
			return {
				"powerAllocated": 0.0,
				"bandwidthAllocated": 0.0,
				"computeAllocated": 0.0,
				"dataProcessed": 0.0,
				"dataTransmitted": 0.0,
				"linkQuality": 0.0
			}
		
		# Allouer la puissance aux différents sous-systèmes
		totalPowerAllocated = sum(powerDistribution.values())
		powerScalingFactor = 1.0
		
		# Si l'allocation demandée dépasse l'énergie disponible, réduire proportionnellement
		maxAvailablePower = self.state.batteryLevel * self.maxBatteryCapacity * 0.9  # Garder 10% de marge
		if totalPowerAllocated > maxAvailablePower:
			powerScalingFactor = maxAvailablePower / totalPowerAllocated
			
		# Allouer la bande passante aux liens de communication
		self.activeLinks = []
		totalBandwidthAllocated = 0.0
		
		for targetId, allocation in bandwidthAllocation.items():
			# Vérifier si la cible est visible
			if int(targetId) in self.visibleSatellites or int(targetId) in self.visibleGroundStations:
				# La bande passante allouée est proportionnelle à l'allocation demandée
				allocatedBandwidth = allocation * self.communicationBandwidth * powerScalingFactor
				self.activeLinks.append((int(targetId), allocatedBandwidth))
				totalBandwidthAllocated += allocatedBandwidth
		
		# Traiter les données selon les priorités
		dataProcessed = 0.0
		dataTransmitted = 0.0
		
		for taskId in dataProcessingPriority:
			if taskId < len(self.taskQueue):
				task = self.taskQueue[taskId]
				
				# Vérifier si le satellite a assez de ressources pour traiter cette tâche
				requiredCompute = task.get("computeRequired", 0.0)
				requiredBandwidth = task.get("bandwidthRequired", 0.0)
				
				if (requiredCompute <= self.computationalCapacity * powerScalingFactor and
					requiredBandwidth <= totalBandwidthAllocated):
					# Traiter la tâche
					dataProcessed += task.get("dataSize", 0.0)
					dataTransmitted += task.get("outputSize", 0.0)
					
					# Réduire les ressources disponibles
					totalBandwidthAllocated -= requiredBandwidth
					
					# Marquer la tâche comme traitée
					task["processed"] = True
		
		# Nettoyer la file d'attente des tâches traitées
		self.taskQueue = [task for task in self.taskQueue if not task.get("processed", False)]
		
		# Calculer la qualité moyenne des liens
		linkQuality = 0.0
		if len(self.activeLinks) > 0:
			# La qualité du lien est une fonction de la distance et de l'allocation de bande passante
			totalQuality = 0.0
			for targetId, bandwidth in self.activeLinks:
				# Simuler la qualité du lien en fonction de la bande passante allouée
				qualityFactor = bandwidth / (self.communicationBandwidth * 0.1)  # 10% est considéré comme bon
				qualityFactor = min(1.0, qualityFactor)  # Plafonner à 1.0
				totalQuality += qualityFactor
				
			linkQuality = totalQuality / len(self.activeLinks)
		
		return {
			"powerAllocated": totalPowerAllocated * powerScalingFactor,
			"bandwidthAllocated": sum(bw for _, bw in self.activeLinks),
			"computeAllocated": dataProcessed * requiredCompute / (self.computationalCapacity + 1e-6),
			"dataProcessed": dataProcessed,
			"dataTransmitted": dataTransmitted,
			"linkQuality": linkQuality
		}
	
	def addTask(self, task: Dict[str, Any]) -> bool:
		"""
		Ajoute une tâche à la file d'attente du satellite.
		
		Args:
			task: Dictionnaire décrivant la tâche
			
		Returns:
			True si la tâche a été ajoutée avec succès, False sinon
		"""
		# Vérifier si le satellite a assez d'espace de stockage
		currentStorageUsed = sum(task.get("dataSize", 0.0) for task in self.taskQueue)
		
		if currentStorageUsed + task.get("dataSize", 0.0) <= self.dataStorageCapacity:
			self.taskQueue.append(task)
			return True
		else:
			# Pas assez d'espace disponible
			return False
	
	def getObservation(self) -> Dict[str, Any]:
		"""
		Renvoie une observation complète de l'état du satellite pour l'agent d'apprentissage.
		
		Returns:
			Dictionnaire d'observation
		"""
		# Convertir l'état en tenseur
		stateTensor = self.state.toTensor()
		
		# Créer des tenseurs pour les satellites et stations visibles (one-hot encoding)
		maxSatellites = 100  # Supposons un maximum de 100 satellites dans la constellation
		maxGroundStations = 50  # Supposons un maximum de 50 stations au sol
		
		visibleSatTensor = torch.zeros(maxSatellites)
		for satId in self.visibleSatellites:
			if 0 <= satId < maxSatellites:
				visibleSatTensor[satId] = 1.0
				
		visibleGSTensor = torch.zeros(maxGroundStations)
		for gsId in self.visibleGroundStations:
			if 0 <= gsId < maxGroundStations:
				visibleGSTensor[gsId] = 1.0
		
		# Créer un tenseur pour représenter l'allocation actuelle des ressources
		resourceAllocationTensor = torch.tensor([
			self.currentTransmissionPower / self.powerConsumptionTransmit,
			self.currentComputingPower / self.powerConsumptionCompute,
			len(self.activeLinks) / max(1, len(self.visibleSatellites) + len(self.visibleGroundStations)),
			self.currentDataStored / self.dataStorageCapacity,
			len(self.taskQueue)
		])
		
		# Créer un tenseur pour représenter les 5 premières tâches de la file d'attente
		taskQueueTensor = torch.zeros(5, 4)  # 5 tâches, 4 caractéristiques par tâche
		
		for i, task in enumerate(self.taskQueue[:5]):
			taskQueueTensor[i, 0] = task.get("priority", 0.0)
			taskQueueTensor[i, 1] = task.get("dataSize", 0.0) / self.dataStorageCapacity
			taskQueueTensor[i, 2] = task.get("computeRequired", 0.0) / self.computationalCapacity
			taskQueueTensor[i, 3] = task.get("bandwidthRequired", 0.0) / self.communicationBandwidth
		
		# Créer une représentation des défauts actifs
		faultsTensor = torch.zeros(len(self.faultProbabilities))
		for i, fault in enumerate(self.activeFaults):
			if fault in list(self.faultProbabilities.keys()):
				idx = list(self.faultProbabilities.keys()).index(fault)
				faultsTensor[idx] = 1.0
		
		return {
			"state": stateTensor,
			"orbital_elements": torch.tensor([
				self.currentOrbitalElements.semimajorAxis,
				self.currentOrbitalElements.eccentricity,
				self.currentOrbitalElements.inclination,
				self.currentOrbitalElements.longitudeOfAscendingNode,
				self.currentOrbitalElements.argumentOfPeriapsis,
				self.currentOrbitalElements.trueAnomaly
			]),
			"visible_satellites": visibleSatTensor,
			"visible_ground_stations": visibleGSTensor,
			"resource_allocation": resourceAllocationTensor,
			"task_queue": taskQueueTensor.flatten(),
			"faults": faultsTensor,
			"is_eclipsed": torch.tensor([float(self.isEclipsed)])
		}
	
	def computeReward(self, metrics: Dict[str, float], globalMetrics: Dict[str, float]) -> float:
		"""
		Calcule la récompense pour l'agent d'apprentissage.
		
		Args:
			metrics: Métriques spécifiques à ce satellite
			globalMetrics: Métriques globales du système
			
		Returns:
			Valeur de récompense
		"""
		# Pondérations des différentes composantes de la récompense
		weights = {
			"power_efficiency": 0.2,
			"data_processed": 0.3,
			"communication_quality": 0.3,
			"orbit_stability": 0.1,
			"system_contribution": 0.1
		}
		
		# Calcul de l'efficacité énergétique
		powerEfficiency = 0.0
		if self.state.solarPanelOutput > 0:
			powerEfficiency = metrics.get("powerAllocated", 0.0) / self.state.solarPanelOutput
			powerEfficiency = min(powerEfficiency, 1.0)  # Plafonner à 1.0
		
		# Récompense pour le traitement de données
		dataReward = metrics.get("dataProcessed", 0.0) / (self.dataStorageCapacity * 0.1)  # 10% traité est bon
		dataReward = min(dataReward, 1.0)  # Plafonner à 1.0
		
		# Qualité de la communication
		commQuality = metrics.get("linkQuality", 0.0)
		
		# Stabilité orbitale - pénaliser les changements d'orbite non nécessaires
		orbitStability = 1.0
		if len(self.stateHistory) > 10:  # Besoin d'historique pour évaluer la stabilité
			# Comparer les éléments orbitaux actuels avec ceux d'il y a 10 pas de temps
			oldOrbit = OrbitalElements.fromPosVel(
				self.stateHistory[-10].position,
				self.stateHistory[-10].velocity,
				self.gravitationalParameter
			)
			
			# Calculer le changement relatif des éléments orbitaux
			deltaSMA = abs(self.currentOrbitalElements.semimajorAxis - oldOrbit.semimajorAxis) / oldOrbit.semimajorAxis
			deltaEcc = abs(self.currentOrbitalElements.eccentricity - oldOrbit.eccentricity) / max(0.001, oldOrbit.eccentricity)
			deltaInc = abs(self.currentOrbitalElements.inclination - oldOrbit.inclination) / max(0.001, oldOrbit.inclination)
			
			# Moyenne des changements
			orbitChangeMagnitude = (deltaSMA + deltaEcc + deltaInc) / 3.0
			
			# La stabilité diminue avec l'ampleur des changements
			orbitStability = np.exp(-5.0 * orbitChangeMagnitude)
		
		# Contribution au système global
		systemContribution = 0.0
		if globalMetrics.get("totalSatellites", 0) > 0:
			# Contribution à la couverture globale
			coverageContribution = metrics.get("areasCovered", 0.0) / globalMetrics.get("totalAreasCovered", 1.0)
			
			# Contribution au traitement de données
			dataContribution = metrics.get("dataProcessed", 0.0) / globalMetrics.get("totalDataProcessed", 1.0)
			
			# Contribution à la transmission de données
			transmissionContribution = metrics.get("dataTransmitted", 0.0) / globalMetrics.get("totalDataTransmitted", 1.0)
			
			# Moyenne pondérée des contributions
			systemContribution = (coverageContribution + 2*dataContribution + 2*transmissionContribution) / 5.0
		
		# Calculer la récompense totale
		reward = (
			weights["power_efficiency"] * powerEfficiency +
			weights["data_processed"] * dataReward +
			weights["communication_quality"] * commQuality +
			weights["orbit_stability"] * orbitStability +
			weights["system_contribution"] * systemContribution
		)
		
		# Stocker la récompense pour analyse
		self.rewardHistory.append(reward)
		
		return reward
	
	def simulateFault(self, deltaTime: float) -> List[str]:
		"""
		Simule l'apparition de défauts en fonction des probabilités.
		
		Args:
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
			
		Returns:
			Liste des nouveaux défauts
		"""
		newFaults = []
		
		# Pour chaque type de défaut possible
		for faultType, probability in self.faultProbabilities.items():
			# Si le défaut n'est pas déjà actif
			if faultType not in self.activeFaults:
				# Probabilité de défaut par seconde (conversion de la probabilité par heure)
				probPerSecond = 1.0 - (1.0 - probability) ** (deltaTime / 3600.0)
				
				# Tirage aléatoire
				if np.random.random() < probPerSecond:
					self.activeFaults.append(faultType)
					newFaults.append(faultType)
		
		return newFaults
	
	def repairFault(self, faultType: str) -> bool:
		"""
		Tente de réparer un défaut.
		
		Args:
			faultType: Type de défaut à réparer
			
		Returns:
			True si la réparation a réussi, False sinon
		"""
		if faultType in self.activeFaults:
			# Simuler une probabilité de réparation réussie
			repairSuccess = np.random.random() < 0.7  # 70% de chance de succès
			
			if repairSuccess:
				self.activeFaults.remove(faultType)
			
			return repairSuccess
		
		return False  # Le défaut n'était pas actif
	
	def getState(self) -> SatelliteState:
		"""
		Renvoie l'état actuel du satellite.
		
		Returns:
			État actuel du satellite
		"""
		return self.state
	
	def getPosition(self) -> np.ndarray:
		"""
		Renvoie la position actuelle du satellite.
		
		Returns:
			Position en coordonnées cartésiennes
		"""
		return self.state.position
	
	def getVelocity(self) -> np.ndarray:
		"""
		Renvoie la vitesse actuelle du satellite.
		
		Returns:
			Vitesse en coordonnées cartésiennes
		"""
		return self.state.velocity