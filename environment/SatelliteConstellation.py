from typing import List, Dict, Tuple, Any, Optional
import numpy as np
import json
import os

from environment.Satellite import Satellite
from environment.GroundStation import GroundStation
from environment.EarthModel import EarthModel
from environment.Atmosphere import Atmosphere
from environment.utils.OrbitalElements import OrbitalElements
from environment.utils.CoordinateTransforms import CoordinateTransforms

class SatelliteConstellation:
	"""
	Classe représentant une constellation complète de satellites.
	Gère la création, l'initialisation et les mises à jour de tous les satellites.
	"""
	
	def __init__(
		self,
		config: Dict[str, Any],
		numSatellites: int,
		earthModel: EarthModel,
		atmosphere: Atmosphere
	):
		"""
		Initialise une constellation de satellites.
		
		Args:
			config: Configuration de la constellation
			numSatellites: Nombre total de satellites
			earthModel: Modèle de la Terre
			atmosphere: Modèle atmosphérique
		"""
		self.config = config
		self.numSatellites = numSatellites
		self.earthModel = earthModel
		self.atmosphere = atmosphere
		
		# Extraire la configuration
		self.numPlanes = config.get("num_planes", 3)
		self.satellitesPerPlane = config.get("satellites_per_plane", 10)
		self.altitude = config.get("altitude", 800.0)  # km
		self.inclination = np.radians(config.get("inclination", 53.0))  # Convertir en radians
		
		# S'assurer que le nombre total de satellites correspond
		if self.numPlanes * self.satellitesPerPlane != self.numSatellites:
			# Ajuster le nombre de plans et de satellites par plan
			self._adjustPlaneConfiguration()
		
		# Liste des satellites
		self.satellites: List[Satellite] = []
		
		# Créer les satellites
		self._createSatellites()
	
	def _adjustPlaneConfiguration(self) -> None:
		"""
		Ajuste la configuration des plans orbitaux pour correspondre au nombre total de satellites.
		"""
		# Factoriser le nombre total de satellites
		factors = []
		for i in range(1, int(np.sqrt(self.numSatellites)) + 1):
			if self.numSatellites % i == 0:
				factors.append(i)
		
		# Trouver le facteur le plus proche du nombre actuel de plans
		bestFactor = factors[-1]  # Prendre le plus grand facteur par défaut
		minDifference = abs(bestFactor - self.numPlanes)
		
		for factor in factors:
			difference = abs(factor - self.numPlanes)
			if difference < minDifference:
				minDifference = difference
				bestFactor = factor
		
		# Mettre à jour la configuration
		self.numPlanes = bestFactor
		self.satellitesPerPlane = self.numSatellites // bestFactor
	
	def _createSatellites(self) -> None:
		"""
		Crée tous les satellites de la constellation.
		"""
		# Obtenir les paramètres par défaut des satellites
		satelliteDefaults = self.config.get("satellite_defaults", {})
		
		# Créer les éléments orbitaux pour chaque plan
		for plane in range(self.numPlanes):
			# Calculer la longitude du nœud ascendant pour ce plan
			longitudeOfAscendingNode = 2 * np.pi * plane / self.numPlanes
			
			# Créer l'orbite pour ce plan
			orbitalElements = OrbitalElements.createConstellationPlane(
				self.satellitesPerPlane,
				self.altitude,
				self.inclination,
				longitudeOfAscendingNode
			)
			
			# Créer les satellites dans ce plan
			for i, orbital in enumerate(orbitalElements):
				satelliteId = plane * self.satellitesPerPlane + i
				
				# Créer le satellite avec ses paramètres
				satellite = Satellite(
					satelliteId=satelliteId,
					initialOrbitalElements=orbital,
					mass=satelliteDefaults.get("mass", 250.0),
					maxBatteryCapacity=satelliteDefaults.get("max_battery_capacity", 1500.0),
					maxFuel=satelliteDefaults.get("max_fuel", 50.0),
					solarPanelArea=satelliteDefaults.get("solar_panel_area", 8.0),
					solarPanelEfficiency=satelliteDefaults.get("solar_panel_efficiency", 0.3),
					dragCoefficient=satelliteDefaults.get("drag_coefficient", 2.2),
					crossSectionalArea=satelliteDefaults.get("cross_sectional_area", 1.5),
					maxThrust=satelliteDefaults.get("max_thrust", 0.1),
					specificImpulse=satelliteDefaults.get("specific_impulse", 220.0),
					powerConsumptionIdle=satelliteDefaults.get("power_consumption_idle", 150.0),
					powerConsumptionTransmit=satelliteDefaults.get("power_consumption_transmit", 450.0),
					powerConsumptionCompute=satelliteDefaults.get("power_consumption_compute", 350.0),
					dataStorageCapacity=satelliteDefaults.get("data_storage_capacity", 256.0),
					communicationBandwidth=satelliteDefaults.get("communication_bandwidth", 150.0),
					computationalCapacity=satelliteDefaults.get("computational_capacity", 5e9),
					thermalProperties=satelliteDefaults.get("thermal_properties", {
						"heat_capacity": 800.0,
						"emissivity": 0.8,
						"absorptivity": 0.3,
						"radiating_area": 9.0
					}),
					faultProbabilities=satelliteDefaults.get("fault_probabilities", {
						"solar_panel_degradation": 0.0001,
						"battery_degradation": 0.00005,
						"communication_failure": 0.0002,
						"computer_failure": 0.0001,
						"attitude_control_failure": 0.0003
					}),
					attitudeControlProperties=satelliteDefaults.get("attitude_control_properties", {
						"control_gain": 0.1,
						"damping": 0.05,
						"max_torque": 0.02
					})
				)
				
				self.satellites.append(satellite)
	
	def reset(self) -> None:
		"""
		Réinitialise tous les satellites de la constellation.
		"""
		# Recréer tous les satellites
		self.satellites = []
		self._createSatellites()
	
	def update(
		self,
		deltaTime: float,
		actions: List[Dict[str, Any]],
		earthModel: EarthModel,
		atmosphere: Atmosphere
	) -> List[Dict[str, float]]:
		"""
		Met à jour tous les satellites de la constellation.
		
		Args:
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
			actions: Liste des actions pour chaque satellite
			earthModel: Modèle de la Terre
			atmosphere: Modèle atmosphérique
			
		Returns:
			Liste des métriques pour chaque satellite
		"""
		# Position de la Terre et du Soleil
		earthPosition = np.array([0.0, 0.0, 0.0])  # La Terre est à l'origine dans le référentiel ECI
		sunPosition = earthModel.sunPosition
		
		# Métriques par satellite
		satelliteMetrics = []
		
		# Mettre à jour chaque satellite
		for i, satellite in enumerate(self.satellites):
			# Obtenir l'action pour ce satellite
			action = actions[i] if i < len(actions) else {}
			
			# Calculer la densité atmosphérique à la position du satellite
			position = satellite.getPosition()
			altitude = np.linalg.norm(position) - earthModel.radius
			atmosphericDensity = atmosphere.getDensity(altitude)
			
			# Mettre à jour l'état du satellite
			satellite.updateState(deltaTime, earthPosition, sunPosition, atmosphericDensity, action)
			
			# Simuler l'apparition de défauts
			satellite.simulateFault(deltaTime)
			
			# Allouer les ressources selon l'action
			allocationMetrics = satellite.allocateResources(action)
			
			# Ajouter d'autres métriques spécifiques au satellite
			allocationMetrics["areasCovered"] = self._calculateSatelliteCoverage(satellite)
			
			# Stocker les métriques
			satelliteMetrics.append(allocationMetrics)
		
		return satelliteMetrics
	
	def updateVisibility(self, groundStations: List[GroundStation]) -> None:
		"""
		Met à jour la visibilité entre tous les satellites et les stations au sol.
		
		Args:
			groundStations: Liste des stations au sol
		"""
		# Mettre à jour la visibilité pour chaque satellite
		for satellite in self.satellites:
			satellite.computeVisibility(self.satellites, groundStations)
	
	def _calculateSatelliteCoverage(self, satellite: Satellite) -> float:
		"""
		Calcule la superficie couverte par un satellite.
		
		Args:
			satellite: Satellite à évaluer
			
		Returns:
			Superficie couverte en km²
		"""
		# Position du satellite
		position = satellite.getPosition()
		altitude = np.linalg.norm(position) - self.earthModel.radius
		
		# Calculer l'angle d'empreinte (footprint)
		footprintAngle = np.arccos(self.earthModel.radius / (self.earthModel.radius + altitude))
		
		# Superficie couverte = 2πr²(1-cos(θ)) où θ est l'angle d'empreinte
		areaCovered = 2 * np.pi * (self.earthModel.radius ** 2) * (1 - np.cos(footprintAngle))
		
		return areaCovered
	
	def getSatelliteById(self, satelliteId: int) -> Optional[Satellite]:
		"""
		Récupère un satellite par son identifiant.
		
		Args:
			satelliteId: Identifiant du satellite
			
		Returns:
			Satellite correspondant ou None si non trouvé
		"""
		for satellite in self.satellites:
			if satellite.satelliteId == satelliteId:
				return satellite
		return None
	
	def getState(self) -> Dict[str, Any]:
		"""
		Renvoie l'état complet de la constellation pour sauvegarde/chargement.
		
		Returns:
			Dictionnaire de l'état complet
		"""
		return {
			"num_planes": self.numPlanes,
			"satellites_per_plane": self.satellitesPerPlane,
			"altitude": self.altitude,
			"inclination": self.inclination,
			"satellites": [
				{
					"satellite_id": satellite.satelliteId,
					"position": satellite.getPosition().tolist(),
					"velocity": satellite.getVelocity().tolist(),
					"orbital_elements": satellite.currentOrbitalElements.toDictionary(),
					"state": {
						"position": satellite.state.position.tolist(),
						"velocity": satellite.state.velocity.tolist(),
						"attitude": satellite.state.attitude.tolist(),
						"angular_velocity": satellite.state.angularVelocity.tolist(),
						"battery_level": satellite.state.batteryLevel,
						"solar_panel_output": satellite.state.solarPanelOutput,
						"temperature": satellite.state.temperature,
						"fuel_remaining": satellite.state.fuelRemaining,
						"time_stamp": satellite.state.timeStamp
					},
					"active_faults": satellite.activeFaults,
					"visible_satellites": satellite.visibleSatellites,
					"visible_ground_stations": satellite.visibleGroundStations
				}
				for satellite in self.satellites
			]
		}
	
	def setState(self, state: Dict[str, Any]) -> None:
		"""
		Restaure l'état complet de la constellation.
		
		Args:
			state: État à restaurer
		"""
		self.numPlanes = state["num_planes"]
		self.satellitesPerPlane = state["satellites_per_plane"]
		self.altitude = state["altitude"]
		self.inclination = state["inclination"]
		
		# Recréer les satellites si nécessaire
		if len(self.satellites) != len(state["satellites"]):
			self.reset()
		
		# Mettre à jour l'état de chaque satellite
		for i, satellite_state in enumerate(state["satellites"]):
			if i < len(self.satellites):
				satellite = self.satellites[i]
				
				# Mettre à jour les défauts actifs
				satellite.activeFaults = satellite_state["active_faults"]
				
				# Mettre à jour les satellites et stations visibles
				satellite.visibleSatellites = satellite_state["visible_satellites"]
				satellite.visibleGroundStations = satellite_state["visible_ground_stations"]
				
				# Reconstruire l'état du satellite
				from environment.Satellite import SatelliteState
				
				state_dict = satellite_state["state"]
				satellite.state = SatelliteState(
					position=np.array(state_dict["position"]),
					velocity=np.array(state_dict["velocity"]),
					attitude=np.array(state_dict["attitude"]),
					angularVelocity=np.array(state_dict["angular_velocity"]),
					batteryLevel=state_dict["battery_level"],
					solarPanelOutput=state_dict["solar_panel_output"],
					temperature=state_dict["temperature"],
					fuelRemaining=state_dict["fuel_remaining"],
					timeStamp=state_dict["time_stamp"]
				)
				
				# Reconstruire les éléments orbitaux
				orbital_dict = satellite_state["orbital_elements"]
				satellite.currentOrbitalElements = OrbitalElements(
					semimajorAxis=orbital_dict["semimajor_axis"],
					eccentricity=orbital_dict["eccentricity"],
					inclination=orbital_dict["inclination"],
					longitudeOfAscendingNode=orbital_dict["longitude_of_ascending_node"],
					argumentOfPeriapsis=orbital_dict["argument_of_periapsis"],
					trueAnomaly=orbital_dict["true_anomaly"]
				)
	
	def getConnectivityGraph(self) -> Tuple[List[int], List[Tuple[int, int]]]:
		"""
		Génère un graphe de connectivité entre tous les satellites.
		
		Returns:
			Tuple (nœuds, arêtes) où les nœuds sont les IDs des satellites
			et les arêtes sont des paires d'IDs de satellites connectés
		"""
		nodes = [sat.satelliteId for sat in self.satellites]
		edges = []
		
		for satellite in self.satellites:
			for visibleId in satellite.visibleSatellites:
				# Ajouter une arête seulement dans une direction pour éviter les doublons
				if satellite.satelliteId < visibleId:
					edges.append((satellite.satelliteId, visibleId))
		
		return nodes, edges
	
	def getCoverageMask(self, resolution: int = 180) -> np.ndarray:
		"""
		Génère une matrice de couverture de la Terre par la constellation.
		
		Args:
			resolution: Résolution de la grille (nombre de divisions par hémisphère)
			
		Returns:
			Matrice 2D de dimensions (2*resolution, 2*resolution) où 1 indique une zone couverte
		"""
		# Créer une grille de couverture
		coverage = np.zeros((2*resolution, 2*resolution))
		
		# Pour chaque point de la grille, vérifier s'il est couvert par au moins un satellite
		for i in range(2*resolution):
			lat = 90 - i * 180 / (2*resolution)  # De 90° à -90°
			
			for j in range(2*resolution):
				lon = -180 + j * 360 / (2*resolution)  # De -180° à 180°
				
				# Convertir les coordonnées géodésiques en cartésiennes
				position = self.earthModel.geodeticToCartesian(lat, lon, 0)
				
				# Vérifier si ce point est couvert par au moins un satellite
				for satellite in self.satellites:
					satPosition = satellite.getPosition()
					altitude = np.linalg.norm(satPosition) - self.earthModel.radius
					
					# Calculer l'angle d'empreinte (footprint)
					footprintAngle = np.arccos(self.earthModel.radius / (self.earthModel.radius + altitude))
					
					# Calculer le vecteur entre le point et le satellite
					vectorToSat = satPosition - position
					
					# Normaliser le vecteur
					vectorToSatNorm = np.linalg.norm(vectorToSat)
					if vectorToSatNorm < 1e-6:
						continue
						
					vectorToSatDir = vectorToSat / vectorToSatNorm
					
					# Calculer l'angle entre la normale à la surface et le vecteur vers le satellite
					surfaceNormal = position / np.linalg.norm(position)
					
					# L'angle est l'arc cosinus du produit scalaire
					angle = np.arccos(np.clip(np.dot(surfaceNormal, vectorToSatDir), -1.0, 1.0))
					
					# Si l'angle est inférieur à π/2 + footprintAngle, le point est couvert
					if angle < np.pi/2 + footprintAngle:
						# Vérifier s'il y a une ligne de vue directe (pas bloquée par la Terre)
						if satellite._hasLineOfSight(position):
							coverage[i, j] = 1
							break
		
		return coverage
	
	def generateSatelliteTasks(self, userDemand: Any) -> None:
		"""
		Génère des tâches pour les satellites en fonction de la demande des utilisateurs.
		
		Args:
			userDemand: Objet de demande des utilisateurs
		"""
		# Obtenir la demande actuelle
		dataRequestRate = userDemand.dataRequestRate
		computationRequestRate = userDemand.computationRequestRate
		communicationRequestRate = userDemand.communicationRequestRate
		
		# Distribuer les tâches entre les satellites
		# Pour simplifier, on répartit uniformément
		numSatellites = len(self.satellites)
		
		if numSatellites == 0:
			return
		
		# Données par satellite
		dataPerSatellite = dataRequestRate / numSatellites
		computePerSatellite = computationRequestRate / numSatellites
		commPerSatellite = communicationRequestRate / numSatellites
		
		# Générer des tâches
		for satellite in self.satellites:
			# Génération aléatoire de 0 à 3 tâches par satellite
			numTasks = np.random.randint(0, 4)
			
			for _ in range(numTasks):
				# Type de tâche aléatoire
				taskType = np.random.choice(["data", "compute", "comm"])
				
				if taskType == "data":
					# Tâche de collecte de données
					task = {
						"type": "data_collection",
						"priority": np.random.uniform(0.5, 1.0),
						"dataSize": np.random.uniform(0.1, 1.0) * dataPerSatellite,
						"computeRequired": np.random.uniform(0.1, 0.3) * satellite.computationalCapacity,
						"bandwidthRequired": np.random.uniform(0.1, 0.3) * satellite.communicationBandwidth,
						"outputSize": np.random.uniform(0.5, 0.9) * dataPerSatellite,
						"deadline": np.random.uniform(300, 3600)  # Entre 5 minutes et 1 heure
					}
				elif taskType == "compute":
					# Tâche de calcul
					task = {
						"type": "computation",
						"priority": np.random.uniform(0.5, 1.0),
						"dataSize": np.random.uniform(0.1, 0.5) * dataPerSatellite,
						"computeRequired": np.random.uniform(0.3, 0.8) * computePerSatellite,
						"bandwidthRequired": np.random.uniform(0.1, 0.2) * satellite.communicationBandwidth,
						"outputSize": np.random.uniform(0.1, 0.3) * dataPerSatellite,
						"deadline": np.random.uniform(300, 3600)
					}
				else:
					# Tâche de communication
					task = {
						"type": "communication",
						"priority": np.random.uniform(0.5, 1.0),
						"dataSize": np.random.uniform(0.5, 1.0) * commPerSatellite,
						"computeRequired": np.random.uniform(0.1, 0.2) * satellite.computationalCapacity,
						"bandwidthRequired": np.random.uniform(0.5, 0.9) * satellite.communicationBandwidth,
						"outputSize": np.random.uniform(0.8, 1.0) * commPerSatellite,
						"deadline": np.random.uniform(300, 3600)
					}
				
				# Ajouter la tâche à la file d'attente du satellite
				satellite.addTask(task)