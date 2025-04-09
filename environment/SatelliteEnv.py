import gym
from gym import spaces
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os

from environment.SatelliteConstellation import SatelliteConstellation
from environment.Satellite import Satellite, SatelliteState
from environment.EarthModel import EarthModel
from environment.GroundStation import GroundStation
from environment.UserDemand import UserDemand
from environment.Atmosphere import Atmosphere

class SatelliteEnv(gym.Env):
	"""
	Environnement d'apprentissage par renforcement pour gérer une constellation de satellites.
	Implémente l'interface gym pour l'entraînement d'agents.
	"""
	
	def __init__(
		self, 
		configPath: str = None,
		numSatellites: int = 30,
		numGroundStations: int = 20,
		simulationTimeStep: float = 60.0,  # en secondes
		maxEpisodeSteps: int = 1440,  # 24 heures avec un pas de 60 secondes
		renderMode: str = 'none'
	):
		"""
		Initialise l'environnement de simulation.
		
		Args:
			configPath: Chemin vers le fichier de configuration
			numSatellites: Nombre de satellites dans la constellation
			numGroundStations: Nombre de stations au sol
			simulationTimeStep: Pas de temps de simulation en secondes
			maxEpisodeSteps: Nombre maximum d'étapes par épisode
			renderMode: Mode de rendu ('none', '2d', '3d')
		"""
		super().__init__()
		
		# Charger la configuration si elle existe
		self.config = self._loadConfig(configPath)
		
		# Paramètres de simulation
		self.numSatellites = self.config.get("num_satellites", numSatellites)
		self.numGroundStations = self.config.get("num_ground_stations", numGroundStations)
		self.simulationTimeStep = self.config.get("simulation_time_step", simulationTimeStep)
		self.maxEpisodeSteps = self.config.get("max_episode_steps", maxEpisodeSteps)
		self.renderMode = self.config.get("render_mode", renderMode)
		
		# Initialiser les modèles
		self.earthModel = EarthModel()
		self.atmosphere = Atmosphere()
		self.userDemand = UserDemand(self.config.get("user_demand", {}))
		
		# Créer la constellation de satellites
		self.constellation = SatelliteConstellation(
			self.config.get("constellation", {}),
			self.numSatellites,
			self.earthModel,
			self.atmosphere
		)
		
		# Créer les stations au sol
		self.groundStations = self._createGroundStations()
		
		# Variables d'état
		self.currentStep = 0
		self.simulationTime = 0.0  # Temps de simulation en secondes
		self.globalMetrics: Dict[str, float] = {}
		
		# Stockage des historiques
		self.metricHistory: List[Dict[str, float]] = []
		self.stateHistory: List[Dict[str, Any]] = []
		self.actionHistory: List[List[Dict[str, Any]]] = []
		
		# Espaces d'observation et d'action (pour gym)
		self._setupSpaces()
		
		# Variables pour le rendu
		self.renderer = None
		if self.renderMode != 'none':
			self._setupRenderer()
			
		self.reset()
	
	def _loadConfig(self, configPath: Optional[str]) -> Dict[str, Any]:
		"""
		Charge la configuration à partir d'un fichier JSON.
		
		Args:
			configPath: Chemin vers le fichier de configuration
			
		Returns:
			Dictionnaire de configuration
		"""
		defaultConfig = {
			"num_satellites": 30,
			"num_ground_stations": 20,
			"simulation_time_step": 60.0,
			"max_episode_steps": 1440,
			"render_mode": "none",
			"constellation": {
				"num_planes": 3,
				"satellites_per_plane": 10,
				"altitude": 800.0,  # km
				"inclination": 53.0,  # degrés
				"satellite_defaults": {
					"mass": 250.0,  # kg
					"max_battery_capacity": 1500.0,  # Wh
					"max_fuel": 50.0,  # kg
					"solar_panel_area": 8.0,  # m²
					"solar_panel_efficiency": 0.3,
					"drag_coefficient": 2.2,
					"cross_sectional_area": 1.5,  # m²
					"max_thrust": 0.1,  # N
					"specific_impulse": 220.0,  # s
					"power_consumption_idle": 150.0,  # W
					"power_consumption_transmit": 450.0,  # W
					"power_consumption_compute": 350.0,  # W
					"data_storage_capacity": 256.0,  # GB
					"communication_bandwidth": 150.0,  # Mbps
					"computational_capacity": 5e9,  # FLOPS
					"thermal_properties": {
						"heat_capacity": 800.0,  # J/kg·K
						"emissivity": 0.8,
						"absorptivity": 0.3,
						"radiating_area": 9.0  # m²
					},
					"fault_probabilities": {
						"solar_panel_degradation": 0.0001,  # par heure
						"battery_degradation": 0.00005,  # par heure
						"communication_failure": 0.0002,  # par heure
						"computer_failure": 0.0001,  # par heure
						"attitude_control_failure": 0.0003  # par heure
					},
					"attitude_control_properties": {
						"control_gain": 0.1,
						"damping": 0.05,
						"max_torque": 0.02  # N·m
					}
				}
			},
			"ground_stations": {
				"min_latitude": -60.0,  # degrés
				"max_latitude": 60.0,  # degrés
				"bandwidth": 500.0,  # Mbps
				"storage_capacity": 10000.0  # GB
			},
			"user_demand": {
				"data_request_rate": 50.0,  # GB/jour
				"computation_request_rate": 20.0,  # TFLOPS/jour
				"communication_request_rate": 5000.0,  # GB/jour
				"population_based_weighting": True,
				"time_varying_factor": 0.3
			},
			"reward_weights": {
				"coverage": 0.3,
				"data_throughput": 0.3,
				"energy_efficiency": 0.2,
				"network_resilience": 0.1,
				"user_satisfaction": 0.1
			}
		}
		
		if configPath is not None and os.path.exists(configPath):
			try:
				with open(configPath, 'r') as f:
					config = json.load(f)
				
				# Fusionner avec la configuration par défaut
				self._deepUpdate(defaultConfig, config)
				
				return defaultConfig
			except Exception as e:
				print(f"Erreur lors du chargement de la configuration: {e}")
				return defaultConfig
		else:
			return defaultConfig
	
	def _deepUpdate(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
		"""
		Met à jour récursivement un dictionnaire avec les valeurs d'un autre dictionnaire.
		
		Args:
			d: Dictionnaire à mettre à jour
			u: Dictionnaire contenant les nouvelles valeurs
		"""
		for k, v in u.items():
			if isinstance(v, dict) and k in d and isinstance(d[k], dict):
				self._deepUpdate(d[k], v)
			else:
				d[k] = v
	
	def _createGroundStations(self) -> List[GroundStation]:
		"""
		Crée les stations au sol réparties sur la Terre.
		
		Returns:
			Liste des stations au sol
		"""
		stations = []
		
		# Paramètres des stations au sol
		groundConfig = self.config.get("ground_stations", {})
		minLatitude = groundConfig.get("min_latitude", -60.0)
		maxLatitude = groundConfig.get("max_latitude", 60.0)
		bandwidth = groundConfig.get("bandwidth", 500.0)
		storageCapacity = groundConfig.get("storage_capacity", 10000.0)
		
		# Générer des emplacements de stations répartis sur la Terre
		# On utilise une distribution pseudo-aléatoire basée sur les centres de population
		populationCenters = [
			{"name": "New York", "lat": 40.7128, "lon": -74.0060, "importance": 1.0},
			{"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "importance": 0.9},
			{"name": "Chicago", "lat": 41.8781, "lon": -87.6298, "importance": 0.8},
			{"name": "London", "lat": 51.5074, "lon": -0.1278, "importance": 1.0},
			{"name": "Paris", "lat": 48.8566, "lon": 2.3522, "importance": 0.9},
			{"name": "Berlin", "lat": 52.5200, "lon": 13.4050, "importance": 0.8},
			{"name": "Moscow", "lat": 55.7558, "lon": 37.6173, "importance": 0.9},
			{"name": "Beijing", "lat": 39.9042, "lon": 116.4074, "importance": 1.0},
			{"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "importance": 1.0},
			{"name": "Delhi", "lat": 28.6139, "lon": 77.2090, "importance": 0.9},
			{"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "importance": 0.8},
			{"name": "São Paulo", "lat": -23.5505, "lon": -46.6333, "importance": 0.9},
			{"name": "Mexico City", "lat": 19.4326, "lon": -99.1332, "importance": 0.8},
			{"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "importance": 0.7},
			{"name": "Lagos", "lat": 6.5244, "lon": 3.3792, "importance": 0.7},
			{"name": "Johannesburg", "lat": -26.2041, "lon": 28.0473, "importance": 0.7},
			{"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "importance": 0.8},
			{"name": "Melbourne", "lat": -37.8136, "lon": 144.9631, "importance": 0.7},
			{"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816, "importance": 0.7},
			{"name": "Santiago", "lat": -33.4489, "lon": -70.6693, "importance": 0.6}
		]
		
		# Si nous avons besoin de plus de stations que les centres de population prédéfinis
		while len(populationCenters) < self.numGroundStations:
			# Générer des emplacements aléatoires
			lat = np.random.uniform(minLatitude, maxLatitude)
			lon = np.random.uniform(-180.0, 180.0)
			
			populationCenters.append({
				"name": f"Station-{len(populationCenters) + 1}",
				"lat": lat,
				"lon": lon,
				"importance": np.random.uniform(0.3, 0.6)  # Moins important que les grands centres
			})
		
		# Sélectionner les stations en fonction de leur importance
		populationCenters.sort(key=lambda x: x["importance"], reverse=True)
		selectedCenters = populationCenters[:self.numGroundStations]
		
		# Créer les objets GroundStation
		for i, center in enumerate(selectedCenters):
			station = GroundStation(
				stationId=i,
				name=center["name"],
				latitude=center["lat"],
				longitude=center["lon"],
				altitude=0.0,  # Au niveau de la mer
				bandwidth=bandwidth * center["importance"],  # Bande passante proportionnelle à l'importance
				storageCapacity=storageCapacity * center["importance"],  # Stockage proportionnel à l'importance
				earthModel=self.earthModel
			)
			stations.append(station)
		
		return stations
	
	def _setupSpaces(self) -> None:
		"""
		Configure les espaces d'observation et d'action pour gym.
		"""
		# L'espace d'observation est un Dict space avec une entrée par satellite
		# plus des informations globales sur l'environnement
		observationSpaces = {}
		
		# Espace d'observation pour chaque satellite
		for i in range(self.numSatellites):
			# L'observation d'un satellite comprend:
			# - Son état (position, vitesse, attitude, etc.) - 17 valeurs
			# - Ses éléments orbitaux - 6 valeurs
			# - Satellites visibles (one-hot encoding) - numSatellites valeurs
			# - Stations au sol visibles (one-hot encoding) - numGroundStations valeurs
			# - Allocation de ressources - 5 valeurs
			# - File d'attente des tâches - 20 valeurs (5 tâches x 4 caractéristiques)
			# - Défauts actifs - 5 valeurs (one-hot encoding pour chaque type de défaut)
			# - Est éclipsé - 1 valeur
			
			satelliteObsSize = 17 + 6 + self.numSatellites + self.numGroundStations + 5 + 20 + 5 + 1
			
			observationSpaces[f"satellite_{i}"] = spaces.Box(
				low=-float('inf'),
				high=float('inf'),
				shape=(satelliteObsSize,),
				dtype=np.float32
			)
		
		# Espace d'observation global
		# - Temps de simulation - 1 valeur
		# - Métriques globales - 5 valeurs (couverture, débit, efficacité, résilience, satisfaction)
		# - Demande des utilisateurs - 3 valeurs (données, calcul, communication)
		# - Position du Soleil - 3 valeurs
		
		observationSpaces["global"] = spaces.Box(
			low=-float('inf'),
			high=float('inf'),
			shape=(12,),
			dtype=np.float32
		)
		
		self.observation_space = spaces.Dict(observationSpaces)
		
		# Espace d'action
		# Pour chaque satellite, nous avons:
		# - Propulsion: 3 valeurs pour la direction, 1 pour la magnitude
		# - Attitude: 3 valeurs pour le couple
		# - Allocation de puissance: 3 valeurs (transmission, calcul, autres)
		# - Allocation de bande passante: numSatellites + numGroundStations valeurs (proportion allouée à chaque cible)
		# - Priorité des tâches: 5 valeurs (indices de priorité des tâches)
		
		actionSize = 4 + 3 + 3 + (self.numSatellites + self.numGroundStations) + 5
		
		self.action_space = spaces.Box(
			low=-1.0,
			high=1.0,
			shape=(self.numSatellites, actionSize),
			dtype=np.float32
		)
	
	def _setupRenderer(self) -> None:
		"""
		Configure le module de rendu pour la visualisation.
		"""
		if self.renderMode == '2d':
			from visualization.SimulationVisualizer import SimulationVisualizer
			self.renderer = SimulationVisualizer(mode='2d')
		elif self.renderMode == '3d':
			from visualization.SimulationVisualizer import SimulationVisualizer
			self.renderer = SimulationVisualizer(mode='3d')
	
	def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
		"""
		Réinitialise l'environnement pour un nouvel épisode.
		
		Args:
			seed: Graine aléatoire pour la reproductibilité
			options: Options supplémentaires pour la réinitialisation
			
		Returns:
			Observation initiale
		"""
		# Réinitialiser le générateur aléatoire si un seed est fourni
		if seed is not None:
			np.random.seed(seed)
			torch.manual_seed(seed)
		
		# Réinitialiser les variables d'état
		self.currentStep = 0
		self.simulationTime = 0.0
		self.globalMetrics = {
			"coverage": 0.0,
			"data_throughput": 0.0,
			"energy_efficiency": 0.0,
			"network_resilience": 0.0,
			"user_satisfaction": 0.0,
			"totalSatellites": self.numSatellites,
			"totalGroundStations": self.numGroundStations,
			"totalAreasCovered": 0.0,
			"totalDataProcessed": 0.0,
			"totalDataTransmitted": 0.0
		}
		
		# Réinitialiser les historiques
		self.metricHistory = []
		self.stateHistory = []
		self.actionHistory = []
		
		# Réinitialiser les modèles
		self.earthModel.reset(self.simulationTime)
		self.atmosphere.reset()
		self.userDemand.reset(self.simulationTime)
		
		# Réinitialiser la constellation
		self.constellation.reset()
		
		# Réinitialiser les stations au sol
		for station in self.groundStations:
			station.reset()
		
		# Mettre à jour la visibilité initiale
		self.constellation.updateVisibility(self.groundStations)
		
		# Réinitialiser le rendu si nécessaire
		if self.renderer is not None:
			self.renderer.reset(
				self.constellation.satellites,
				self.groundStations,
				self.earthModel
			)
		
		# Retourner l'observation initiale
		return self._getObservation()
	
	def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
		"""
		Exécute une étape de simulation avec les actions données.
		
		Args:
			action: Actions pour tous les satellites
			
		Returns:
			Tuple (observation, reward, done, info)
		"""
		# Convertir les actions brutes en dictionnaires d'actions par satellite
		processedActions = self._processActions(action)
		
		# Enregistrer les actions
		self.actionHistory.append(processedActions)
		
		# Mettre à jour le modèle de la Terre et l'atmosphère
		self.earthModel.update(self.simulationTime)
		self.atmosphere.update(self.simulationTime, self.earthModel)
		
		# Mettre à jour la demande des utilisateurs
		self.userDemand.update(self.simulationTime)
		
		# Mettre à jour la constellation
		satelliteMetrics = self.constellation.update(
			self.simulationTimeStep,
			processedActions,
			self.earthModel,
			self.atmosphere
		)
		
		# Mettre à jour la visibilité entre satellites et stations
		self.constellation.updateVisibility(self.groundStations)
		
		# Mettre à jour les stations au sol
		for station in self.groundStations:
			station.update(self.simulationTimeStep, self.constellation.satellites)
		
		# Calculer les métriques globales
		self._updateGlobalMetrics(satelliteMetrics)
		
		# Mettre à jour le temps de simulation
		self.simulationTime += self.simulationTimeStep
		self.currentStep += 1
		
		# Calculer la récompense
		reward = self._computeReward(satelliteMetrics)
		
		# Déterminer si l'épisode est terminé
		done = self.currentStep >= self.maxEpisodeSteps
		
		# Mettre à jour le rendu si nécessaire
		if self.renderer is not None:
			self.renderer.update(
				self.constellation.satellites,
				self.groundStations,
				self.globalMetrics,
				self.simulationTime
			)
		
		# Obtenir l'observation
		observation = self._getObservation()
		
		# Informations supplémentaires
		info = {
			"simulation_time": self.simulationTime,
			"metrics": self.globalMetrics,
			"satellite_metrics": satelliteMetrics
		}
		
		return observation, reward, done, info
	
	def _processActions(self, action: np.ndarray) -> List[Dict[str, Any]]:
		"""
		Convertit les actions brutes en format utilisable par la constellation.
		
		Args:
			action: Tableau d'actions pour tous les satellites
			
		Returns:
			Liste de dictionnaires d'actions par satellite
		"""
		processedActions = []
		
		for i in range(self.numSatellites):
			satelliteAction = action[i]
			
			# Indices pour extraire les différentes composantes de l'action
			thrustDirStart = 0
			thrustMagIdx = 3
			attitudeTorqueStart = 4
			powerAllocStart = 7
			bandwidthAllocStart = 10
			taskPriorityStart = 10 + self.numSatellites + self.numGroundStations
			
			# Extraire et traiter les composantes
			
			# 1. Propulsion
			thrustDirection = satelliteAction[thrustDirStart:thrustDirStart+3]
			thrustMagnitude = (satelliteAction[thrustMagIdx] + 1.0) / 2.0  # Normaliser de [-1,1] à [0,1]
			
			# 2. Contrôle d'attitude
			attitudeTorque = satelliteAction[attitudeTorqueStart:attitudeTorqueStart+3]
			
			# 3. Allocation de puissance
			powerRaw = satelliteAction[powerAllocStart:powerAllocStart+3]
			# Convertir en proportions positives qui somment à 1
			powerRaw = np.exp(powerRaw)  # Assurer que toutes les valeurs sont positives
			powerSum = np.sum(powerRaw)
			if powerSum > 0:
				powerAllocation = powerRaw / powerSum
			else:
				powerAllocation = np.ones(3) / 3.0
			
			# 4. Allocation de bande passante
			bandwidthRaw = satelliteAction[bandwidthAllocStart:bandwidthAllocStart+self.numSatellites+self.numGroundStations]
			# Convertir en proportions positives qui somment à 1
			bandwidthRaw = np.exp(bandwidthRaw)
			bandwidthSum = np.sum(bandwidthRaw)
			if bandwidthSum > 0:
				bandwidthAllocation = bandwidthRaw / bandwidthSum
			else:
				bandwidthAllocation = np.ones(self.numSatellites+self.numGroundStations) / (self.numSatellites+self.numGroundStations)
			
			# 5. Priorité des tâches
			taskPriorityRaw = satelliteAction[taskPriorityStart:taskPriorityStart+5]
			# Convertir en ordre de priorité (0 à 4)
			taskPriority = np.argsort(-taskPriorityRaw)  # Ordre décroissant
			
			# Créer le dictionnaire d'action pour ce satellite
			satellite_action_dict = {
				"thrustDirection": thrustDirection.tolist(),
				"thrustMagnitude": float(thrustMagnitude),
				"attitudeTorque": attitudeTorque.tolist(),
				"powerDistribution": {
					"transmit": float(powerAllocation[0]),
					"compute": float(powerAllocation[1]),
					"other": float(powerAllocation[2])
				},
				"bandwidthAllocation": {},
				"computeAllocation": {},
				"dataProcessingPriority": taskPriority.tolist()
			}
			
			# Remplir l'allocation de bande passante
			bandwidthAllocDict = {}
			for j in range(self.numSatellites):
				if j != i:  # Pas d'allocation à soi-même
					bandwidthAllocDict[str(j)] = float(bandwidthAllocation[j])
			
			for j in range(self.numGroundStations):
				bandwidthAllocDict[str(j + self.numSatellites)] = float(bandwidthAllocation[j + self.numSatellites])
			
			satellite_action_dict["bandwidthAllocation"] = bandwidthAllocDict
			
			processedActions.append(satellite_action_dict)
		
		return processedActions
	
	def _getObservation(self) -> Dict[str, np.ndarray]:
		"""
		Construit l'observation à partir de l'état actuel de l'environnement.
		
		Returns:
			Dictionnaire d'observation
		"""
		observation = {}
		
		# Obtenir l'observation pour chaque satellite
		satellites = self.constellation.satellites
		for i, satellite in enumerate(satellites):
			observation[f"satellite_{i}"] = np.array(list(satellite.getObservation().values()), dtype=np.float32)
		
		# Observation globale
		globalObs = np.array([
			self.simulationTime,
			self.globalMetrics.get("coverage", 0.0),
			self.globalMetrics.get("data_throughput", 0.0),
			self.globalMetrics.get("energy_efficiency", 0.0),
			self.globalMetrics.get("network_resilience", 0.0),
			self.globalMetrics.get("user_satisfaction", 0.0),
			self.userDemand.dataRequestRate,
			self.userDemand.computationRequestRate,
			self.userDemand.communicationRequestRate,
			self.earthModel.sunPosition[0],
			self.earthModel.sunPosition[1],
			self.earthModel.sunPosition[2]
		], dtype=np.float32)
		
		observation["global"] = globalObs
		
		return observation
	
	def _updateGlobalMetrics(self, satelliteMetrics: List[Dict[str, float]]) -> None:
		"""
		Met à jour les métriques globales à partir des métriques de chaque satellite.
		
		Args:
			satelliteMetrics: Liste des métriques par satellite
		"""
		# Calculer les métriques globales
		totalDataProcessed = sum(metrics.get("dataProcessed", 0.0) for metrics in satelliteMetrics)
		totalDataTransmitted = sum(metrics.get("dataTransmitted", 0.0) for metrics in satelliteMetrics)
		totalPowerAllocated = sum(metrics.get("powerAllocated", 0.0) for metrics in satelliteMetrics)
		totalSolarOutput = sum(sat.state.solarPanelOutput for sat in self.constellation.satellites)
		
		# Calculer la couverture (pourcentage de zones couvertes)
		coveredAreas = set()
		for sat in self.constellation.satellites:
			# Simuler une grille de couverture
			position = sat.getPosition()
			altitude = np.linalg.norm(position) - self.earthModel.radius
			
			# Calculer l'angle d'empreinte (footprint)
			footprintAngle = np.arccos(self.earthModel.radius / (self.earthModel.radius + altitude))
			
			# Convertir la position en latitude/longitude
			lat, lon, _ = self.earthModel.cartesianToGeodetic(position)
			
			# Simuler une grille de 1 degré x 1 degré
			gridSize = 1.0
			
			# Calculer combien de "cases" de la grille sont couvertes par ce satellite
			footprintDegrees = np.degrees(footprintAngle)
			
			# Calculer la zone couverte en degrés
			latRange = min(90, max(-90, lat + footprintDegrees))
			latMin = max(-90, lat - footprintDegrees)
			
			for latIdx in range(int(latMin / gridSize), int(latRange / gridSize) + 1):
				latCovered = latIdx * gridSize
				
				# La couverture en longitude dépend de la latitude
				lonCoverage = footprintDegrees / np.cos(np.radians(abs(latCovered))) if abs(latCovered) < 89 else 180
				lonRange = min(180, lon + lonCoverage)
				lonMin = max(-180, lon - lonCoverage)
				
				for lonIdx in range(int(lonMin / gridSize), int(lonRange / gridSize) + 1):
					lonCovered = lonIdx * gridSize
					coveredAreas.add((latCovered, lonCovered))
		
		# Calculer le pourcentage de couverture (surface totale = 180 * 360 = 64800 degrés carrés)
		totalAreasCovered = len(coveredAreas)
		coveragePercentage = totalAreasCovered / 64800.0
		
		# Calculer le débit de données global
		dataThroughputPercentage = totalDataTransmitted / (self.userDemand.communicationRequestRate + 1e-6)
		dataThroughputPercentage = min(1.0, dataThroughputPercentage)
		
		# Calculer l'efficacité énergétique
		energyEfficiency = 0.0
		if totalSolarOutput > 0:
			energyEfficiency = totalPowerAllocated / totalSolarOutput
			energyEfficiency = min(1.0, energyEfficiency)
		
		# Calculer la résilience du réseau (basée sur le nombre de défauts actifs)
		totalFaults = sum(len(sat.activeFaults) for sat in self.constellation.satellites)
		maxPossibleFaults = self.numSatellites * 5  # 5 types de défauts possibles par satellite
		networkResilience = 1.0 - (totalFaults / maxPossibleFaults) if maxPossibleFaults > 0 else 1.0
		
		# Calculer la satisfaction des utilisateurs (combinaison de couverture et débit)
		userSatisfaction = 0.6 * coveragePercentage + 0.4 * dataThroughputPercentage
		
		# Mettre à jour les métriques globales
		self.globalMetrics = {
			"coverage": coveragePercentage,
			"data_throughput": dataThroughputPercentage,
			"energy_efficiency": energyEfficiency,
			"network_resilience": networkResilience,
			"user_satisfaction": userSatisfaction,
			"totalSatellites": self.numSatellites,
			"totalGroundStations": self.numGroundStations,
			"totalAreasCovered": totalAreasCovered,
			"totalDataProcessed": totalDataProcessed,
			"totalDataTransmitted": totalDataTransmitted
		}
		
		# Enregistrer les métriques
		self.metricHistory.append(self.globalMetrics)
	
	def _computeReward(self, satelliteMetrics: List[Dict[str, float]]) -> float:
		"""
		Calcule la récompense globale à partir des métriques.
		
		Args:
			satelliteMetrics: Liste des métriques par satellite
			
		Returns:
			Valeur de récompense
		"""
		# Obtenir les poids de récompense depuis la configuration
		rewardWeights = self.config.get("reward_weights", {
			"coverage": 0.3,
			"data_throughput": 0.3,
			"energy_efficiency": 0.2,
			"network_resilience": 0.1,
			"user_satisfaction": 0.1
		})
		
		# Calculer la récompense pondérée
		reward = (
			rewardWeights["coverage"] * self.globalMetrics["coverage"] +
			rewardWeights["data_throughput"] * self.globalMetrics["data_throughput"] +
			rewardWeights["energy_efficiency"] * self.globalMetrics["energy_efficiency"] +
			rewardWeights["network_resilience"] * self.globalMetrics["network_resilience"] +
			rewardWeights["user_satisfaction"] * self.globalMetrics["user_satisfaction"]
		)
		
		return reward
	
	def render(self, mode: str = "human"):
		"""
		Rend l'état actuel de l'environnement.
		
		Args:
			mode: Mode de rendu ('human', 'rgb_array')
			
		Returns:
			Image rendue en fonction du mode
		"""
		if self.renderer is None:
			return None
		
		return self.renderer.render(mode)
	
	def close(self):
		"""
		Ferme l'environnement et libère les ressources.
		"""
		if self.renderer is not None:
			self.renderer.close()
	
	def seed(self, seed: Optional[int] = None):
		"""
		Définit la graine aléatoire pour la reproductibilité.
		
		Args:
			seed: Graine aléatoire
			
		Returns:
			Liste contenant la graine
		"""
		if seed is not None:
			np.random.seed(seed)
			torch.manual_seed(seed)
		return [seed]
	
	def getState(self) -> Dict[str, Any]:
		"""
		Renvoie l'état complet de l'environnement pour sauvegarde/chargement.
		
		Returns:
			Dictionnaire de l'état complet
		"""
		return {
			"simulation_time": self.simulationTime,
			"current_step": self.currentStep,
			"global_metrics": self.globalMetrics,
			"earth_state": self.earthModel.getState(),
			"atmosphere_state": self.atmosphere.getState(),
			"user_demand_state": self.userDemand.getState(),
			"constellation_state": self.constellation.getState(),
			"ground_stations_state": [station.getState() for station in self.groundStations]
		}
	
	def setState(self, state: Dict[str, Any]) -> None:
		"""
		Restaure l'état complet de l'environnement.
		
		Args:
			state: État à restaurer
		"""
		self.simulationTime = state["simulation_time"]
		self.currentStep = state["current_step"]
		self.globalMetrics = state["global_metrics"]
		
		self.earthModel.setState(state["earth_state"])
		self.atmosphere.setState(state["atmosphere_state"])
		self.userDemand.setState(state["user_demand_state"])
		self.constellation.setState(state["constellation_state"])
		
		for i, station_state in enumerate(state["ground_stations_state"]):
			if i < len(self.groundStations):
				self.groundStations[i].setState(station_state)
		
		# Mettre à jour le renderer si nécessaire
		if self.renderer is not None:
			self.renderer.update(
				self.constellation.satellites,
				self.groundStations,
				self.globalMetrics,
				self.simulationTime
			)