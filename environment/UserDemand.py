import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

class UserDemand:
	"""
	Modèle de la demande des utilisateurs pour les services de la constellation de satellites.
	Génère des modèles de trafic réalistes pour la simulation.
	"""
	
	def __init__(
		self,
		config: Dict[str, Any] = None
	):
		"""
		Initialise le modèle de demande des utilisateurs.
		
		Args:
			config: Dictionnaire de configuration
		"""
		# Paramètres par défaut
		self.dataRequestRate = 50.0  # GB/jour
		self.computationRequestRate = 20.0  # TFLOPS/jour
		self.communicationRequestRate = 5000.0  # GB/jour
		self.populationBasedWeighting = True
		self.timeVaryingFactor = 0.3
		
		# Charger la configuration si elle est fournie
		if config is not None:
			self.dataRequestRate = config.get("data_request_rate", self.dataRequestRate)
			self.computationRequestRate = config.get("computation_request_rate", self.computationRequestRate)
			self.communicationRequestRate = config.get("communication_request_rate", self.communicationRequestRate)
			self.populationBasedWeighting = config.get("population_based_weighting", self.populationBasedWeighting)
			self.timeVaryingFactor = config.get("time_varying_factor", self.timeVaryingFactor)
		
		# Heure de démarrage
		self.startTime = datetime.now()
		self.currentTime = self.startTime
		self.simulationTime = 0.0
		
		# Variations temporelles et spatiales
		self.timeFactors = self._initializeTimeFactors()
		self.regionFactors = self._initializeRegionFactors()
		
		# Génération de demandes
		self.demandQueue: List[Dict[str, Any]] = []
		self.historicalDemand: List[Dict[str, Any]] = []
		
		# Facteurs actuels
		self.currentTimeFactor = 1.0
		self.currentGlobalFactor = 1.0
	
	def reset(self, simulationTime: float = 0.0) -> None:
		"""
		Réinitialise le modèle de demande des utilisateurs.
		
		Args:
			simulationTime: Temps de simulation en secondes
		"""
		self.startTime = datetime.now()
		self.currentTime = self.startTime + timedelta(seconds=simulationTime)
		self.simulationTime = simulationTime
		
		self.demandQueue = []
		self.historicalDemand = []
		
		# Mise à jour des facteurs
		self._updateFactors()
	
	def update(self, simulationTime: float) -> None:
		"""
		Met à jour le modèle de demande des utilisateurs.
		
		Args:
			simulationTime: Temps de simulation en secondes
		"""
		# Mettre à jour le temps
		deltaTime = simulationTime - self.simulationTime
		self.simulationTime = simulationTime
		self.currentTime = self.startTime + timedelta(seconds=simulationTime)
		
		# Mettre à jour les facteurs
		self._updateFactors()
		
		# Générer de nouvelles demandes
		self._generateDemands(deltaTime)
	
	def _updateFactors(self) -> None:
		"""
		Met à jour les facteurs de variation temporelle et spatiale.
		"""
		# Facteur horaire
		hour = self.currentTime.hour
		minute = self.currentTime.minute
		hourFraction = hour + minute / 60.0
		hourIndex = int(hourFraction)
		
		# Interpolation entre les heures
		nextHourIndex = (hourIndex + 1) % 24
		interpolationFactor = hourFraction - hourIndex
		
		self.currentTimeFactor = (1 - interpolationFactor) * self.timeFactors[hourIndex] + \
								 interpolationFactor * self.timeFactors[nextHourIndex]
		
		# Facteur global (variations hebdomadaires, mensuelles, etc.)
		# Simplification: variation sinusoïdale sur une semaine
		dayOfWeek = self.currentTime.weekday()  # 0 = lundi, 6 = dimanche
		dayFraction = dayOfWeek + hourFraction / 24.0
		weekFraction = dayFraction / 7.0
		
		# Maximum le vendredi après-midi, minimum le dimanche matin
		self.currentGlobalFactor = 1.0 + 0.2 * np.sin(2 * np.pi * (weekFraction - 0.2))
	
	def _generateDemands(self, deltaTime: float) -> None:
		"""
		Génère de nouvelles demandes pour le delta de temps spécifié.
		
		Args:
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
		"""
		if deltaTime <= 0:
			return
		
		# Convertir en jours
		deltaTimeDays = deltaTime / (24 * 3600)
		
		# Appliquer les facteurs de variation
		effectiveFactor = self.currentTimeFactor * self.currentGlobalFactor
		
		# Calcul des demandes pour cette période
		expectedDataRequests = self.dataRequestRate * deltaTimeDays * effectiveFactor
		expectedComputationRequests = self.computationRequestRate * deltaTimeDays * effectiveFactor
		expectedCommunicationRequests = self.communicationRequestRate * deltaTimeDays * effectiveFactor
		
		# Génération de demandes stochastiques (distribution de Poisson)
		numDataRequests = np.random.poisson(expectedDataRequests)
		numComputationRequests = np.random.poisson(expectedComputationRequests)
		numCommunicationRequests = np.random.poisson(expectedCommunicationRequests)
		
		# Générer les demandes individuelles
		self._generateDataRequests(numDataRequests)
		self._generateComputationRequests(numComputationRequests)
		self._generateCommunicationRequests(numCommunicationRequests)
	
	def _generateDataRequests(self, numRequests: int) -> None:
		"""
		Génère des demandes de collecte de données.
		
		Args:
			numRequests: Nombre de demandes à générer
		"""
		for _ in range(numRequests):
			# Sélectionner une région avec pondération selon la population
			if self.populationBasedWeighting:
				regionIndices = list(range(len(self.regionFactors)))
				regionWeights = [region["population_weight"] for region in self.regionFactors]
				regionIndex = np.random.choice(regionIndices, p=regionWeights)
			else:
				regionIndex = np.random.randint(0, len(self.regionFactors))
			
			region = self.regionFactors[regionIndex]
			
			# Générer les caractéristiques de la demande
			request = {
				"type": "data_collection",
				"timestamp": self.simulationTime,
				"region": region["name"],
				"latitude": region["latitude"] + np.random.uniform(-5, 5),
				"longitude": region["longitude"] + np.random.uniform(-5, 5),
				"data_size": np.random.exponential(0.5),  # GB
				"priority": np.random.uniform(0.1, 1.0),
				"deadline": self.simulationTime + np.random.uniform(300, 3600)  # 5 minutes à 1 heure
			}
			
			# Ajouter à la file d'attente
			self.demandQueue.append(request)
			self.historicalDemand.append(request)
	
	def _generateComputationRequests(self, numRequests: int) -> None:
		"""
		Génère des demandes de calcul.
		
		Args:
			numRequests: Nombre de demandes à générer
		"""
		for _ in range(numRequests):
			# Sélectionner une région avec pondération selon la population
			if self.populationBasedWeighting:
				regionIndices = list(range(len(self.regionFactors)))
				regionWeights = [region["population_weight"] for region in self.regionFactors]
				regionIndex = np.random.choice(regionIndices, p=regionWeights)
			else:
				regionIndex = np.random.randint(0, len(self.regionFactors))
			
			region = self.regionFactors[regionIndex]
			
			# Générer les caractéristiques de la demande
			request = {
				"type": "computation",
				"timestamp": self.simulationTime,
				"region": region["name"],
				"latitude": region["latitude"] + np.random.uniform(-5, 5),
				"longitude": region["longitude"] + np.random.uniform(-5, 5),
				"input_data_size": np.random.exponential(0.2),  # GB
				"compute_required": np.random.exponential(0.5),  # TFLOPS
				"priority": np.random.uniform(0.1, 1.0),
				"deadline": self.simulationTime + np.random.uniform(60, 1800)  # 1 minute à 30 minutes
			}
			
			# Ajouter à la file d'attente
			self.demandQueue.append(request)
			self.historicalDemand.append(request)
	
	def _generateCommunicationRequests(self, numRequests: int) -> None:
		"""
		Génère des demandes de communication.
		
		Args:
			numRequests: Nombre de demandes à générer
		"""
		for _ in range(numRequests):
			# Sélectionner une paire de régions (source et destination)
			if self.populationBasedWeighting:
				regionIndices = list(range(len(self.regionFactors)))
				regionWeights = [region["population_weight"] for region in self.regionFactors]
				sourceIndex = np.random.choice(regionIndices, p=regionWeights)
				destIndex = np.random.choice(regionIndices, p=regionWeights)
			else:
				sourceIndex = np.random.randint(0, len(self.regionFactors))
				destIndex = np.random.randint(0, len(self.regionFactors))
			
			sourceRegion = self.regionFactors[sourceIndex]
			destRegion = self.regionFactors[destIndex]
			
			# Générer les caractéristiques de la demande
			request = {
				"type": "communication",
				"timestamp": self.simulationTime,
				"source_region": sourceRegion["name"],
				"source_latitude": sourceRegion["latitude"] + np.random.uniform(-5, 5),
				"source_longitude": sourceRegion["longitude"] + np.random.uniform(-5, 5),
				"dest_region": destRegion["name"],
				"dest_latitude": destRegion["latitude"] + np.random.uniform(-5, 5),
				"dest_longitude": destRegion["longitude"] + np.random.uniform(-5, 5),
				"data_size": np.random.exponential(1.0),  # GB
				"bandwidth_required": np.random.uniform(1, 50),  # Mbps
				"priority": np.random.uniform(0.1, 1.0),
				"deadline": self.simulationTime + np.random.uniform(30, 600)  # 30 secondes à 10 minutes
			}
			
			# Ajouter à la file d'attente
			self.demandQueue.append(request)
			self.historicalDemand.append(request)
	
	def getDemands(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
		"""
		Récupère les demandes actuelles dans la file d'attente.
		
		Args:
			limit: Nombre maximum de demandes à récupérer
			
		Returns:
			Liste des demandes
		"""
		if limit is None or limit >= len(self.demandQueue):
			# Retourner toutes les demandes
			return self.demandQueue
		else:
			# Retourner les N premières demandes
			return self.demandQueue[:limit]
	
	def removeDemand(self, demandId: int) -> None:
		"""
		Supprime une demande de la file d'attente.
		
		Args:
			demandId: Indice de la demande à supprimer
		"""
		if 0 <= demandId < len(self.demandQueue):
			del self.demandQueue[demandId]
	
	def clearExpiredDemands(self) -> int:
		"""
		Supprime les demandes expirées de la file d'attente.
		
		Returns:
			Nombre de demandes supprimées
		"""
		initialCount = len(self.demandQueue)
		
		# Filtrer les demandes non expirées
		self.demandQueue = [demand for demand in self.demandQueue 
							if demand.get("deadline", float('inf')) > self.simulationTime]
		
		return initialCount - len(self.demandQueue)
	
	def getOverallDemandRate(self) -> Dict[str, float]:
		"""
		Calcule le taux de demande global actuel.
		
		Returns:
			Dictionnaire des taux de demande par type
		"""
		# Appliquer les facteurs de variation
		effectiveFactor = self.currentTimeFactor * self.currentGlobalFactor
		
		return {
			"data_collection": self.dataRequestRate * effectiveFactor,
			"computation": self.computationRequestRate * effectiveFactor,
			"communication": self.communicationRequestRate * effectiveFactor
		}
	
	def getDemandDensity(self, latitude: float, longitude: float) -> float:
		"""
		Calcule la densité de demande pour une position géographique donnée.
		
		Args:
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			
		Returns:
			Densité de demande relative (0.0 - 1.0)
		"""
		# Trouver la région la plus proche
		minDistance = float('inf')
		closestRegion = None
		
		for region in self.regionFactors:
			regionLat = region["latitude"]
			regionLon = region["longitude"]
			
			# Distance approximative (pondérée pour prendre en compte que les degrés de longitude
			# sont plus proches à haute latitude)
			latWeight = np.cos(np.radians((latitude + regionLat) / 2))
			lonDiff = min(abs(longitude - regionLon), 360 - abs(longitude - regionLon))
			latDiff = abs(latitude - regionLat)
			distance = np.sqrt((latDiff ** 2) + (lonDiff * latWeight) ** 2)
			
			if distance < minDistance:
				minDistance = distance
				closestRegion = region
		
		if closestRegion is None:
			return 0.0
		
		# Densité basée sur la population et atténuée par la distance
		populationFactor = closestRegion["population_weight"] * len(self.regionFactors)
		distanceFactor = np.exp(-0.05 * minDistance)  # Atténuation exponentielle avec la distance
		
		# Facteur de temps actuel
		timeFactor = self.currentTimeFactor * self.currentGlobalFactor
		
		return populationFactor * distanceFactor * timeFactor
	
	def getRegionalDemand(self) -> List[Dict[str, Any]]:
		"""
		Renvoie les statistiques de demande par région.
		
		Returns:
			Liste des statistiques par région
		"""
		result = []
		
		for region in self.regionFactors:
			# Compter les demandes associées à cette région
			regionName = region["name"]
			dataCount = sum(1 for demand in self.historicalDemand 
						  if demand["type"] == "data_collection" and demand.get("region") == regionName)
			computeCount = sum(1 for demand in self.historicalDemand 
							 if demand["type"] == "computation" and demand.get("region") == regionName)
			commSourceCount = sum(1 for demand in self.historicalDemand 
								if demand["type"] == "communication" and demand.get("source_region") == regionName)
			commDestCount = sum(1 for demand in self.historicalDemand 
							  if demand["type"] == "communication" and demand.get("dest_region") == regionName)
			
			result.append({
				"region": regionName,
				"latitude": region["latitude"],
				"longitude": region["longitude"],
				"population_weight": region["population_weight"],
				"data_requests": dataCount,
				"computation_requests": computeCount,
				"communication_source_requests": commSourceCount,
				"communication_dest_requests": commDestCount,
				"total_requests": dataCount + computeCount + commSourceCount + commDestCount
			})
		
		return result
	
	def getState(self) -> Dict[str, Any]:
		"""
		Retourne l'état actuel du modèle pour sauvegarde/chargement.
		
		Returns:
			Dictionnaire de l'état
		"""
		return {
			"simulation_time": self.simulationTime,
			"current_time": self.currentTime.isoformat(),
			"current_time_factor": self.currentTimeFactor,
			"current_global_factor": self.currentGlobalFactor,
			"data_request_rate": self.dataRequestRate,
			"computation_request_rate": self.computationRequestRate,
			"communication_request_rate": self.communicationRequestRate,
			"demand_queue_size": len(self.demandQueue),
			"historical_demand_size": len(self.historicalDemand)
		}
	
	def setState(self, state: Dict[str, Any]) -> None:
		"""
		Restaure l'état du modèle.
		
		Args:
			state: État à restaurer
		"""
		self.simulationTime = state["simulation_time"]
		self.currentTime = datetime.fromisoformat(state["current_time"])
		self.currentTimeFactor = state["current_time_factor"]
		self.currentGlobalFactor = state["current_global_factor"]
		self.dataRequestRate = state["data_request_rate"]
		self.computationRequestRate = state["computation_request_rate"]
		self.communicationRequestRate = state["communication_request_rate"]
	
	def _initializeTimeFactors(self) -> List[float]:
		"""
		Initialise les facteurs de variation temporelle (par heure).
		
		Returns:
			Liste des facteurs de variation par heure
		"""
		# Modèle de variation diurne:
		# - Minimum pendant la nuit (2h-5h)
		# - Premier pic le matin (8h-10h)
		# - Plateau l'après-midi (14h-16h)
		# - Second pic en soirée (20h-22h)
		
		baseFactors = [
			0.6,  # 0h
			0.4,  # 1h
			0.3,  # 2h
			0.2,  # 3h
			0.2,  # 4h
			0.3,  # 5h
			0.5,  # 6h
			0.7,  # 7h
			1.2,  # 8h
			1.5,  # 9h
			1.3,  # 10h
			1.2,  # 11h
			1.1,  # 12h
			1.0,  # 13h
			1.1,  # 14h
			1.2,  # 15h
			1.3,  # 16h
			1.2,  # 17h
			1.1,  # 18h
			1.0,  # 19h
			1.4,  # 20h
			1.5,  # 21h
			1.2,  # 22h
			0.8   # 23h
		]
		
		# Normaliser pour que la moyenne soit 1.0
		avgFactor = sum(baseFactors) / len(baseFactors)
		normalizedFactors = [factor / avgFactor for factor in baseFactors]
		
		return normalizedFactors
	
	def _initializeRegionFactors(self) -> List[Dict[str, Any]]:
		"""
		Initialise les facteurs de variation spatiale (par région).
		
		Returns:
			Liste des facteurs de variation par région
		"""
		# Définir les principales régions géographiques avec leur population
		regions = [
			{"name": "Amérique du Nord", "latitude": 40.0, "longitude": -100.0, "population": 579e6},
			{"name": "Amérique du Sud", "latitude": -20.0, "longitude": -60.0, "population": 430e6},
			{"name": "Europe", "latitude": 50.0, "longitude": 10.0, "population": 747e6},
			{"name": "Afrique", "latitude": 0.0, "longitude": 20.0, "population": 1340e6},
			{"name": "Moyen-Orient", "latitude": 25.0, "longitude": 45.0, "population": 240e6},
			{"name": "Asie du Sud", "latitude": 20.0, "longitude": 80.0, "population": 1800e6},
			{"name": "Asie de l'Est", "latitude": 35.0, "longitude": 115.0, "population": 1600e6},
			{"name": "Asie du Sud-Est", "latitude": 10.0, "longitude": 105.0, "population": 650e6},
			{"name": "Océanie", "latitude": -25.0, "longitude": 135.0, "population": 42e6},
			{"name": "Arctique", "latitude": 80.0, "longitude": 0.0, "population": 4e6},
			{"name": "Antarctique", "latitude": -80.0, "longitude": 0.0, "population": 0.001e6}
		]
		
		# Calculer les poids de population
		totalPopulation = sum(region["population"] for region in regions)
		
		for region in regions:
			region["population_weight"] = region["population"] / totalPopulation
		
		return regions