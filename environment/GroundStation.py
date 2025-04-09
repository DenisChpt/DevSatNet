import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import math

from environment.EarthModel import EarthModel
from environment.utils.CoordinateTransforms import CoordinateTransforms

class GroundStation:
	"""
	Classe représentant une station terrestre pour la communication avec les satellites.
	"""
	
	def __init__(
		self,
		stationId: int,
		name: str,
		latitude: float,
		longitude: float,
		altitude: float = 0.0,
		bandwidth: float = 500.0,  # Mbps
		storageCapacity: float = 10000.0,  # GB
		earthModel: Optional[EarthModel] = None,
		minimumElevation: float = 10.0,  # degrés
		transmitPower: float = 1000.0,  # W
		antennaGain: float = 40.0,  # dBi
		systemTemperature: float = 290.0  # K
	):
		"""
		Initialise une station terrestre.
		
		Args:
			stationId: Identifiant unique de la station
			name: Nom de la station
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			altitude: Altitude en mètres au-dessus du niveau de la mer
			bandwidth: Bande passante maximale en Mbps
			storageCapacity: Capacité de stockage en GB
			earthModel: Modèle de la Terre
			minimumElevation: Angle d'élévation minimum pour la communication en degrés
			transmitPower: Puissance d'émission en Watts
			antennaGain: Gain d'antenne en dBi
			systemTemperature: Température du système en Kelvin
		"""
		self.stationId = stationId
		self.name = name
		self.latitude = latitude
		self.longitude = longitude
		self.altitude = altitude
		self.bandwidth = bandwidth
		self.storageCapacity = storageCapacity
		self.earthModel = earthModel
		self.minimumElevation = minimumElevation
		self.transmitPower = transmitPower
		self.antennaGain = antennaGain
		self.systemTemperature = systemTemperature
		
		# Convertir la position géodésique en coordonnées cartésiennes
		self.position = self._calculateCartesianPosition()
		
		# État opérationnel
		self.availableBandwidth = bandwidth
		self.usedStorage = 0.0
		self.activeConnections: List[Tuple[int, float]] = []  # (satellite_id, allocated_bandwidth)
		self.dataQueue: List[Dict[str, Any]] = []
		self.weather = "clear"  # État météorologique (clear, cloudy, rainy, stormy)
		self.weatherAttenuation = 0.0  # Atténuation due à la météo en dB
		
		# Variables pour les statistiques
		self.dataReceived = 0.0  # GB
		self.dataTransmitted = 0.0  # GB
		self.connectionTime = 0.0  # secondes
	
	def reset(self) -> None:
		"""
		Réinitialise l'état de la station terrestre.
		"""
		self.availableBandwidth = self.bandwidth
		self.usedStorage = 0.0
		self.activeConnections = []
		self.dataQueue = []
		self.weather = "clear"
		self.weatherAttenuation = 0.0
		self.dataReceived = 0.0
		self.dataTransmitted = 0.0
		self.connectionTime = 0.0
	
	def update(self, deltaTime: float, satellites: List[Any]) -> None:
		"""
		Met à jour l'état de la station terrestre.
		
		Args:
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
			satellites: Liste des satellites de la constellation
		"""
		# Mettre à jour la position cartésienne si le modèle de la Terre a été mis à jour
		if self.earthModel is not None:
			self.position = self._calculateCartesianPosition()
		
		# Mettre à jour les conditions météorologiques (simulation simplifiée)
		self._updateWeather(deltaTime)
		
		# Traiter les données dans la file d'attente
		self._processDataQueue(deltaTime)
		
		# Mettre à jour les connexions actives
		self._updateConnections(satellites, deltaTime)
	
	def _calculateCartesianPosition(self) -> np.ndarray:
		"""
		Calcule les coordonnées cartésiennes de la station à partir de sa position géodésique.
		
		Returns:
			Position en coordonnées cartésiennes [x, y, z] en km dans le référentiel ECI
		"""
		if self.earthModel is not None:
			# Utiliser le modèle de la Terre pour la conversion
			return self.earthModel.geodeticToCartesian(self.latitude, self.longitude, self.altitude / 1000.0)
		else:
			# Conversion simplifiée sans modèle de la Terre
			return CoordinateTransforms.geodeticToEcef(self.latitude, self.longitude, self.altitude / 1000.0)
	
	def getPosition(self) -> np.ndarray:
		"""
		Renvoie la position de la station en coordonnées cartésiennes.
		
		Returns:
			Position [x, y, z] en km dans le référentiel ECI
		"""
		return self.position
	
	def isSatelliteVisible(self, satellitePosition: np.ndarray) -> bool:
		"""
		Détermine si un satellite est visible depuis la station terrestre.
		
		Args:
			satellitePosition: Position du satellite [x, y, z] en km dans le référentiel ECI
			
		Returns:
			True si le satellite est visible, False sinon
		"""
		# Vecteur de la station au satellite
		stationToSatellite = satellitePosition - self.position
		distance = np.linalg.norm(stationToSatellite)
		
		# Vecteur normalisé de la station au satellite
		direction = stationToSatellite / distance
		
		# Vecteur normalisé de la position de la station (direction vers le zénith local)
		zenith = self.position / np.linalg.norm(self.position)
		
		# Calculer l'angle d'élévation
		dotProduct = np.dot(zenith, direction)
		elevationAngle = np.arcsin(np.clip(dotProduct, -1.0, 1.0))
		elevationDegrees = np.degrees(elevationAngle)
		
		# Vérifier si l'angle d'élévation est supérieur au minimum requis
		if elevationDegrees < self.minimumElevation:
			return False
		
		# Vérifier s'il y a une ligne de vue directe (pas bloquée par la Terre)
		# Simplification: vérifier si la ligne entre la station et le satellite n'intersecte pas la Terre
		earthRadius = 6371.0  # km
		
		# Produit scalaire du vecteur station avec le vecteur vers le satellite
		stationMagnitude = np.linalg.norm(self.position)
		cosAngle = np.dot(self.position, stationToSatellite) / (stationMagnitude * distance)
		
		# Distance au carré du centre de la Terre à la ligne station-satellite
		distanceSquared = stationMagnitude**2 * (1 - cosAngle**2)
		
		# Si cette distance est inférieure au rayon de la Terre au carré, la ligne intersecte la Terre
		return distanceSquared >= earthRadius**2
	
	def calculateLinkQuality(self, satellitePosition: np.ndarray) -> float:
		"""
		Calcule la qualité de la liaison avec un satellite.
		
		Args:
			satellitePosition: Position du satellite [x, y, z] en km dans le référentiel ECI
			
		Returns:
			Qualité de la liaison entre 0.0 et 1.0
		"""
		if not self.isSatelliteVisible(satellitePosition):
			return 0.0
		
		# Calculer la distance
		distance = np.linalg.norm(satellitePosition - self.position)
		
		# Calculer l'atténuation en espace libre
		frequency = 12.0e9  # Fréquence en Hz (par exemple, bande Ku à 12 GHz)
		lightSpeed = 299792458.0  # m/s
		wavelength = lightSpeed / frequency  # m
		
		# Formule de pertes en espace libre en dB
		freeSpaceLoss = 20 * np.log10(4 * np.pi * distance * 1000 / wavelength)
		
		# Ajouter l'atténuation due à la météo
		totalAttenuation = freeSpaceLoss + self.weatherAttenuation
		
		# Calculer le rapport signal/bruit (SNR)
		receiverGain = self.antennaGain  # dBi
		transmitPowerDB = 10 * np.log10(self.transmitPower)  # dBW
		
		# Constante de Boltzmann en dBW/K/Hz
		boltzmannConstantDB = -228.6
		
		# Largeur de bande (par exemple, 50 MHz)
		bandwidthHz = 50e6
		bandwidthDB = 10 * np.log10(bandwidthHz)
		
		# Bruit thermique (N = k * T * B)
		noiseDB = boltzmannConstantDB + 10 * np.log10(self.systemTemperature) + bandwidthDB
		
		# SNR en dB
		snrDB = transmitPowerDB + receiverGain - totalAttenuation - noiseDB
		
		# Convertir en qualité de liaison entre 0 et 1
		minSNR = 0  # 0 dB (rapport signal/bruit = 1)
		maxSNR = 30  # 30 dB (rapport signal/bruit = 1000)
		
		# Limiter SNR entre minSNR et maxSNR et normaliser entre 0 et 1
		normalizedSNR = np.clip((snrDB - minSNR) / (maxSNR - minSNR), 0.0, 1.0)
		
		return normalizedSNR
	
	def allocateBandwidth(self, satelliteId: int, requestedBandwidth: float) -> float:
		"""
		Alloue de la bande passante pour une connexion avec un satellite.
		
		Args:
			satelliteId: Identifiant du satellite
			requestedBandwidth: Bande passante demandée en Mbps
			
		Returns:
			Bande passante allouée en Mbps
		"""
		# Vérifier si un lien existe déjà avec ce satellite
		for i, (satId, bandwidth) in enumerate(self.activeConnections):
			if satId == satelliteId:
				# Mettre à jour la bande passante allouée
				allocatedBandwidth = min(requestedBandwidth, self.availableBandwidth + bandwidth)
				
				# Ajuster la bande passante disponible
				self.availableBandwidth += bandwidth - allocatedBandwidth
				
				# Mettre à jour la connexion
				self.activeConnections[i] = (satelliteId, allocatedBandwidth)
				
				return allocatedBandwidth
		
		# Nouvelle connexion
		allocatedBandwidth = min(requestedBandwidth, self.availableBandwidth)
		
		if allocatedBandwidth > 0:
			# Ajouter la connexion
			self.activeConnections.append((satelliteId, allocatedBandwidth))
			
			# Mettre à jour la bande passante disponible
			self.availableBandwidth -= allocatedBandwidth
		
		return allocatedBandwidth
	
	def receiveData(self, dataSize: float, satelliteId: int) -> float:
		"""
		Reçoit des données d'un satellite.
		
		Args:
			dataSize: Taille des données en GB
			satelliteId: Identifiant du satellite
			
		Returns:
			Quantité de données effectivement reçue en GB
		"""
		# Vérifier l'espace de stockage disponible
		availableStorage = self.storageCapacity - self.usedStorage
		receivedData = min(dataSize, availableStorage)
		
		if receivedData > 0:
			# Mettre à jour le stockage utilisé
			self.usedStorage += receivedData
			self.dataReceived += receivedData
			
			# Ajouter à la file d'attente de traitement
			self.dataQueue.append({
				"size": receivedData,
				"source": satelliteId,
				"progress": 0.0,
				"priority": 1.0
			})
		
		return receivedData
	
	def transmitData(self, dataSize: float, destination: str) -> float:
		"""
		Transmet des données vers une destination externe.
		
		Args:
			dataSize: Taille des données en GB
			destination: Destination des données
			
		Returns:
			Quantité de données effectivement transmise en GB
		"""
		# Vérifier la disponibilité des données
		transmittedData = min(dataSize, self.usedStorage)
		
		if transmittedData > 0:
			# Mettre à jour le stockage utilisé
			self.usedStorage -= transmittedData
			self.dataTransmitted += transmittedData
		
		return transmittedData
	
	def getState(self) -> Dict[str, Any]:
		"""
		Retourne l'état actuel de la station pour sauvegarde/chargement.
		
		Returns:
			Dictionnaire de l'état
		"""
		return {
			"station_id": self.stationId,
			"name": self.name,
			"latitude": self.latitude,
			"longitude": self.longitude,
			"altitude": self.altitude,
			"available_bandwidth": self.availableBandwidth,
			"used_storage": self.usedStorage,
			"active_connections": self.activeConnections,
			"weather": self.weather,
			"weather_attenuation": self.weatherAttenuation,
			"data_received": self.dataReceived,
			"data_transmitted": self.dataTransmitted,
			"connection_time": self.connectionTime
		}
	
	def setState(self, state: Dict[str, Any]) -> None:
		"""
		Restaure l'état de la station.
		
		Args:
			state: État à restaurer
		"""
		self.stationId = state["station_id"]
		self.name = state["name"]
		self.latitude = state["latitude"]
		self.longitude = state["longitude"]
		self.altitude = state["altitude"]
		self.availableBandwidth = state["available_bandwidth"]
		self.usedStorage = state["used_storage"]
		self.activeConnections = state["active_connections"]
		self.weather = state["weather"]
		self.weatherAttenuation = state["weather_attenuation"]
		self.dataReceived = state["data_received"]
		self.dataTransmitted = state["data_transmitted"]
		self.connectionTime = state["connection_time"]
		
		# Recalculer la position
		self.position = self._calculateCartesianPosition()
	
	def _updateWeather(self, deltaTime: float) -> None:
		"""
		Met à jour les conditions météorologiques de la station.
		
		Args:
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
		"""
		# Modèle météorologique simplifié avec transitions aléatoires
		if deltaTime <= 0:
			return
		
		# Probabilités de transition par heure
		# De clear -> cloudy: 0.1
		# De cloudy -> rainy: 0.2
		# De rainy -> stormy: 0.1
		# De stormy -> rainy: 0.3
		# De rainy -> cloudy: 0.3
		# De cloudy -> clear: 0.2
		
		# Convertir les probabilités par heure en probabilités pour deltaTime
		hourFraction = deltaTime / 3600.0
		transitionProbability = {
			"clear": {"cloudy": 0.1 * hourFraction},
			"cloudy": {"clear": 0.2 * hourFraction, "rainy": 0.2 * hourFraction},
			"rainy": {"cloudy": 0.3 * hourFraction, "stormy": 0.1 * hourFraction},
			"stormy": {"rainy": 0.3 * hourFraction}
		}
		
		# Générer un nombre aléatoire
		randomValue = np.random.random()
		
		# Vérifier les transitions possibles depuis l'état actuel
		if self.weather in transitionProbability:
			cumulativeProbability = 0.0
			
			for nextState, probability in transitionProbability[self.weather].items():
				cumulativeProbability += probability
				
				if randomValue < cumulativeProbability:
					self.weather = nextState
					break
		
		# Mettre à jour l'atténuation en fonction de la météo
		if self.weather == "clear":
			self.weatherAttenuation = 0.0
		elif self.weather == "cloudy":
			self.weatherAttenuation = 0.5
		elif self.weather == "rainy":
			self.weatherAttenuation = 3.0
		elif self.weather == "stormy":
			self.weatherAttenuation = 10.0
	
	def _processDataQueue(self, deltaTime: float) -> None:
		"""
		Traite les données dans la file d'attente.
		
		Args:
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
		"""
		# Simplification: traiter automatiquement les données et les envoyer vers la destination
		processRate = 2.0  # GB par seconde
		totalProcessed = 0.0
		
		# Trier la file d'attente par priorité
		self.dataQueue.sort(key=lambda x: x["priority"], reverse=True)
		
		# Traiter les données
		remainingQueue = []
		
		for dataItem in self.dataQueue:
			if totalProcessed < processRate * deltaTime:
				# Capacité de traitement restante
				remainingCapacity = processRate * deltaTime - totalProcessed
				
				# Taille des données restantes à traiter
				remainingSize = dataItem["size"] * (1.0 - dataItem["progress"])
				
				if remainingSize <= remainingCapacity:
					# Les données peuvent être complètement traitées
					totalProcessed += remainingSize
					
					# Transmettre les données
					self.transmitData(dataItem["size"], "external")
				else:
					# Traitement partiel
					processedFraction = remainingCapacity / dataItem["size"]
					totalProcessed += remainingCapacity
					
					# Mettre à jour la progression
					dataItem["progress"] += processedFraction
					remainingQueue.append(dataItem)
			else:
				# Plus de capacité de traitement disponible
				remainingQueue.append(dataItem)
		
		# Mettre à jour la file d'attente
		self.dataQueue = remainingQueue
	
	def _updateConnections(self, satellites: List[Any], deltaTime: float) -> None:
		"""
		Met à jour les connexions actives avec les satellites.
		
		Args:
			satellites: Liste des satellites
			deltaTime: Temps écoulé depuis la dernière mise à jour en secondes
		"""
		# Mettre à jour le temps de connexion
		if self.activeConnections:
			self.connectionTime += deltaTime
		
		# Vérifier les connexions actives
		updatedConnections = []
		
		for satelliteId, bandwidth in self.activeConnections:
			# Trouver le satellite
			satellite = None
			for sat in satellites:
				if sat.satelliteId == satelliteId:
					satellite = sat
					break
			
			if satellite is not None and self.isSatelliteVisible(satellite.getPosition()):
				# Connexion toujours valide
				updatedConnections.append((satelliteId, bandwidth))
			else:
				# Connexion perdue, libérer la bande passante
				self.availableBandwidth += bandwidth
		
		# Mettre à jour les connexions
		self.activeConnections = updatedConnections