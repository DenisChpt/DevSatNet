import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from environment.EarthModel import EarthModel

class Atmosphere:
	"""
	Modèle de l'atmosphère terrestre pour la simulation de la traînée atmosphérique
	et d'autres effets sur les satellites en orbite basse.
	"""
	
	def __init__(
		self,
		atmosphericModel: str = 'NRLMSISE',
		f107Flux: float = 150.0,
		ap: float = 4.0,
		dayNightRatio: float = 1.2
	):
		"""
		Initialise le modèle atmosphérique.
		
		Args:
			atmosphericModel: Modèle atmosphérique ('NRLMSISE', 'JB2008', 'SIMPLE')
			f107Flux: Flux solaire F10.7 (indice d'activité solaire) en SFU
			ap: Indice géomagnétique Ap
			dayNightRatio: Ratio de densité jour/nuit
		"""
		self.atmosphericModel = atmosphericModel
		self.f107Flux = f107Flux
		self.ap = ap
		self.dayNightRatio = dayNightRatio
		
		# Variables de temps
		self.currentTime = datetime.now()
		self.simulationTime = 0.0
		
		# Variations atmosphériques
		self.densityVariation = 0.0  # Variation de densité due à l'activité solaire
		self.temperatureVariation = 0.0  # Variation de température
		
		# Paramètres du modèle exponentiel simplifié
		self.scaleHeight = {}  # km, dépend de l'altitude
		self.baseDensity = {}  # kg/m³, dépend de l'altitude
		self._initializeExponentialModel()
	
	def reset(self) -> None:
		"""
		Réinitialise le modèle atmosphérique.
		"""
		self.currentTime = datetime.now()
		self.simulationTime = 0.0
		self.densityVariation = 0.0
		self.temperatureVariation = 0.0
	
	def update(self, simulationTime: float, earthModel: Optional[EarthModel] = None) -> None:
		"""
		Met à jour le modèle atmosphérique au temps de simulation spécifié.
		
		Args:
			simulationTime: Temps de simulation en secondes
			earthModel: Modèle de la Terre pour les calculs dépendant de la position du Soleil
		"""
		self.simulationTime = simulationTime
		
		if earthModel is not None:
			self.currentTime = earthModel.currentTime
			
			# Mise à jour des variations basées sur l'activité solaire
			# Simplification: oscillation sinusoïdale sur une période de 27 jours (rotation solaire)
			solar_cycle_phase = (simulationTime / (27 * 24 * 3600)) % 1.0
			self.densityVariation = 0.2 * np.sin(2 * np.pi * solar_cycle_phase)
			
			# Variation jour/nuit pour la température (simplifiée)
			hour_of_day = self.currentTime.hour + self.currentTime.minute / 60
			day_phase = (hour_of_day / 24) % 1.0
			self.temperatureVariation = 50 * np.sin(2 * np.pi * (day_phase - 0.25))  # Maximum à 6h
	
	def getDensity(self, altitude: float, latitude: float = 0.0, longitude: float = 0.0) -> float:
		"""
		Calcule la densité atmosphérique à une altitude et position données.
		
		Args:
			altitude: Altitude au-dessus de la surface terrestre en km
			latitude: Latitude en degrés (optionnel, utilisé par certains modèles)
			longitude: Longitude en degrés (optionnel, utilisé par certains modèles)
			
		Returns:
			Densité atmosphérique en kg/m³
		"""
		if self.atmosphericModel == 'SIMPLE':
			return self._getSimpleDensity(altitude, latitude, longitude)
		elif self.atmosphericModel == 'NRLMSISE':
			return self._getNRLMSISEDensity(altitude, latitude, longitude)
		elif self.atmosphericModel == 'JB2008':
			return self._getJB2008Density(altitude, latitude, longitude)
		else:
			# Modèle par défaut
			return self._getSimpleDensity(altitude, latitude, longitude)
	
	def getTemperature(self, altitude: float, latitude: float = 0.0, longitude: float = 0.0) -> float:
		"""
		Calcule la température atmosphérique à une altitude et position données.
		
		Args:
			altitude: Altitude au-dessus de la surface terrestre en km
			latitude: Latitude en degrés (optionnel)
			longitude: Longitude en degrés (optionnel)
			
		Returns:
			Température en Kelvin
		"""
		# Modèle de température simplifié
		if altitude < 0:
			return 288.15  # Température standard au niveau de la mer (K)
			
		if altitude < 11:
			# Troposphère (-6.5 K/km)
			return 288.15 - 6.5 * altitude
		elif altitude < 20:
			# Stratosphère inférieure (isotherme)
			return 216.65
		elif altitude < 32:
			# Stratosphère supérieure (+1 K/km)
			return 216.65 + (altitude - 20)
		elif altitude < 47:
			# Mésosphère inférieure (+2.8 K/km)
			return 228.65 + 2.8 * (altitude - 32)
		elif altitude < 51:
			# Stratopause (isotherme)
			return 270.65
		elif altitude < 71:
			# Mésosphère (-2.8 K/km)
			return 270.65 - 2.8 * (altitude - 51)
		elif altitude < 84.852:
			# Mésopause (isotherme)
			return 214.65
		else:
			# Thermosphère (augmentation rapide avec l'altitude)
			base_temp = 214.65
			height_above_mesopause = altitude - 84.852
			
			# Simplification pour la thermosphère
			thermosphere_gain = 12.0  # K/km
			temp = base_temp + thermosphere_gain * height_above_mesopause
			
			# Ajouter la variation due à l'activité solaire
			temp += self.f107Flux / 150.0 * 500.0  # Ajustement basé sur F10.7
			
			# Ajouter la variation jour/nuit
			temp += self.temperatureVariation
			
			return temp
	
	def getWindVelocity(self, altitude: float, latitude: float = 0.0, longitude: float = 0.0) -> np.ndarray:
		"""
		Calcule la vitesse du vent à une altitude et position données.
		
		Args:
			altitude: Altitude au-dessus de la surface terrestre en km
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			
		Returns:
			Vecteur vitesse du vent [vx, vy, vz] en m/s dans le référentiel ECEF
		"""
		# Modèle de vent simplifié pour la thermosphère
		if altitude < 100:
			# En dessous de la thermosphère, vents faibles dans le modèle simplifié
			wind_speed = 10.0  # m/s
		else:
			# Dans la thermosphère, vents plus forts
			wind_speed = 50.0 + (altitude - 100) * 2  # m/s, augmente avec l'altitude
		
		# Direction du vent influencée par la rotation terrestre (simplification)
		lat_rad = np.radians(latitude)
		lon_rad = np.radians(longitude)
		
		# Composantes du vent
		vx = -wind_speed * np.sin(lon_rad)
		vy = wind_speed * np.cos(lon_rad)
		vz = wind_speed * 0.2 * np.sin(2 * lat_rad)  # Composante verticale faible
		
		return np.array([vx, vy, vz])
	
	def calculateDragForce(
		self,
		velocity: np.ndarray,
		crossSectionalArea: float,
		dragCoefficient: float,
		altitude: float,
		latitude: float = 0.0,
		longitude: float = 0.0
	) -> np.ndarray:
		"""
		Calcule la force de traînée sur un satellite.
		
		Args:
			velocity: Vecteur vitesse relative à l'atmosphère [vx, vy, vz] en m/s
			crossSectionalArea: Surface exposée en m²
			dragCoefficient: Coefficient de traînée (sans unité)
			altitude: Altitude en km
			latitude: Latitude en degrés (optionnel)
			longitude: Longitude en degrés (optionnel)
			
		Returns:
			Vecteur force de traînée [Fx, Fy, Fz] en N
		"""
		# Obtenir la densité atmosphérique
		density = self.getDensity(altitude, latitude, longitude)
		
		# Magnitude de la vitesse
		velocity_magnitude = np.linalg.norm(velocity)
		
		if velocity_magnitude < 1e-6:
			return np.zeros(3)
		
		# Direction unitaire de la vitesse
		velocity_direction = velocity / velocity_magnitude
		
		# Calcul de la force de traînée: F = 0.5 * rho * v^2 * Cd * A
		drag_magnitude = 0.5 * density * velocity_magnitude**2 * dragCoefficient * crossSectionalArea
		
		# Vecteur force (opposé à la direction de la vitesse)
		drag_force = -drag_magnitude * velocity_direction
		
		return drag_force
	
	def getState(self) -> Dict[str, Any]:
		"""
		Retourne l'état actuel du modèle pour sauvegarde/chargement.
		
		Returns:
			Dictionnaire de l'état
		"""
		return {
			"simulation_time": self.simulationTime,
			"current_time": self.currentTime.isoformat(),
			"f107_flux": self.f107Flux,
			"ap": self.ap,
			"density_variation": self.densityVariation,
			"temperature_variation": self.temperatureVariation
		}
	
	def setState(self, state: Dict[str, Any]) -> None:
		"""
		Restaure l'état du modèle.
		
		Args:
			state: État à restaurer
		"""
		self.simulationTime = state["simulation_time"]
		self.currentTime = datetime.fromisoformat(state["current_time"])
		self.f107Flux = state["f107_flux"]
		self.ap = state["ap"]
		self.densityVariation = state["density_variation"]
		self.temperatureVariation = state["temperature_variation"]
	
	def _initializeExponentialModel(self) -> None:
		"""
		Initialise les paramètres du modèle exponentiel simplifié.
		"""
		# Définir les altitudes de référence et les densités correspondantes
		self.baseDensity = {
			0: 1.225,       # Niveau de la mer
			25: 3.899e-2,   # 25 km
			50: 1.027e-3,   # 50 km
			75: 3.170e-5,   # 75 km
			100: 5.604e-7,  # 100 km
			150: 2.076e-9,  # 150 km
			200: 2.541e-10, # 200 km
			300: 1.916e-11, # 300 km
			400: 3.158e-12, # 400 km
			500: 6.967e-13, # 500 km
			600: 1.454e-13, # 600 km
			700: 3.614e-14, # 700 km
			800: 1.170e-14, # 800 km
			900: 5.245e-15, # 900 km
			1000: 3.019e-15 # 1000 km
		}
		
		# Définir les hauteurs d'échelle pour différentes plages d'altitude
		self.scaleHeight = {
			0: 7.249,      # 0-25 km
			25: 6.349,     # 25-50 km
			50: 6.682,     # 50-75 km
			75: 5.927,     # 75-100 km
			100: 8.713,    # 100-150 km
			150: 7.554,    # 150-200 km
			200: 6.304,    # 200-300 km
			300: 6.430,    # 300-400 km
			400: 7.355,    # 400-500 km
			500: 9.208,    # 500-600 km
			600: 12.366,   # 600-700 km
			700: 16.204,   # 700-800 km
			800: 19.984,   # 800-900 km
			900: 24.528    # 900-1000 km
		}
	
	def _getSimpleDensity(self, altitude: float, latitude: float = 0.0, longitude: float = 0.0) -> float:
		"""
		Implémentation du modèle de densité exponentiel simplifié.
		
		Args:
			altitude: Altitude en km
			latitude: Latitude en degrés (utilisé pour les variations)
			longitude: Longitude en degrés (utilisé pour les variations)
			
		Returns:
			Densité atmosphérique en kg/m³
		"""
		if altitude < 0:
			return self.baseDensity[0]
		elif altitude > 1000:
			# Extrapolation pour les très hautes altitudes
			base_altitude = 1000
			base_density = self.baseDensity[base_altitude]
			scale_height = self.scaleHeight[900]  # Utiliser la hauteur d'échelle la plus élevée
			return base_density * np.exp(-(altitude - base_altitude) / scale_height)
		else:
			# Trouver les altitudes de référence encadrant l'altitude donnée
			base_altitudes = list(self.baseDensity.keys())
			base_altitudes.sort()
			
			# Trouver l'altitude de base inférieure
			lower_base = base_altitudes[0]
			for base in base_altitudes:
				if base <= altitude:
					lower_base = base
				else:
					break
			
			# Utiliser le modèle exponentiel
			base_density = self.baseDensity[lower_base]
			scale_height = self.scaleHeight[lower_base]
			density = base_density * np.exp(-(altitude - lower_base) / scale_height)
			
			# Appliquer les variations
			# Variation jour/nuit (simplifiée)
			hour = self.currentTime.hour
			is_day = 6 <= hour <= 18  # Simplification grossière jour/nuit
			day_night_factor = 1.0 + (self.dayNightRatio - 1.0) * float(is_day)
			
			# Variation de latitude (densité plus élevée à l'équateur)
			lat_factor = 1.0 - 0.2 * np.abs(latitude) / 90.0
			
			# Variation due à l'activité solaire
			solar_factor = 1.0 + self.densityVariation
			
			# Appliquer tous les facteurs
			density *= day_night_factor * lat_factor * solar_factor
			
			return density
	
	def _getNRLMSISEDensity(self, altitude: float, latitude: float = 0.0, longitude: float = 0.0) -> float:
		"""
		Implémentation simplifiée du modèle NRLMSISE-00.
		Dans une implémentation réelle, cela appellerait une bibliothèque externe.
		
		Args:
			altitude: Altitude en km
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			
		Returns:
			Densité atmosphérique en kg/m³
		"""
		# Pour la simulation, nous utilisons le modèle simple modifié par des facteurs de correction
		base_density = self._getSimpleDensity(altitude, latitude, longitude)
		
		# Facteurs de correction pour approximer NRLMSISE-00
		# Ces facteurs sont des approximations grossières basées sur des tendances générales
		
		# Correction pour l'activité solaire (F10.7)
		solar_correction = 1.0 + 0.3 * (self.f107Flux - 150) / 150
		
		# Correction pour l'activité géomagnétique (Ap)
		geo_correction = 1.0 + 0.2 * (self.ap - 4) / 10
		
		# Correction saisonnière (simplifiée)
		month = self.currentTime.month
		seasonal_phase = (month - 1) / 12 * 2 * np.pi
		seasonal_correction = 1.0 + 0.1 * np.sin(seasonal_phase)
		
		# Correction de latitude (plus dense aux pôles pendant l'hiver hémisphérique)
		lat_rad = np.radians(latitude)
		winter_hemisphere = (month >= 10 or month <= 3) and latitude < 0 or (month >= 4 and month <= 9) and latitude > 0
		if winter_hemisphere:
			lat_correction = 1.0 + 0.15 * np.abs(np.sin(lat_rad))
		else:
			lat_correction = 1.0 - 0.05 * np.abs(np.sin(lat_rad))
		
		# Appliquer toutes les corrections
		return base_density * solar_correction * geo_correction * seasonal_correction * lat_correction
	
	def _getJB2008Density(self, altitude: float, latitude: float = 0.0, longitude: float = 0.0) -> float:
		"""
		Implémentation simplifiée du modèle Jacchia-Bowman 2008.
		Dans une implémentation réelle, cela appellerait une bibliothèque externe.
		
		Args:
			altitude: Altitude en km
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			
		Returns:
			Densité atmosphérique en kg/m³
		"""
		# Pour la simulation, approximation basée sur le modèle simple avec des corrections
		base_density = self._getSimpleDensity(altitude, latitude, longitude)
		
		# JB2008 est plus sensible aux variations solaires à haute altitude
		if altitude > 500:
			solar_factor = 1.0 + 0.5 * (self.f107Flux - 150) / 150
		else:
			solar_factor = 1.0 + 0.3 * (self.f107Flux - 150) / 150
		
		# Variation diurne plus prononcée
		hour = self.currentTime.hour + self.currentTime.minute / 60
		day_phase = (hour / 24) % 1.0
		diurnal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * (day_phase - 0.25))
		
		# Effet plus prononcé de l'activité géomagnétique
		geo_factor = 1.0 + 0.3 * (self.ap - 4) / 10
		
		# Effets saisonniers et latitudinaux
		month = self.currentTime.month
		seasonal_phase = (month - 1) / 12 * 2 * np.pi
		
		# Terme saisonnier
		seasonal_term = 0.15 * np.sin(seasonal_phase)
		
		# Terme latitudinal
		lat_rad = np.radians(latitude)
		lat_term = 0.1 * np.sin(lat_rad) * np.sin(seasonal_phase)
		
		# Facteur combiné
		seasonal_lat_factor = 1.0 + seasonal_term + lat_term
		
		# Appliquer tous les facteurs
		return base_density * solar_factor * diurnal_factor * geo_factor * seasonal_lat_factor