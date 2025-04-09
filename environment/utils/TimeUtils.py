import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Optional

class TimeUtils:
	"""
	Classe utilitaire pour la gestion du temps et des conversions entre
	différents systèmes de temps utilisés dans la simulation de satellites.
	"""
	
	# Constantes
	J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)  # Époque J2000
	SECONDS_PER_DAY = 86400.0
	EARTH_ROTATION_RATE = 7.2921150e-5  # rad/s
	
	@staticmethod
	def utcToJulianDate(utcTime: datetime) -> float:
		"""
		Convertit une date UTC en date julienne.
		
		Args:
			utcTime: Date et heure UTC
			
		Returns:
			Date julienne correspondante
		"""
		# Formule pour calculer la date julienne
		year, month, day = utcTime.year, utcTime.month, utcTime.day
		
		# Ajustement pour les mois de janvier et février
		if month <= 2:
			year -= 1
			month += 12
		
		A = int(year / 100)
		B = 2 - A + int(A / 4)
		
		# Calcul de la partie entière
		jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
		
		# Ajouter la partie fractionnaire pour l'heure
		hour_fraction = (utcTime.hour + utcTime.minute / 60.0 + utcTime.second / 3600.0) / 24.0
		
		return jd + hour_fraction
	
	@staticmethod
	def julianDateToUtc(jd: float) -> datetime:
		"""
		Convertit une date julienne en date UTC.
		
		Args:
			jd: Date julienne
			
		Returns:
			Date et heure UTC correspondante
		"""
		# Formule pour convertir la date julienne en date
		jd_plus_05 = jd + 0.5
		Z = int(jd_plus_05)
		F = jd_plus_05 - Z
		
		if Z < 2299161:
			A = Z
		else:
			alpha = int((Z - 1867216.25) / 36524.25)
			A = Z + 1 + alpha - int(alpha / 4)
		
		B = A + 1524
		C = int((B - 122.1) / 365.25)
		D = int(365.25 * C)
		E = int((B - D) / 30.6001)
		
		# Jour du mois avec partie fractionnaire
		day_frac = B - D - int(30.6001 * E) + F
		day = int(day_frac)
		
		# Heure du jour (partie fractionnaire du jour)
		hour_frac = (day_frac - day) * 24.0
		hour = int(hour_frac)
		
		# Minute (partie fractionnaire de l'heure)
		minute_frac = (hour_frac - hour) * 60.0
		minute = int(minute_frac)
		
		# Seconde (partie fractionnaire de la minute)
		second_frac = (minute_frac - minute) * 60.0
		second = int(second_frac)
		microsecond = int((second_frac - second) * 1e6)
		
		# Déterminer le mois
		if E < 14:
			month = E - 1
		else:
			month = E - 13
		
		# Déterminer l'année
		if month > 2:
			year = C - 4716
		else:
			year = C - 4715
		
		return datetime(year, month, day, hour, minute, second, microsecond)
	
	@staticmethod
	def utcToJ2000SecondsOffset(utcTime: datetime) -> float:
		"""
		Calcule le nombre de secondes écoulées depuis l'époque J2000.
		
		Args:
			utcTime: Date et heure UTC
			
		Returns:
			Nombre de secondes depuis J2000
		"""
		delta = utcTime - TimeUtils.J2000_EPOCH
		return delta.total_seconds()
	
	@staticmethod
	def j2000SecondsOffsetToUtc(seconds: float) -> datetime:
		"""
		Convertit un nombre de secondes depuis J2000 en date UTC.
		
		Args:
			seconds: Nombre de secondes depuis J2000
			
		Returns:
			Date et heure UTC correspondante
		"""
		return TimeUtils.J2000_EPOCH + timedelta(seconds=seconds)
	
	@staticmethod
	def calculateGst(utcTime: datetime) -> float:
		"""
		Calcule le temps sidéral de Greenwich (GST) à partir d'une date UTC.
		
		Args:
			utcTime: Date et heure UTC
			
		Returns:
			Temps sidéral de Greenwich en radians
		"""
		# Calculer la date julienne
		jd = TimeUtils.utcToJulianDate(utcTime)
		
		# Calculer le nombre de siècles juliens depuis J2000
		T = (jd - 2451545.0) / 36525.0
		
		# Calcul du GST en degrés
		theta_g = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2 - T**3 / 38710000.0
		
		# Normaliser à [0, 360] degrés
		theta_g = theta_g % 360.0
		
		# Convertir en radians
		return np.radians(theta_g)
	
	@staticmethod
	def calculateLst(utcTime: datetime, longitude: float) -> float:
		"""
		Calcule le temps sidéral local (LST) à partir d'une date UTC et d'une longitude.
		
		Args:
			utcTime: Date et heure UTC
			longitude: Longitude en degrés
			
		Returns:
			Temps sidéral local en radians
		"""
		# Calculer le temps sidéral de Greenwich
		gst = TimeUtils.calculateGst(utcTime)
		
		# Ajouter la longitude (convertie en radians)
		lst = gst + np.radians(longitude)
		
		# Normaliser à [0, 2π]
		return lst % (2 * np.pi)
	
	@staticmethod
	def propagateOrbitTime(initialTime: float, deltaTime: float) -> float:
		"""
		Propage le temps de l'orbite d'un certain incrément.
		
		Args:
			initialTime: Temps initial en secondes depuis J2000
			deltaTime: Incrément de temps en secondes
			
		Returns:
			Nouveau temps en secondes depuis J2000
		"""
		return initialTime + deltaTime
	
	@staticmethod
	def simulationTimeToUtc(simulationTime: float, startTime: Optional[datetime] = None) -> datetime:
		"""
		Convertit un temps de simulation en date UTC.
		
		Args:
			simulationTime: Temps de simulation en secondes
			startTime: Date et heure UTC de début de la simulation (défaut: J2000)
			
		Returns:
			Date et heure UTC correspondante
		"""
		if startTime is None:
			startTime = TimeUtils.J2000_EPOCH
		
		return startTime + timedelta(seconds=simulationTime)
	
	@staticmethod
	def utcToSimulationTime(utcTime: datetime, startTime: Optional[datetime] = None) -> float:
		"""
		Convertit une date UTC en temps de simulation.
		
		Args:
			utcTime: Date et heure UTC
			startTime: Date et heure UTC de début de la simulation (défaut: J2000)
			
		Returns:
			Temps de simulation en secondes
		"""
		if startTime is None:
			startTime = TimeUtils.J2000_EPOCH
		
		delta = utcTime - startTime
		return delta.total_seconds()
	
	@staticmethod
	def calculateSunPosition(utcTime: datetime) -> np.ndarray:
		"""
		Calcule la position du Soleil dans le référentiel ECI à un instant donné.
		Modèle simplifié sans perturbations.
		
		Args:
			utcTime: Date et heure UTC
			
		Returns:
			Vecteur position du Soleil [x, y, z] en unités astronomiques
		"""
		# Calculer la date julienne
		jd = TimeUtils.utcToJulianDate(utcTime)
		
		# Calculer le nombre de siècles juliens depuis J2000
		T = (jd - 2451545.0) / 36525.0
		
		# Longitude moyenne du Soleil (en degrés)
		L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
		
		# Anomalie moyenne du Soleil (en degrés)
		M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
		
		# Convertir en radians
		M_rad = np.radians(M)
		
		# Équation du centre
		C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * np.sin(M_rad) + \
			(0.019993 - 0.000101 * T) * np.sin(2 * M_rad) + \
			0.000289 * np.sin(3 * M_rad)
		
		# Longitude vraie du Soleil (en degrés)
		L_true = L0 + C
		
		# Correction pour aberration
		L_apparent = L_true - 0.00569 - 0.00478 * np.sin(np.radians(125.04 - 1934.136 * T))
		
		# Convertir en radians
		L_apparent_rad = np.radians(L_apparent)
		
		# Distance Terre-Soleil en UA (1 UA = 149 597 870.7 km)
		r = 1.000001018 * (1 - 0.016708634 * np.cos(M_rad) - 0.000139589 * np.cos(2 * M_rad))
		
		# Obliquité de l'écliptique (en degrés)
		eps = 23.43929 - 0.01300417 * T - 0.00000016 * T**2
		
		# Convertir en radians
		eps_rad = np.radians(eps)
		
		# Coordonnées équatoriales
		x = r * np.cos(L_apparent_rad)
		y = r * np.cos(eps_rad) * np.sin(L_apparent_rad)
		z = r * np.sin(eps_rad) * np.sin(L_apparent_rad)
		
		# Convertir en km (1 UA = 149 597 870.7 km)
		au_to_km = 149597870.7
		
		return np.array([x, y, z]) * au_to_km