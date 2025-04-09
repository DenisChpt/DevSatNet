import numpy as np
from typing import Tuple

class CoordinateTransforms:
	"""
	Classe utilitaire pour les transformations de coordonnées entre différents 
	systèmes de référence utilisés dans la simulation de satellites.
	"""
	
	@staticmethod
	def eciToEcef(position: np.ndarray, velocity: np.ndarray, gst: float) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Convertit des coordonnées du référentiel ECI (Earth-Centered Inertial) 
		vers ECEF (Earth-Centered Earth-Fixed).
		
		Args:
			position: Vecteur position [x, y, z] en km dans ECI
			velocity: Vecteur vitesse [vx, vy, vz] en km/s dans ECI
			gst: Temps sidéral de Greenwich en radians
			
		Returns:
			Tuple de (position ECEF, vitesse ECEF)
		"""
		# Matrice de rotation de ECI vers ECEF
		cos_gst = np.cos(gst)
		sin_gst = np.sin(gst)
		
		rotation_matrix = np.array([
			[cos_gst, sin_gst, 0],
			[-sin_gst, cos_gst, 0],
			[0, 0, 1]
		])
		
		# Matrice de rotation pour la vitesse angulaire de la Terre
		earth_rotation_rate = 7.2921150e-5  # rad/s
		omega_matrix = np.array([
			[0, earth_rotation_rate, 0],
			[-earth_rotation_rate, 0, 0],
			[0, 0, 0]
		])
		
		# Calcul de la position ECEF
		position_ecef = rotation_matrix @ position
		
		# Calcul de la vitesse ECEF
		velocity_ecef = rotation_matrix @ velocity - omega_matrix @ position_ecef
		
		return position_ecef, velocity_ecef
	
	@staticmethod
	def ecefToEci(position: np.ndarray, velocity: np.ndarray, gst: float) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Convertit des coordonnées du référentiel ECEF (Earth-Centered Earth-Fixed) 
		vers ECI (Earth-Centered Inertial).
		
		Args:
			position: Vecteur position [x, y, z] en km dans ECEF
			velocity: Vecteur vitesse [vx, vy, vz] en km/s dans ECEF
			gst: Temps sidéral de Greenwich en radians
			
		Returns:
			Tuple de (position ECI, vitesse ECI)
		"""
		# Matrice de rotation de ECEF vers ECI (transposée de la matrice ECI vers ECEF)
		cos_gst = np.cos(gst)
		sin_gst = np.sin(gst)
		
		rotation_matrix = np.array([
			[cos_gst, -sin_gst, 0],
			[sin_gst, cos_gst, 0],
			[0, 0, 1]
		])
		
		# Matrice de rotation pour la vitesse angulaire de la Terre
		earth_rotation_rate = 7.2921150e-5  # rad/s
		omega_matrix = np.array([
			[0, earth_rotation_rate, 0],
			[-earth_rotation_rate, 0, 0],
			[0, 0, 0]
		])
		
		# Calcul de la position ECI
		position_eci = rotation_matrix @ position
		
		# Calcul de la vitesse ECI
		velocity_eci = rotation_matrix @ (velocity + omega_matrix @ position)
		
		return position_eci, velocity_eci
	
	@staticmethod
	def ecefToGeodetic(position: np.ndarray) -> Tuple[float, float, float]:
		"""
		Convertit des coordonnées ECEF (Earth-Centered Earth-Fixed) 
		vers des coordonnées géodétiques (latitude, longitude, altitude).
		
		Args:
			position: Vecteur position [x, y, z] en km dans ECEF
			
		Returns:
			Tuple de (latitude en degrés, longitude en degrés, altitude en km)
		"""
		x, y, z = position
		
		# Paramètres de l'ellipsoïde WGS-84
		a = 6378.137  # demi-grand axe en km
		f = 1/298.257223563  # aplatissement
		b = a * (1 - f)  # demi-petit axe en km
		e_squared = 1 - (b**2 / a**2)  # carré de l'excentricité
		
		# Longitude (simple)
		longitude = np.arctan2(y, x)
		
		# Distance du point à l'axe Z
		p = np.sqrt(x**2 + y**2)
		
		# Première approximation de la latitude
		latitude = np.arctan2(z, p * (1 - e_squared))
		
		# Itération pour améliorer la précision
		for _ in range(5):  # Généralement, 3-5 itérations sont suffisantes
			sin_lat = np.sin(latitude)
			N = a / np.sqrt(1 - e_squared * sin_lat**2)  # rayon de courbure prime vertical
			
			# Nouvelle estimation de la latitude
			latitude = np.arctan2(z + e_squared * N * sin_lat, p)
		
		# Altitude au-dessus de l'ellipsoïde
		sin_lat = np.sin(latitude)
		N = a / np.sqrt(1 - e_squared * sin_lat**2)
		altitude = p / np.cos(latitude) - N
		
		# Convertir en degrés
		latitude_deg = np.degrees(latitude)
		longitude_deg = np.degrees(longitude)
		
		return latitude_deg, longitude_deg, altitude
	
	@staticmethod
	def geodeticToEcef(latitude: float, longitude: float, altitude: float) -> np.ndarray:
		"""
		Convertit des coordonnées géodétiques (latitude, longitude, altitude)
		vers des coordonnées ECEF (Earth-Centered Earth-Fixed).
		
		Args:
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			altitude: Altitude en km au-dessus de l'ellipsoïde
			
		Returns:
			Vecteur position [x, y, z] en km dans ECEF
		"""
		# Convertir en radians
		lat_rad = np.radians(latitude)
		lon_rad = np.radians(longitude)
		
		# Paramètres de l'ellipsoïde WGS-84
		a = 6378.137  # demi-grand axe en km
		f = 1/298.257223563  # aplatissement
		e_squared = 2*f - f**2  # carré de l'excentricité
		
		# Rayon de courbure prime vertical
		N = a / np.sqrt(1 - e_squared * np.sin(lat_rad)**2)
		
		# Calcul des coordonnées ECEF
		x = (N + altitude) * np.cos(lat_rad) * np.cos(lon_rad)
		y = (N + altitude) * np.cos(lat_rad) * np.sin(lon_rad)
		z = (N * (1 - e_squared) + altitude) * np.sin(lat_rad)
		
		return np.array([x, y, z])
	
	@staticmethod
	def eciToLla(position: np.ndarray, gst: float) -> Tuple[float, float, float]:
		"""
		Convertit des coordonnées ECI (Earth-Centered Inertial) 
		vers des coordonnées géodétiques (latitude, longitude, altitude).
		
		Args:
			position: Vecteur position [x, y, z] en km dans ECI
			gst: Temps sidéral de Greenwich en radians
			
		Returns:
			Tuple de (latitude en degrés, longitude en degrés, altitude en km)
		"""
		# Convertir d'abord en ECEF
		position_ecef, _ = CoordinateTransforms.eciToEcef(position, np.zeros(3), gst)
		
		# Puis convertir ECEF en géodétique
		return CoordinateTransforms.ecefToGeodetic(position_ecef)
	
	@staticmethod
	def llaToEci(latitude: float, longitude: float, altitude: float, gst: float) -> np.ndarray:
		"""
		Convertit des coordonnées géodétiques (latitude, longitude, altitude)
		vers des coordonnées ECI (Earth-Centered Inertial).
		
		Args:
			latitude: Latitude en degrés
			longitude: Longitude en degrés
			altitude: Altitude en km au-dessus de l'ellipsoïde
			gst: Temps sidéral de Greenwich en radians
			
		Returns:
			Vecteur position [x, y, z] en km dans ECI
		"""
		# Convertir d'abord en ECEF
		position_ecef = CoordinateTransforms.geodeticToEcef(latitude, longitude, altitude)
		
		# Puis convertir ECEF en ECI
		position_eci, _ = CoordinateTransforms.ecefToEci(position_ecef, np.zeros(3), gst)
		
		return position_eci