from typing import Tuple, Dict
import numpy as np

class OrbitalElements:
	"""
	Classe représentant les éléments orbitaux d'un satellite.
	Permet de convertir entre éléments orbitaux et vecteurs position/vitesse.
	"""
	
	def __init__(
		self,
		semimajorAxis: float,
		eccentricity: float,
		inclination: float,
		longitudeOfAscendingNode: float,
		argumentOfPeriapsis: float,
		trueAnomaly: float
	):
		"""
		Initialise les éléments orbitaux.
		
		Args:
			semimajorAxis: Demi-grand axe de l'orbite (km)
			eccentricity: Excentricité de l'orbite (sans unité)
			inclination: Inclinaison de l'orbite (radians)
			longitudeOfAscendingNode: Longitude du nœud ascendant (radians)
			argumentOfPeriapsis: Argument du périapse (radians)
			trueAnomaly: Anomalie vraie (radians)
		"""
		self.semimajorAxis: float = semimajorAxis
		self.eccentricity: float = eccentricity
		self.inclination: float = inclination
		self.longitudeOfAscendingNode: float = longitudeOfAscendingNode
		self.argumentOfPeriapsis: float = argumentOfPeriapsis
		self.trueAnomaly: float = trueAnomaly
		
		# Constante gravitationnelle de la Terre (m³/s²)
		self.gravitationalParameter: float = 3.986004418e14
	
	def toPosVel(self) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Convertit les éléments orbitaux en vecteurs position et vitesse dans le référentiel inertiel.
		
		Returns:
			Tuple contenant (position [x, y, z] en km, vitesse [vx, vy, vz] en km/s)
		"""
		# Paramètre pour plus de lisibilité
		a = self.semimajorAxis
		e = self.eccentricity
		i = self.inclination
		omega = self.longitudeOfAscendingNode
		w = self.argumentOfPeriapsis
		nu = self.trueAnomaly
		mu = self.gravitationalParameter / 1e9  # Convertir en km³/s²
		
		# Calcul du paramètre de l'orbite
		p = a * (1 - e**2)
		
		# Distance du satellite au foyer
		r = p / (1 + e * np.cos(nu))
		
		# Position dans le plan orbital
		x_orbit = r * np.cos(nu)
		y_orbit = r * np.sin(nu)
		
		# Matrice de rotation pour passer des coordonnées du plan orbital aux coordonnées ECI
		cos_omega = np.cos(omega)
		sin_omega = np.sin(omega)
		cos_i = np.cos(i)
		sin_i = np.sin(i)
		cos_w = np.cos(w)
		sin_w = np.sin(w)
		
		# Matrice de rotation complète (3x3)
		R11 = cos_omega * cos_w - sin_omega * sin_w * cos_i
		R12 = -cos_omega * sin_w - sin_omega * cos_w * cos_i
		R21 = sin_omega * cos_w + cos_omega * sin_w * cos_i
		R22 = -sin_omega * sin_w + cos_omega * cos_w * cos_i
		R31 = sin_w * sin_i
		R32 = cos_w * sin_i
		
		# Position dans le référentiel inertiel ECI
		x = R11 * x_orbit + R12 * y_orbit
		y = R21 * x_orbit + R22 * y_orbit
		z = R31 * x_orbit + R32 * y_orbit
		
		# Calcul de la vitesse
		# Vitesse dans le plan orbital
		h = np.sqrt(mu * p)
		vx_orbit = -(h / r) * np.sin(nu)
		vy_orbit = (h / r) * (e + np.cos(nu))
		
		# Vitesse dans le référentiel inertiel ECI
		vx = R11 * vx_orbit + R12 * vy_orbit
		vy = R21 * vx_orbit + R22 * vy_orbit
		vz = R31 * vx_orbit + R32 * vy_orbit
		
		return np.array([x, y, z]), np.array([vx, vy, vz])
	
	@classmethod
	def fromPosVel(cls, position: np.ndarray, velocity: np.ndarray, mu: float = 3.986004418e14) -> 'OrbitalElements':
		"""
		Crée un objet OrbitalElements à partir de vecteurs position et vitesse.
		
		Args:
			position: Vecteur position [x, y, z] en km
			velocity: Vecteur vitesse [vx, vy, vz] en km/s
			mu: Paramètre gravitationnel (m³/s²)
			
		Returns:
			Instance d'OrbitalElements correspondant aux vecteurs position et vitesse
		"""
		# Convertir mu en km³/s²
		mu_km = mu / 1e9
		
		# Vecteur moment cinétique
		h_vec = np.cross(position, velocity)
		h = np.linalg.norm(h_vec)
		
		# Vecteur excentricité
		r = np.linalg.norm(position)
		v = np.linalg.norm(velocity)
		v_rad = np.dot(position, velocity) / r  # Composante radiale de la vitesse
		
		e_vec = ((v**2 - mu_km / r) * position - r * v_rad * velocity) / mu_km
		e = np.linalg.norm(e_vec)
		
		# Énergie orbitale spécifique
		energy = v**2 / 2 - mu_km / r
		
		# Demi-grand axe
		if abs(e - 1.0) < 1e-10:  # Orbite parabolique
			a = float('inf')
		else:
			a = -mu_km / (2 * energy)
		
		# Vecteur du nœud ascendant (ligne des nœuds)
		k_vec = np.array([0, 0, 1])  # Vecteur unitaire dans la direction Z
		n_vec = np.cross(k_vec, h_vec)
		n = np.linalg.norm(n_vec)
		
		# Inclinaison
		i = np.arccos(h_vec[2] / h)
		
		# Longitude du nœud ascendant
		if n < 1e-10:  # Orbite équatoriale
			omega = 0.0
		else:
			omega = np.arccos(n_vec[0] / n)
			if n_vec[1] < 0:
				omega = 2 * np.pi - omega
		
		# Argument du périapsis
		if n < 1e-10:  # Orbite équatoriale
			if e < 1e-10:  # Orbite circulaire
				w = 0.0
			else:
				w = np.arccos(e_vec[0] / e)
				if e_vec[1] < 0:
					w = 2 * np.pi - w
		else:
			w = np.arccos(np.dot(n_vec, e_vec) / (n * e))
			if e_vec[2] < 0:
				w = 2 * np.pi - w
		
		# Anomalie vraie
		if e < 1e-10:  # Orbite circulaire
			if n < 1e-10:  # Orbite équatoriale
				nu = np.arctan2(position[1], position[0])
			else:
				nu = np.arccos(np.dot(n_vec, position) / (n * r))
				if np.dot(position, np.cross(n_vec, h_vec)) < 0:
					nu = 2 * np.pi - nu
		else:
			nu = np.arccos(np.dot(e_vec, position) / (e * r))
			if np.dot(position, velocity) < 0:
				nu = 2 * np.pi - nu
		
		return cls(a, e, i, omega, w, nu)
	
	def propagate(self, deltaTime: float) -> 'OrbitalElements':
		"""
		Propage les éléments orbitaux pour un temps donné.
		
		Args:
			deltaTime: Temps de propagation en secondes
			
		Returns:
			Nouveaux éléments orbitaux après propagation
		"""
		# Calculer le moyen mouvement (vitesse angulaire moyenne)
		n = np.sqrt(self.gravitationalParameter / (self.semimajorAxis**3))  # rad/s
		
		# Convertir l'anomalie vraie en anomalie excentrique
		E = self._trueToEccentricAnomaly(self.trueAnomaly, self.eccentricity)
		
		# Convertir l'anomalie excentrique en anomalie moyenne
		M = E - self.eccentricity * np.sin(E)
		
		# Propager l'anomalie moyenne
		M_new = M + n * deltaTime
		
		# Normaliser M_new entre 0 et 2π
		M_new = M_new % (2 * np.pi)
		
		# Convertir la nouvelle anomalie moyenne en anomalie excentrique
		E_new = self._solveKeplersEquation(M_new, self.eccentricity)
		
		# Convertir la nouvelle anomalie excentrique en anomalie vraie
		nu_new = self._eccentricToTrueAnomaly(E_new, self.eccentricity)
		
		# Retourner les nouveaux éléments orbitaux
		return OrbitalElements(
			self.semimajorAxis,
			self.eccentricity,
			self.inclination,
			self.longitudeOfAscendingNode,
			self.argumentOfPeriapsis,
			nu_new
		)
	
	def _trueToEccentricAnomaly(self, nu: float, e: float) -> float:
		"""
		Convertit l'anomalie vraie en anomalie excentrique.
		
		Args:
			nu: Anomalie vraie en radians
			e: Excentricité
			
		Returns:
			Anomalie excentrique en radians
		"""
		return 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
	
	def _eccentricToTrueAnomaly(self, E: float, e: float) -> float:
		"""
		Convertit l'anomalie excentrique en anomalie vraie.
		
		Args:
			E: Anomalie excentrique en radians
			e: Excentricité
			
		Returns:
			Anomalie vraie en radians
		"""
		return 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
	
	def _solveKeplersEquation(self, M: float, e: float, tolerance: float = 1e-8, maxIterations: int = 1000) -> float:
		"""
		Résout l'équation de Kepler M = E - e*sin(E) pour E.
		
		Args:
			M: Anomalie moyenne en radians
			e: Excentricité
			tolerance: Tolérance pour la convergence
			maxIterations: Nombre maximum d'itérations
			
		Returns:
			Anomalie excentrique en radians
		"""
		# Pour les orbites presque circulaires, l'anomalie moyenne est une bonne approximation initiale
		if e < 0.3:
			E = M
		else:
			# Pour les orbites plus excentriques, utiliser l'approximation de Danby
			E = M + np.sign(np.sin(M)) * 0.85 * e
		
		# Méthode de Newton-Raphson pour résoudre l'équation
		iteration = 0
		while iteration < maxIterations:
			E_next = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
			if abs(E_next - E) < tolerance:
				return E_next
			E = E_next
			iteration += 1
		
		# Si on atteint le nombre maximum d'itérations sans converger
		return E
	
	def toDictionary(self) -> Dict[str, float]:
		"""
		Convertit les éléments orbitaux en dictionnaire.
		
		Returns:
			Dictionnaire des éléments orbitaux
		"""
		return {
			"semimajor_axis": self.semimajorAxis,
			"eccentricity": self.eccentricity,
			"inclination": self.inclination,
			"longitude_of_ascending_node": self.longitudeOfAscendingNode,
			"argument_of_periapsis": self.argumentOfPeriapsis,
			"true_anomaly": self.trueAnomaly
		}
	
	@classmethod
	def createCircularOrbit(cls, altitude: float, inclination: float, longitudeOfAscendingNode: float, argumentOfPeriapsis: float = 0.0, trueAnomaly: float = 0.0) -> 'OrbitalElements':
		"""
		Crée une orbite circulaire à une altitude donnée.
		
		Args:
			altitude: Altitude au-dessus de la surface de la Terre (km)
			inclination: Inclinaison de l'orbite (radians)
			longitudeOfAscendingNode: Longitude du nœud ascendant (radians)
			argumentOfPeriapsis: Argument du périapsis (radians)
			trueAnomaly: Anomalie vraie initiale (radians)
			
		Returns:
			Instance d'OrbitalElements pour l'orbite circulaire
		"""
		# Rayon de la Terre (km)
		earthRadius = 6371.0
		
		# Demi-grand axe (altitude + rayon de la Terre)
		semimajorAxis = earthRadius + altitude
		
		# Excentricité = 0 pour une orbite circulaire
		eccentricity = 0.0
		
		return cls(
			semimajorAxis,
			eccentricity,
			inclination,
			longitudeOfAscendingNode,
			argumentOfPeriapsis,
			trueAnomaly
		)
	
	@classmethod
	def createSunSynchronousOrbit(cls, altitude: float, localTimeAtAscendingNode: float, 
								argumentOfPeriapsis: float = 0.0, trueAnomaly: float = 0.0) -> 'OrbitalElements':
		"""
		Crée une orbite héliosynchrone (sun-synchronous).
		
		Args:
			altitude: Altitude au-dessus de la surface de la Terre (km)
			localTimeAtAscendingNode: Heure locale au nœud ascendant (heures)
			argumentOfPeriapsis: Argument du périapsis (radians)
			trueAnomaly: Anomalie vraie initiale (radians)
			
		Returns:
			Instance d'OrbitalElements pour l'orbite héliosynchrone
		"""
		# Rayon de la Terre (km)
		earthRadius = 6371.0
		
		# Constantes pour le calcul de l'inclinaison héliosynchrone
		J2 = 1.08263e-3  # Coefficient J2 du potentiel terrestre
		earthRadius_m = earthRadius * 1000  # Rayon de la Terre en mètres
		
		# Demi-grand axe (altitude + rayon de la Terre)
		semimajorAxis = earthRadius + altitude
		
		# Excentricité (généralement très faible pour ces orbites)
		eccentricity = 0.0
		
		# Précession requise pour maintenir l'orbite héliosynchrone (rad/s)
		earthOrbitalRate = 2 * np.pi / 365.2564  # rad/jour
		
		# Calculer l'inclinaison requise
		a_m = semimajorAxis * 1000  # demi-grand axe en mètres
		n = np.sqrt(cls.gravitationalParameter / (a_m**3))  # moyen mouvement (rad/s)
		
		cosI = -2 * earthOrbitalRate / (3 * n * J2 * (earthRadius_m / a_m)**2)
		cosI = np.clip(cosI, -1.0, 1.0)  # Assurer que cosI est dans l'intervalle [-1, 1]
		
		inclination = np.arccos(cosI)
		
		# Calculer la longitude du nœud ascendant en fonction de l'heure locale
		# Convertir l'heure locale en angle (15° par heure)
		angle = (localTimeAtAscendingNode - 12) * 15 * np.pi / 180
		# L'angle est par rapport à la direction du Soleil
		longitudeOfAscendingNode = angle
		
		return cls(
			semimajorAxis,
			eccentricity,
			inclination,
			longitudeOfAscendingNode,
			argumentOfPeriapsis,
			trueAnomaly
		)
	
	@classmethod
	def createConstellationPlane(cls, numSatellites: int, altitude: float, inclination: float, 
							  longitudeOfAscendingNode: float, eccentricity: float = 0.0) -> list['OrbitalElements']:
		"""
		Crée un plan de constellation avec des satellites également espacés.
		
		Args:
			numSatellites: Nombre de satellites dans le plan
			altitude: Altitude de l'orbite (km)
			inclination: Inclinaison du plan orbital (radians)
			longitudeOfAscendingNode: Longitude du nœud ascendant (radians)
			eccentricity: Excentricité des orbites
			
		Returns:
			Liste d'instances d'OrbitalElements pour chaque satellite
		"""
		# Rayon de la Terre (km)
		earthRadius = 6371.0
		
		# Demi-grand axe (altitude + rayon de la Terre)
		semimajorAxis = earthRadius + altitude
		
		# Créer les éléments orbitaux pour chaque satellite
		orbitalElements = []
		
		for i in range(numSatellites):
			# Répartir les anomalies vraies uniformément
			trueAnomaly = 2 * np.pi * i / numSatellites
			
			# Pour les orbites non circulaires, on utilise souvent l'anomalie moyenne
			# pour espacer uniformément les satellites
			if eccentricity > 0:
				# Convertir l'anomalie vraie en anomalie excentrique
				E = cls._trueToEccentricAnomaly(cls, trueAnomaly, eccentricity)
				
				# Convertir l'anomalie excentrique en anomalie moyenne
				M = E - eccentricity * np.sin(E)
				
				# Répartir les anomalies moyennes uniformément
				M_spaced = 2 * np.pi * i / numSatellites
				
				# Reconvertir en anomalie excentrique
				E_spaced = cls._solveKeplersEquation(cls, M_spaced, eccentricity)
				
				# Reconvertir en anomalie vraie
				trueAnomaly = cls._eccentricToTrueAnomaly(cls, E_spaced, eccentricity)
			
			# Créer l'élément orbital pour ce satellite
			orbital = cls(
				semimajorAxis,
				eccentricity,
				inclination,
				longitudeOfAscendingNode,
				0.0,  # Argument du périapsis (0 pour une orbite circulaire)
				trueAnomaly
			)
			
			orbitalElements.append(orbital)
		
		return orbitalElements