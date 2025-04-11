import numpy as np
from typing import Tuple, List, Optional, Union, Callable
import math

# Types pour les annotations de type
Vector2D = Tuple[float, float]
Vector3D = Tuple[float, float, float]
Vector = Union[Vector2D, Vector3D, np.ndarray]
Matrix = np.ndarray
Quaternion = Tuple[float, float, float, float]  # (w, x, y, z)


def normalize_vector(v: Vector) -> np.ndarray:
	"""
	Normalise un vecteur à une longueur unitaire.
	
	Args:
		v: Vecteur à normaliser
		
	Returns:
		Vecteur normalisé
	"""
	v_array = np.array(v, dtype=np.float32)
	norm = np.linalg.norm(v_array)
	
	# Éviter la division par zéro
	if norm < 1e-10:
		return np.zeros_like(v_array)
		
	return v_array / norm


def distance(a: Vector, b: Vector) -> float:
	"""
	Calcule la distance euclidienne entre deux points.
	
	Args:
		a: Premier point
		b: Second point
		
	Returns:
		Distance entre les points
	"""
	return np.linalg.norm(np.array(a) - np.array(b))


def squared_distance(a: Vector, b: Vector) -> float:
	"""
	Calcule le carré de la distance euclidienne entre deux points.
	Plus rapide que la distance quand seule la comparaison est nécessaire.
	
	Args:
		a: Premier point
		b: Second point
		
	Returns:
		Carré de la distance entre les points
	"""
	diff = np.array(a) - np.array(b)
	return np.dot(diff, diff)


def dot_product(a: Vector, b: Vector) -> float:
	"""
	Calcule le produit scalaire de deux vecteurs.
	
	Args:
		a: Premier vecteur
		b: Second vecteur
		
	Returns:
		Produit scalaire
	"""
	return np.dot(np.array(a), np.array(b))


def cross_product(a: Vector3D, b: Vector3D) -> np.ndarray:
	"""
	Calcule le produit vectoriel de deux vecteurs 3D.
	
	Args:
		a: Premier vecteur
		b: Second vecteur
		
	Returns:
		Produit vectoriel
	"""
	return np.cross(np.array(a), np.array(b))


def angle_between(a: Vector, b: Vector) -> float:
	"""
	Calcule l'angle en radians entre deux vecteurs.
	
	Args:
		a: Premier vecteur
		b: Second vecteur
		
	Returns:
		Angle en radians
	"""
	a_norm = normalize_vector(a)
	b_norm = normalize_vector(b)
	
	# Limiter la valeur du produit scalaire à [-1, 1] pour éviter les erreurs numériques
	cos_angle = np.clip(dot_product(a_norm, b_norm), -1.0, 1.0)
	
	return np.arccos(cos_angle)


def project_vector(v: Vector, onto: Vector) -> np.ndarray:
	"""
	Projette un vecteur sur un autre.
	
	Args:
		v: Vecteur à projeter
		onto: Vecteur sur lequel projeter
		
	Returns:
		Vecteur projeté
	"""
	onto_array = np.array(onto, dtype=np.float32)
	onto_normalized = normalize_vector(onto_array)
	
	return dot_product(v, onto_normalized) * onto_normalized


def reflect_vector(v: Vector, normal: Vector) -> np.ndarray:
	"""
	Réfléchit un vecteur par rapport à une normale.
	
	Args:
		v: Vecteur à réfléchir
		normal: Vecteur normal (doit être normalisé)
		
	Returns:
		Vecteur réfléchi
	"""
	v_array = np.array(v, dtype=np.float32)
	normal_array = np.array(normal, dtype=np.float32)
	
	# Vérifier que la normale est normalisée
	normal_array = normalize_vector(normal_array)
	
	# Calculer la réflexion: r = v - 2(v·n)n
	return v_array - 2 * dot_product(v_array, normal_array) * normal_array


def euler_to_rotation_matrix(euler_angles: Vector3D) -> np.ndarray:
	"""
	Convertit des angles d'Euler (pitch, yaw, roll) en matrice de rotation 3x3.
	
	Args:
		euler_angles: Angles d'Euler en radians (pitch, yaw, roll)
		
	Returns:
		Matrice de rotation 3x3
	"""
	pitch, yaw, roll = euler_angles
	
	# Calcul des sinus et cosinus
	cp, sp = np.cos(pitch), np.sin(pitch)
	cy, sy = np.cos(yaw), np.sin(yaw)
	cr, sr = np.cos(roll), np.sin(roll)
	
	# Construire la matrice de rotation
	rotation_matrix = np.array([
		[cy*cr, cy*sr*sp - sy*cp, cy*sr*cp + sy*sp],
		[sy*cr, sy*sr*sp + cy*cp, sy*sr*cp - cy*sp],
		[-sr, cr*sp, cr*cp]
	], dtype=np.float32)
	
	return rotation_matrix


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
	"""
	Convertit une matrice de rotation 3x3 en angles d'Euler (pitch, yaw, roll).
	
	Args:
		rotation_matrix: Matrice de rotation 3x3
		
	Returns:
		Angles d'Euler en radians (pitch, yaw, roll)
	"""
	# Extraction des angles d'Euler
	roll = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2))
	
	# Cas particulier: Gimbal lock
	if np.isclose(rotation_matrix[2, 0], 1.0):
		pitch = 0.0
		yaw = np.arctan2(rotation_matrix[0, 1], rotation_matrix[1, 1])
	elif np.isclose(rotation_matrix[2, 0], -1.0):
		pitch = 0.0
		yaw = np.arctan2(-rotation_matrix[0, 1], -rotation_matrix[1, 1])
	else:
		pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
		yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
	
	return np.array([pitch, yaw, roll], dtype=np.float32)


def euler_to_quaternion(euler_angles: Vector3D) -> Quaternion:
	"""
	Convertit des angles d'Euler (pitch, yaw, roll) en quaternion.
	
	Args:
		euler_angles: Angles d'Euler en radians (pitch, yaw, roll)
		
	Returns:
		Quaternion (w, x, y, z)
	"""
	pitch, yaw, roll = euler_angles
	
	# Demi-angles
	cp, sp = np.cos(pitch/2), np.sin(pitch/2)
	cy, sy = np.cos(yaw/2), np.sin(yaw/2)
	cr, sr = np.cos(roll/2), np.sin(roll/2)
	
	# Calcul des composantes du quaternion
	w = cr * cp * cy + sr * sp * sy
	x = sr * cp * cy - cr * sp * sy
	y = cr * sp * cy + sr * cp * sy
	z = cr * cp * sy - sr * sp * cy
	
	return (w, x, y, z)


def quaternion_to_euler(quaternion: Quaternion) -> np.ndarray:
	"""
	Convertit un quaternion en angles d'Euler (pitch, yaw, roll).
	
	Args:
		quaternion: Quaternion (w, x, y, z)
		
	Returns:
		Angles d'Euler en radians (pitch, yaw, roll)
	"""
	w, x, y, z = quaternion
	
	# Roll (x-axis rotation)
	sinr_cosp = 2 * (w * x + y * z)
	cosr_cosp = 1 - 2 * (x * x + y * y)
	roll = np.arctan2(sinr_cosp, cosr_cosp)
	
	# Pitch (y-axis rotation)
	sinp = 2 * (w * y - z * x)
	if abs(sinp) >= 1:
		pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
	else:
		pitch = np.arcsin(sinp)
	
	# Yaw (z-axis rotation)
	siny_cosp = 2 * (w * z + x * y)
	cosy_cosp = 1 - 2 * (y * y + z * z)
	yaw = np.arctan2(siny_cosp, cosy_cosp)
	
	return np.array([pitch, yaw, roll], dtype=np.float32)


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
	"""
	Multiplie deux quaternions.
	
	Args:
		q1: Premier quaternion (w, x, y, z)
		q2: Second quaternion (w, x, y, z)
		
	Returns:
		Quaternion résultant
	"""
	w1, x1, y1, z1 = q1
	w2, x2, y2, z2 = q2
	
	w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
	x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
	y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
	z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
	
	return (w, x, y, z)


def quaternion_conjugate(q: Quaternion) -> Quaternion:
	"""
	Calcule le conjugué d'un quaternion.
	
	Args:
		q: Quaternion (w, x, y, z)
		
	Returns:
		Quaternion conjugué
	"""
	w, x, y, z = q
	return (w, -x, -y, -z)


def quaternion_normalize(q: Quaternion) -> Quaternion:
	"""
	Normalise un quaternion.
	
	Args:
		q: Quaternion (w, x, y, z)
		
	Returns:
		Quaternion normalisé
	"""
	w, x, y, z = q
	norm = np.sqrt(w*w + x*x + y*y + z*z)
	
	if norm < 1e-10:
		return (1.0, 0.0, 0.0, 0.0)
		
	return (w/norm, x/norm, y/norm, z/norm)


def rotate_vector_by_quaternion(v: Vector3D, q: Quaternion) -> np.ndarray:
	"""
	Applique une rotation représentée par un quaternion à un vecteur.
	
	Args:
		v: Vecteur 3D à tourner
		q: Quaternion représentant la rotation
		
	Returns:
		Vecteur tourné
	"""
	# Convertir le vecteur en quaternion pur (0, vx, vy, vz)
	q_vec = (0.0, v[0], v[1], v[2])
	
	# Calculer q * q_vec * q^(-1)
	q_conj = quaternion_conjugate(q)
	q_result = quaternion_multiply(q, quaternion_multiply(q_vec, q_conj))
	
	# Extraire la partie vectorielle
	return np.array([q_result[1], q_result[2], q_result[3]], dtype=np.float32)


def lerp(a: float, b: float, t: float) -> float:
	"""
	Réalise une interpolation linéaire entre deux valeurs.
	
	Args:
		a: Première valeur
		b: Seconde valeur
		t: Facteur d'interpolation (0-1)
		
	Returns:
		Valeur interpolée
	"""
	return (1 - t) * a + t * b


def slerp(q1: Quaternion, q2: Quaternion, t: float) -> Quaternion:
	"""
	Réalise une interpolation sphérique entre deux quaternions.
	
	Args:
		q1: Premier quaternion
		q2: Second quaternion
		t: Facteur d'interpolation (0-1)
		
	Returns:
		Quaternion interpolé
	"""
	# Normaliser les quaternions
	q1_norm = quaternion_normalize(q1)
	q2_norm = quaternion_normalize(q2)
	
	# Calculer le produit scalaire
	dot = q1_norm[0]*q2_norm[0] + q1_norm[1]*q2_norm[1] + q1_norm[2]*q2_norm[2] + q1_norm[3]*q2_norm[3]
	
	# Si les quaternions sont très proches, utiliser LERP
	if abs(dot) > 0.9995:
		result = (
			lerp(q1_norm[0], q2_norm[0], t),
			lerp(q1_norm[1], q2_norm[1], t),
			lerp(q1_norm[2], q2_norm[2], t),
			lerp(q1_norm[3], q2_norm[3], t)
		)
		return quaternion_normalize(result)
	
	# Limiter la valeur du produit scalaire à [-1, 1]
	dot = np.clip(dot, -1.0, 1.0)
	
	# Calculer l'angle entre les quaternions
	theta_0 = np.arccos(dot)
	theta = theta_0 * t
	
	# Calculer le quaternion interpolé
	sin_theta = np.sin(theta)
	sin_theta_0 = np.sin(theta_0)
	
	s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
	s1 = sin_theta / sin_theta_0
	
	result = (
		s0 * q1_norm[0] + s1 * q2_norm[0],
		s0 * q1_norm[1] + s1 * q2_norm[1],
		s0 * q1_norm[2] + s1 * q2_norm[2],
		s0 * q1_norm[3] + s1 * q2_norm[3]
	)
	
	return quaternion_normalize(result)


def bezier_curve(control_points: List[Vector], t: float) -> np.ndarray:
	"""
	Calcule un point sur une courbe de Bézier.
	
	Args:
		control_points: Liste des points de contrôle
		t: Paramètre de la courbe (0-1)
		
	Returns:
		Point sur la courbe
	"""
	n = len(control_points) - 1
	point = np.zeros(len(control_points[0]), dtype=np.float32)
	
	for i, ctrl_pt in enumerate(control_points):
		# Coefficient binomial
		coef = math.comb(n, i)
		# Calcul du terme de Bernstein
		term = coef * (t ** i) * ((1 - t) ** (n - i))
		# Ajouter la contribution du point de contrôle
		point += term * np.array(ctrl_pt, dtype=np.float32)
		
	return point


def catmull_rom_spline(p0: Vector, p1: Vector, p2: Vector, p3: Vector, t: float) -> np.ndarray:
	"""
	Calcule un point sur une spline de Catmull-Rom.
	
	Args:
		p0, p1, p2, p3: Points de contrôle
		t: Paramètre de la spline (0-1)
		
	Returns:
		Point sur la spline
	"""
	t2 = t * t
	t3 = t2 * t
	
	coefficients = np.array([
		[0, 1, 0, 0],
		[-0.5, 0, 0.5, 0],
		[1, -2.5, 2, -0.5],
		[-0.5, 1.5, -1.5, 0.5]
	], dtype=np.float32)
	
	# Vecteur de puissances de t
	t_powers = np.array([1, t, t2, t3])
	
	# Matrice des points de contrôle
	points = np.array([p0, p1, p2, p3], dtype=np.float32)
	
	# Calcul du point sur la spline
	basis = np.dot(coefficients, t_powers)
	return np.dot(basis, points)


def signed_angle(v1: Vector2D, v2: Vector2D) -> float:
	"""
	Calcule l'angle signé entre deux vecteurs 2D.
	
	Args:
		v1: Premier vecteur 2D
		v2: Second vecteur 2D
		
	Returns:
		Angle signé en radians
	"""
	# Calculer l'angle non signé
	angle = angle_between((v1[0], v1[1], 0), (v2[0], v2[1], 0))
	
	# Déterminer le signe en utilisant le produit en croix 2D
	cross_z = v1[0] * v2[1] - v1[1] * v2[0]
	
	# Appliquer le signe
	return angle if cross_z >= 0 else -angle


def ray_sphere_intersection(ray_origin: Vector3D, 
						   ray_direction: Vector3D, 
						   sphere_center: Vector3D, 
						   sphere_radius: float) -> Optional[Tuple[float, float]]:
	"""
	Calcule l'intersection d'un rayon avec une sphère.
	
	Args:
		ray_origin: Origine du rayon
		ray_direction: Direction du rayon (normalisée)
		sphere_center: Centre de la sphère
		sphere_radius: Rayon de la sphère
		
	Returns:
		Tuple (t1, t2) des distances d'intersection, ou None s'il n'y a pas d'intersection
	"""
	ray_dir_norm = normalize_vector(ray_direction)
	
	# Vecteur du centre de la sphère à l'origine du rayon
	oc = np.array(ray_origin) - np.array(sphere_center)
	
	# Coefficients de l'équation quadratique
	a = 1.0  # dot_product(ray_dir_norm, ray_dir_norm) = 1 car vecteur normalisé
	b = 2.0 * dot_product(oc, ray_dir_norm)
	c = dot_product(oc, oc) - sphere_radius * sphere_radius
	
	# Discriminant
	discriminant = b * b - 4 * a * c
	
	if discriminant < 0:
		return None  # Pas d'intersection
	
	# Distances aux points d'intersection
	sqrtd = np.sqrt(discriminant)
	t1 = (-b - sqrtd) / (2.0 * a)
	t2 = (-b + sqrtd) / (2.0 * a)
	
	return (t1, t2)


def ray_plane_intersection(ray_origin: Vector3D, 
						  ray_direction: Vector3D, 
						  plane_point: Vector3D, 
						  plane_normal: Vector3D) -> Optional[float]:
	"""
	Calcule l'intersection d'un rayon avec un plan.
	
	Args:
		ray_origin: Origine du rayon
		ray_direction: Direction du rayon
		plane_point: Point sur le plan
		plane_normal: Normale du plan
		
	Returns:
		Distance d'intersection, ou None s'il n'y a pas d'intersection
	"""
	ray_dir_norm = normalize_vector(ray_direction)
	plane_norm = normalize_vector(plane_normal)
	
	# Vérifier si le rayon est parallèle au plan
	denom = dot_product(ray_dir_norm, plane_norm)
	
	if abs(denom) < 1e-6:
		return None  # Rayon parallèle au plan
	
	# Calculer la distance
	t = dot_product(np.array(plane_point) - np.array(ray_origin), plane_norm) / denom
	
	# Vérifier si l'intersection est devant l'origine du rayon
	if t < 0:
		return None
	
	return t


def smooth_step(edge0: float, edge1: float, x: float) -> float:
	"""
	Applique une fonction de lissage Hermite à x.
	
	Args:
		edge0: Valeur inférieure du seuil
		edge1: Valeur supérieure du seuil
		x: Valeur à lisser
		
	Returns:
		Valeur lissée entre 0 et 1
	"""
	# Limiter x à l'intervalle [0,1]
	x = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
	
	# Polynôme de lissage: 3x² - 2x³
	return x * x * (3 - 2 * x)


def perlin_noise(x: float, y: float, z: float, seed: int = 0) -> float:
	"""
	Implémentation simplifiée du bruit de Perlin 3D.
	Pour une implémentation complète, utiliser une bibliothèque comme 'noise'.
	
	Args:
		x, y, z: Coordonnées
		seed: Graine pour la génération
		
	Returns:
		Valeur de bruit entre -1 et 1
	"""
	try:
		import noise
		return noise.pnoise3(x, y, z, base=seed, octaves=1)
	except ImportError:
		# Implémentation de secours simplifiée (inexacte)
		np.random.seed(seed)
		p = np.arange(256, dtype=int)
		np.random.shuffle(p)
		p = np.concatenate((p, p))
		
		# Coordonnées entières
		xi, yi, zi = int(x) & 255, int(y) & 255, int(z) & 255
		
		# Fractions
		xf, yf, zf = x - int(x), y - int(y), z - int(z)
		
		# Fonction de fondu
		u, v, w = xf**3 * (xf * (xf * 6 - 15) + 10), yf**3 * (yf * (yf * 6 - 15) + 10), zf**3 * (zf * (zf * 6 - 15) + 10)
		
		# Permutation
		perm = lambda i: p[i]
		
		# Permutation des coordonnées
		a = perm(xi) + yi
		aa = perm(a) + zi
		ab = perm(a + 1) + zi
		b = perm(xi + 1) + yi
		ba = perm(b) + zi
		bb = perm(b + 1) + zi
		
		# Fonction de hachage simplifiée
		hash_func = lambda n: ((n * 1836311903) ^ (n * 2971215073) + 4807526976) & 1023
		
		# Extraire les valeurs de gradient
		g1 = hash_func(aa) / 1023.0 * 2.0 - 1.0
		g2 = hash_func(ab) / 1023.0 * 2.0 - 1.0
		g3 = hash_func(ba) / 1023.0 * 2.0 - 1.0
		g4 = hash_func(bb) / 1023.0 * 2.0 - 1.0
		g5 = hash_func(aa + 1) / 1023.0 * 2.0 - 1.0
		g6 = hash_func(ab + 1) / 1023.0 * 2.0 - 1.0
		g7 = hash_func(ba + 1) / 1023.0 * 2.0 - 1.0
		g8 = hash_func(bb + 1) / 1023.0 * 2.0 - 1.0
		
		# Interpolation des valeurs de gradient
		c1 = lerp(lerp(g1, g2, w), lerp(g3, g4, w), v)
		c2 = lerp(lerp(g5, g6, w), lerp(g7, g8, w), v)
		return lerp(c1, c2, u)


def fluid_resistance(velocity: Vector, 
					area: float, 
					drag_coefficient: float, 
					fluid_density: float = 1000.0) -> np.ndarray:
	"""
	Calcule la force de résistance d'un fluide sur un objet.
	
	Args:
		velocity: Vecteur de vitesse de l'objet par rapport au fluide
		area: Surface frontale de l'objet
		drag_coefficient: Coefficient de traînée
		fluid_density: Densité du fluide (eau par défaut)
		
	Returns:
		Force de résistance (N)
	"""
	v = np.array(velocity, dtype=np.float32)
	v_squared = np.linalg.norm(v) ** 2
	
	# Si vitesse nulle, pas de résistance
	if v_squared < 1e-10:
		return np.zeros_like(v)
	
	# Direction de la résistance (opposée à la vitesse)
	direction = -v / np.linalg.norm(v)
	
	# Équation de la traînée: F = 0.5 * ρ * v² * Cd * A
	magnitude = 0.5 * fluid_density * v_squared * drag_coefficient * area
	
	return direction * magnitude


def buoyancy_force(volume: float, 
				  fluid_density: float, 
				  gravity: float = 9.81, 
				  direction: Vector3D = (0.0, 1.0, 0.0)) -> np.ndarray:
	"""
	Calcule la force de flottabilité (poussée d'Archimède).
	
	Args:
		volume: Volume de fluide déplacé (m³)
		fluid_density: Densité du fluide (kg/m³)
		gravity: Accélération gravitationnelle (m/s²)
		direction: Direction de la flottabilité (généralement vers le haut)
		
	Returns:
		Force de flottabilité (N)
	"""
	# Formule d'Archimède: F_b = ρ * g * V
	magnitude = fluid_density * gravity * volume
	direction_norm = normalize_vector(direction)
	
	return direction_norm * magnitude


def pressure_to_depth(pressure: float, 
					 atmospheric_pressure: float = 101325.0,
					 fluid_density: float = 1000.0, 
					 gravity: float = 9.81) -> float:
	"""
	Convertit une pression en profondeur dans un fluide.
	
	Args:
		pressure: Pression (Pa)
		atmospheric_pressure: Pression atmosphérique (Pa)
		fluid_density: Densité du fluide (kg/m³)
		gravity: Accélération gravitationnelle (m/s²)
		
	Returns:
		Profondeur (m)
	"""
	# Formule: P = P_atm + ρ * g * h
	# => h = (P - P_atm) / (ρ * g)
	return (pressure - atmospheric_pressure) / (fluid_density * gravity)


def depth_to_pressure(depth: float, 
					 atmospheric_pressure: float = 101325.0,
					 fluid_density: float = 1000.0, 
					 gravity: float = 9.81) -> float:
	"""
	Convertit une profondeur en pression dans un fluide.
	
	Args:
		depth: Profondeur (m)
		atmospheric_pressure: Pression atmosphérique (Pa)
		fluid_density: Densité du fluide (kg/m³)
		gravity: Accélération gravitationnelle (m/s²)
		
	Returns:
		Pression (Pa)
	"""
	# Formule: P = P_atm + ρ * g * h
	return atmospheric_pressure + fluid_density * gravity * depth


def light_attenuation(initial_intensity: float, 
					 depth: float, 
					 attenuation_coefficient: float = 0.2) -> float:
	"""
	Calcule l'atténuation de la lumière avec la profondeur dans l'eau.
	
	Args:
		initial_intensity: Intensité de la lumière à la surface
		depth: Profondeur dans l'eau (m)
		attenuation_coefficient: Coefficient d'atténuation
		
	Returns:
		Intensité de la lumière à la profondeur donnée
	"""
	# Loi de Beer-Lambert: I = I_0 * e^(-μ * x)
	return initial_intensity * np.exp(-attenuation_coefficient * depth)


def viscous_damping(velocity: Vector, 
				   mass: float, 
				   damping_coefficient: float, 
				   delta_time: float) -> np.ndarray:
	"""
	Calcule la vitesse après amortissement visqueux.
	
	Args:
		velocity: Vitesse actuelle
		mass: Masse de l'objet
		damping_coefficient: Coefficient d'amortissement
		delta_time: Intervalle de temps
		
	Returns:
		Nouvelle vitesse après amortissement
	"""
	# Facteur d'amortissement: e^(-(c/m) * dt)
	damping_factor = np.exp(-(damping_coefficient / mass) * delta_time)
	
	return np.array(velocity, dtype=np.float32) * damping_factor


def gaussian_2d(x: float, y: float, 
			   amplitude: float = 1.0, 
			   center_x: float = 0.0, 
			   center_y: float = 0.0,
			   sigma_x: float = 1.0, 
			   sigma_y: float = 1.0) -> float:
	"""
	Calcule la valeur d'une fonction gaussienne 2D.
	Utile pour les champs de potentiel et distributions spatiales.
	
	Args:
		x, y: Coordonnées du point
		amplitude: Amplitude maximum de la gaussienne
		center_x, center_y: Centre de la gaussienne
		sigma_x, sigma_y: Écarts-types dans les directions x et y
		
	Returns:
		Valeur de la gaussienne au point (x, y)
	"""
	# Formule: A * exp(-((x-x0)²/(2σx²) + (y-y0)²/(2σy²)))
	exponent = -((x - center_x)**2 / (2 * sigma_x**2) + (y - center_y)**2 / (2 * sigma_y**2))
	return amplitude * np.exp(exponent)