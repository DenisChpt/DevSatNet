#!/usr/bin/env python
import os
import argparse
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from environment.utils.OrbitalElements import OrbitalElements
from environment.EarthModel import EarthModel
from visualization.OrbitRenderer import OrbitRenderer

def parse_arguments():
	"""
	Parse les arguments de ligne de commande.
	
	Returns:
		Arguments parsés
	"""
	parser = argparse.ArgumentParser(description='Générateur de constellation de satellites DeepSatNet')
	
	# Arguments généraux
	parser.add_argument('--output-dir', type=str, default='constellation_configs', 
						help='Répertoire pour enregistrer les configurations')
	parser.add_argument('--name', type=str, default=None, 
						help='Nom de la constellation (par défaut: timestamp)')
	
	# Arguments spécifiques à la constellation
	parser.add_argument('--type', type=str, default='walker', choices=['walker', 'polar', 'equatorial', 'sun-sync', 'custom'], 
						help='Type de constellation')
	parser.add_argument('--num-planes', type=int, default=3, 
						help='Nombre de plans orbitaux')
	parser.add_argument('--satellites-per-plane', type=int, default=10, 
						help='Nombre de satellites par plan')
	parser.add_argument('--altitude', type=float, default=800.0, 
						help='Altitude en km')
	parser.add_argument('--inclination', type=float, default=53.0, 
						help='Inclinaison en degrés')
	parser.add_argument('--eccentricity', type=float, default=0.0, 
						help='Excentricité')
	parser.add_argument('--phasing', type=float, default=0.0, 
						help='Déphasage entre plans adjacent (en degrés)')
	parser.add_argument('--custom-config', type=str, default=None, 
						help='Fichier de configuration personnalisée pour le type "custom"')
	
	# Arguments pour la visualisation
	parser.add_argument('--visualize', action='store_true', 
						help='Visualiser la constellation')
	parser.add_argument('--save-figure', action='store_true', 
						help='Enregistrer une figure de la constellation')
	
	return parser.parse_args()

def generate_walker_constellation(num_planes, satellites_per_plane, altitude, inclination, eccentricity=0.0, phasing=0.0):
	"""
	Génère une constellation de type Walker (Delta).
	
	Args:
		num_planes: Nombre de plans orbitaux
		satellites_per_plane: Nombre de satellites par plan
		altitude: Altitude en km
		inclination: Inclinaison en degrés
		eccentricity: Excentricité
		phasing: Déphasage entre plans adjacents en degrés
		
	Returns:
		Liste d'éléments orbitaux pour tous les satellites
	"""
	orbital_elements = []
	
	# Convertir en radians
	inclination_rad = np.radians(inclination)
	phasing_rad = np.radians(phasing)
	
	for plane in range(num_planes):
		# Longitude du nœud ascendant pour ce plan
		raan = 2 * np.pi * plane / num_planes
		
		for sat in range(satellites_per_plane):
			# Anomalie vraie pour ce satellite
			true_anomaly = 2 * np.pi * sat / satellites_per_plane
			
			# Ajouter le déphasage
			phase_shift = plane * phasing_rad / satellites_per_plane
			true_anomaly = (true_anomaly + phase_shift) % (2 * np.pi)
			
			# Créer l'élément orbital
			element = OrbitalElements(
				semimajorAxis=6371.0 + altitude,  # Rayon Terre + altitude
				eccentricity=eccentricity,
				inclination=inclination_rad,
				longitudeOfAscendingNode=raan,
				argumentOfPeriapsis=0.0,  # Pour une orbite circulaire, peu importe
				trueAnomaly=true_anomaly
			)
			
			orbital_elements.append(element)
	
	return orbital_elements

def generate_polar_constellation(num_planes, satellites_per_plane, altitude, eccentricity=0.0):
	"""
	Génère une constellation de satellites en orbite polaire.
	
	Args:
		num_planes: Nombre de plans orbitaux
		satellites_per_plane: Nombre de satellites par plan
		altitude: Altitude en km
		eccentricity: Excentricité
		
	Returns:
		Liste d'éléments orbitaux pour tous les satellites
	"""
	# Utiliser l'inclination de 90 degrés (polaire)
	return generate_walker_constellation(num_planes, satellites_per_plane, altitude, 90.0, eccentricity)

def generate_equatorial_constellation(num_satellites, altitude, eccentricity=0.0):
	"""
	Génère une constellation de satellites en orbite équatoriale.
	
	Args:
		num_satellites: Nombre total de satellites
		altitude: Altitude en km
		eccentricity: Excentricité
		
	Returns:
		Liste d'éléments orbitaux pour tous les satellites
	"""
	orbital_elements = []
	
	for sat in range(num_satellites):
		# Anomalie vraie pour ce satellite
		true_anomaly = 2 * np.pi * sat / num_satellites
		
		# Créer l'élément orbital (inclinaison = 0 pour équatorial)
		element = OrbitalElements(
			semimajorAxis=6371.0 + altitude,  # Rayon Terre + altitude
			eccentricity=eccentricity,
			inclination=0.0,
			longitudeOfAscendingNode=0.0,  # N'importe quelle valeur pour une inclinaison de 0
			argumentOfPeriapsis=0.0,  # Pour une orbite circulaire, peu importe
			trueAnomaly=true_anomaly
		)
		
		orbital_elements.append(element)
	
	return orbital_elements

def generate_sun_synchronous_constellation(num_planes, satellites_per_plane, altitude, phase_hours=0.0):
	"""
	Génère une constellation de satellites en orbite héliosynchrone.
	
	Args:
		num_planes: Nombre de plans orbitaux
		satellites_per_plane: Nombre de satellites par plan
		altitude: Altitude en km
		phase_hours: Décalage de l'heure locale au nœud ascendant entre les plans
		
	Returns:
		Liste d'éléments orbitaux pour tous les satellites
	"""
	orbital_elements = []
	
	for plane in range(num_planes):
		# Heure locale au nœud ascendant pour ce plan
		local_time = (6.0 + plane * phase_hours) % 24.0  # Commencer à 6h (heure solaire)
		
		for sat in range(satellites_per_plane):
			# Anomalie vraie pour ce satellite
			true_anomaly = 2 * np.pi * sat / satellites_per_plane
			
			# Créer l'élément orbital
			element = OrbitalElements.createSunSynchronousOrbit(
				altitude=altitude,
				localTimeAtAscendingNode=local_time,
				argumentOfPeriapsis=0.0,
				trueAnomaly=true_anomaly
			)
			
			orbital_elements.append(element)
	
	return orbital_elements

def generate_custom_constellation(config_file):
	"""
	Génère une constellation de satellites à partir d'un fichier de configuration personnalisé.
	
	Args:
		config_file: Chemin vers le fichier de configuration
		
	Returns:
		Liste d'éléments orbitaux pour tous les satellites
	"""
	orbital_elements = []
	
	try:
		with open(config_file, 'r') as f:
			config = json.load(f)
		
		satellites = config.get("satellites", [])
		
		for sat_config in satellites:
			# Extraire les paramètres
			semi_major_axis = sat_config.get("semi_major_axis", 7171.0)  # km
			eccentricity = sat_config.get("eccentricity", 0.0)
			inclination = np.radians(sat_config.get("inclination", 0.0))  # convertir en radians
			raan = np.radians(sat_config.get("raan", 0.0))  # convertir en radians
			arg_perigee = np.radians(sat_config.get("arg_perigee", 0.0))  # convertir en radians
			true_anomaly = np.radians(sat_config.get("true_anomaly", 0.0))  # convertir en radians
			
			# Créer l'élément orbital
			element = OrbitalElements(
				semimajorAxis=semi_major_axis,
				eccentricity=eccentricity,
				inclination=inclination,
				longitudeOfAscendingNode=raan,
				argumentOfPeriapsis=arg_perigee,
				trueAnomaly=true_anomaly
			)
			
			orbital_elements.append(element)
			
	except Exception as e:
		print(f"Erreur lors de la lecture du fichier de configuration: {e}")
		return []
	
	return orbital_elements

def create_constellation_config(orbital_elements, constellation_type, args):
	"""
	Crée un fichier de configuration pour la constellation.
	
	Args:
		orbital_elements: Liste d'éléments orbitaux
		constellation_type: Type de constellation
		args: Arguments de ligne de commande
		
	Returns:
		Dictionnaire de configuration
	"""
	# Paramètres de base de la constellation
	constellation_config = {
		"type": constellation_type,
		"num_satellites": len(orbital_elements),
		"num_planes": args.num_planes,
		"satellites_per_plane": args.satellites_per_plane,
		"altitude": args.altitude,
		"inclination": args.inclination,
		"eccentricity": args.eccentricity,
		"satellites": []
	}
	
	# Ajouter les éléments orbitaux de chaque satellite
	for i, element in enumerate(orbital_elements):
		satellite_config = {
			"id": i,
			"orbital_elements": {
				"semi_major_axis": element.semimajorAxis,
				"eccentricity": element.eccentricity,
				"inclination_deg": np.degrees(element.inclination),
				"raan_deg": np.degrees(element.longitudeOfAscendingNode),
				"arg_perigee_deg": np.degrees(element.argumentOfPeriapsis),
				"true_anomaly_deg": np.degrees(element.trueAnomaly)
			}
		}
		
		constellation_config["satellites"].append(satellite_config)
	
	return constellation_config

def visualize_constellation(orbital_elements, earth_model, save_path=None):
	"""
	Visualise la constellation de satellites.
	
	Args:
		orbital_elements: Liste d'éléments orbitaux
		earth_model: Modèle de la Terre
		save_path: Chemin pour enregistrer la figure (optionnel)
	"""
	# Créer la figure
	fig = plt.figure(figsize=(12, 10))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_title('Visualisation de la constellation de satellites')
	
	# Dessiner la Terre
	earth_radius = earth_model.radius
	u = np.linspace(0, 2 * np.pi, 30)
	v = np.linspace(0, np.pi, 30)
	x = earth_radius * np.outer(np.cos(u), np.sin(v))
	y = earth_radius * np.outer(np.sin(u), np.sin(v))
	z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
	
	ax.plot_surface(x, y, z, color='blue', alpha=0.2)
	
	# Dessiner l'équateur
	theta = np.linspace(0, 2 * np.pi, 100)
	ax.plot(earth_radius * np.cos(theta), earth_radius * np.sin(theta), np.zeros(100), 'k-', alpha=0.3)
	
	# Utiliser le renderer d'orbites pour dessiner les orbites
	orbit_renderer = OrbitRenderer()
	
	# Créer des "satellites" temporaires pour le renderer
	satellites = []
	for i, element in enumerate(orbital_elements):
		# Créer un objet de type "satellite" simplifié pour le renderer
		satellite = type('MockSatellite', (), {})()
		satellite.satelliteId = i
		satellite.currentOrbitalElements = element
		
		# Calculer la position initiale
		position, velocity = element.toPosVel()
		
		# Ajouter les fonctions et attributs nécessaires
		satellite.getPosition = lambda pos=position: pos
		satellite.state = type('MockState', (), {})()
		satellite.state.batteryLevel = 1.0
		satellite.isEclipsed = False
		
		satellites.append(satellite)
	
	# Dessiner les orbites
	orbit_renderer.renderOrbits(ax, satellites, numPoints=100, showLabels=True)
	
	# Configurer les axes
	ax.set_xlabel('X (km)')
	ax.set_ylabel('Y (km)')
	ax.set_zlabel('Z (km)')
	
	# Égaliser les échelles des axes
	max_range = max([
		np.max(np.abs([sat.getPosition()[0] for sat in satellites])),
		np.max(np.abs([sat.getPosition()[1] for sat in satellites])),
		np.max(np.abs([sat.getPosition()[2] for sat in satellites]))
	])
	max_range = max(max_range, earth_radius) * 1.1
	
	ax.set_xlim(-max_range, max_range)
	ax.set_ylim(-max_range, max_range)
	ax.set_zlim(-max_range, max_range)
	
	# Définir une vue initiale
	ax.view_init(elev=30, azim=45)
	
	# Ajuster la mise en page
	plt.tight_layout()
	
	# Enregistrer ou afficher
	if save_path:
		plt.savefig(save_path, dpi=150)
		print(f"Visualisation enregistrée dans {save_path}")
	else:
		plt.show()

def main(args):
	"""
	Fonction principale.
	
	Args:
		args: Arguments de ligne de commande
	"""
	# Créer les répertoires nécessaires
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Déterminer le nom de la constellation
	constellation_name = args.name
	if constellation_name is None:
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		constellation_name = f"{args.type}_constellation_{timestamp}"
	
	# Initialiser le modèle de la Terre pour la visualisation
	earth_model = EarthModel()
	
	# Générer la constellation en fonction du type
	orbital_elements = []
	
	if args.type == 'walker':
		orbital_elements = generate_walker_constellation(
			args.num_planes, args.satellites_per_plane, args.altitude, args.inclination, args.eccentricity, args.phasing
		)
		
	elif args.type == 'polar':
		orbital_elements = generate_polar_constellation(
			args.num_planes, args.satellites_per_plane, args.altitude, args.eccentricity
		)
		
	elif args.type == 'equatorial':
		total_satellites = args.num_planes * args.satellites_per_plane
		orbital_elements = generate_equatorial_constellation(
			total_satellites, args.altitude, args.eccentricity
		)
		
	elif args.type == 'sun-sync':
		phase_hours = 24.0 / args.num_planes if args.num_planes > 0 else 0.0
		orbital_elements = generate_sun_synchronous_constellation(
			args.num_planes, args.satellites_per_plane, args.altitude, phase_hours
		)
		
	elif args.type == 'custom':
		if args.custom_config:
			orbital_elements = generate_custom_constellation(args.custom_config)
		else:
			print("Erreur: le type 'custom' nécessite un fichier de configuration (--custom-config)")
			return
	
	# Vérifier que la génération a réussi
	if not orbital_elements:
		print("Erreur: la génération de la constellation a échoué")
		return
	
	# Créer le fichier de configuration
	constellation_config = create_constellation_config(orbital_elements, args.type, args)
	
	# Enregistrer la configuration
	config_file = os.path.join(args.output_dir, f"{constellation_name}.json")
	with open(config_file, 'w') as f:
		json.dump(constellation_config, f, indent=4)
	
	print(f"Configuration de la constellation enregistrée dans {config_file}")
	print(f"Nombre de satellites: {len(orbital_elements)}")
	
	# Visualiser si demandé
	if args.visualize or args.save_figure:
		figure_path = None
		if args.save_figure:
			figure_path = os.path.join(args.output_dir, f"{constellation_name}.png")
		
		visualize_constellation(orbital_elements, earth_model, figure_path)

if __name__ == "__main__":
	args = parse_arguments()
	main(args)