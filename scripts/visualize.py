#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime

from environment.SatelliteEnv import SatelliteEnv
from models.PPOAgent import PPOAgent
from visualization.SimulationVisualizer import SimulationVisualizer
from visualization.OrbitRenderer import OrbitRenderer

def parse_arguments():
	"""
	Parse les arguments de ligne de commande.
	
	Returns:
		Arguments parsés
	"""
	parser = argparse.ArgumentParser(description='Visualisation de la constellation de satellites DeepSatNet')
	
	# Arguments généraux
	parser.add_argument('--config', type=str, default='configs/default_config.json', 
						help='Chemin vers le fichier de configuration')
	parser.add_argument('--checkpoint', type=str, default=None, 
						help='Chemin vers le checkpoint de l\'agent (si None, actions aléatoires)')
	parser.add_argument('--output-dir', type=str, default='visualization_results', 
						help='Répertoire pour enregistrer les résultats')
	parser.add_argument('--seed', type=int, default=None, 
						help='Graine aléatoire pour la reproductibilité')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
						help='Dispositif d\'exécution')
	
	# Arguments spécifiques à la visualisation
	parser.add_argument('--mode', type=str, default='3d', choices=['2d', '3d', 'both'], 
						help='Mode de visualisation')
	parser.add_argument('--duration', type=float, default=300.0, 
						help='Durée de la simulation en secondes')
	parser.add_argument('--speed', type=float, default=1.0, 
						help='Vitesse de simulation (1.0 = temps réel)')
	parser.add_argument('--save-frames', action='store_true', 
						help='Enregistrer les images de chaque étape')
	parser.add_argument('--frame-interval', type=float, default=5.0, 
						help='Intervalle entre les images enregistrées (en secondes)')
	parser.add_argument('--create-video', action='store_true', 
						help='Créer une vidéo à partir des images enregistrées')
	parser.add_argument('--show-labels', action='store_true', 
						help='Afficher les étiquettes des satellites')
	parser.add_argument('--focus-satellite', type=int, default=None, 
						help='ID du satellite à suivre en particulier')
	
	return parser.parse_args()

def visualize(args):
	"""
	Visualise la constellation de satellites DeepSatNet.
	
	Args:
		args: Arguments de ligne de commande
	"""
	# Créer les répertoires nécessaires
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Timestamp pour cette exécution
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Définir un chemin pour les résultats
	viz_dir = os.path.join(args.output_dir, f"viz_{timestamp}")
	os.makedirs(viz_dir, exist_ok=True)
	
	if args.save_frames:
		frames_dir = os.path.join(viz_dir, "frames")
		os.makedirs(frames_dir, exist_ok=True)
	
	# Créer l'environnement
	env = SatelliteEnv(
		configPath=args.config,
		renderMode=args.mode if args.mode != 'both' else '3d'
	)
	
	# Définir la graine aléatoire si spécifiée
	if args.seed is not None:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		env.seed(args.seed)
	
	# Créer l'agent si un checkpoint est spécifié
	agent = None
	if args.checkpoint:
		agent = PPOAgent(
			observationSpace=env.observation_space,
			actionSpace=env.action_space,
			deviceName=args.device
		)
		
		# Charger le checkpoint
		agent.load(args.checkpoint)
		print(f"Agent chargé à partir de {args.checkpoint}")
	
	# Créer le visualiseur
	visualizer = SimulationVisualizer(mode=args.mode)
	
	# Initialiser l'environnement
	observation = env.reset()
	
	# Initialiser le visualiseur
	visualizer.reset(
		satellites=env.constellation.satellites,
		groundStations=env.groundStations,
		earthModel=env.earthModel
	)
	
	# Créer un renderer d'orbites pour des visualisations détaillées
	orbit_renderer = OrbitRenderer()
	
	# Préparer pour la simulation
	simulationTime = 0.0
	realStartTime = time.time()
	lastFrameTime = 0.0
	frame_count = 0
	
	# Si on focalise sur un satellite spécifique, préparer des données pour suivre son évolution
	if args.focus_satellite is not None:
		focus_data = {
			"time": [],
			"altitude": [],
			"battery": [],
			"solar_output": [],
			"temperature": [],
			"is_eclipsed": []
		}
	
	print(f"Début de la visualisation. Appuyez sur Ctrl+C pour arrêter...")
	
	try:
		while simulationTime < args.duration:
			# Calculer le temps écoulé depuis le début de la simulation
			currentRealTime = time.time()
			elapsedRealTime = currentRealTime - realStartTime
			
			# Calculer le temps de simulation correspondant à la vitesse spécifiée
			targetSimTime = elapsedRealTime * args.speed
			
			# Avancer la simulation jusqu'au temps cible
			while simulationTime < targetSimTime and simulationTime < args.duration:
				# Sélectionner une action
				if agent:
					action, _ = agent.selectAction(observation, deterministic=True)
				else:
					action = env.action_space.sample()
				
				# Exécuter l'action dans l'environnement
				next_observation, reward, done, info = env.step(action)
				
				# Mettre à jour l'observation
				observation = next_observation
				
				# Mettre à jour le temps de simulation
				simulationTime = env.simulationTime
				
				# Si on focalise sur un satellite spécifique, collecter ses données
				if args.focus_satellite is not None:
					for sat in env.constellation.satellites:
						if sat.satelliteId == args.focus_satellite:
							focus_data["time"].append(simulationTime)
							focus_data["altitude"].append(np.linalg.norm(sat.getPosition()) - env.earthModel.radius)
							focus_data["battery"].append(sat.state.batteryLevel)
							focus_data["solar_output"].append(sat.state.solarPanelOutput)
							focus_data["temperature"].append(sat.state.temperature)
							focus_data["is_eclipsed"].append(1.0 if sat.isEclipsed else 0.0)
							break
				
				# Si l'épisode est terminé, réinitialiser l'environnement
				if done:
					observation = env.reset()
				
				# Enregistrer une image si nécessaire
				if args.save_frames and (simulationTime - lastFrameTime >= args.frame_interval or frame_count == 0):
					lastFrameTime = simulationTime
					frame_count += 1
					
					# Mettre à jour le visualiseur
					visualizer.update(
						satellites=env.constellation.satellites,
						groundStations=env.groundStations,
						metrics=info.get("metrics", {}),
						time=simulationTime
					)
					
					# Rendre et enregistrer l'image
					visualizer.render(mode='human')
					frame_file = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
					visualizer.saveFrame(frame_file)
					
					# Si on focalise sur un satellite, générer aussi une vue détaillée
					if args.focus_satellite is not None:
						for sat in env.constellation.satellites:
							if sat.satelliteId == args.focus_satellite:
								# Générer un diagramme d'orbite détaillé
								orbit_fig = orbit_renderer.generateOrbitDiagram(sat)
								orbit_file = os.path.join(frames_dir, f"orbit_{frame_count:04d}.png")
								orbit_fig.savefig(orbit_file, dpi=150)
								plt.close(orbit_fig)
								break
			
			# Mettre à jour le visualiseur
			visualizer.update(
				satellites=env.constellation.satellites,
				groundStations=env.groundStations,
				metrics=info.get("metrics", {}),
				time=simulationTime
			)
			
			# Rendre la visualisation
			visualizer.render(mode='human')
			
			# Petit délai pour éviter de saturer le CPU
			time.sleep(0.01)
			
	except KeyboardInterrupt:
		print("Visualisation interrompue par l'utilisateur.")
	
	# Fermer l'environnement et le visualiseur
	env.close()
	visualizer.close()
	
	# Si on a enregistré des images et qu'on demande une vidéo
	if args.save_frames and args.create_video:
		try:
			import cv2
			
			# Trouver le premier frame pour déterminer la taille
			first_frame = os.path.join(frames_dir, "frame_0001.png")
			if os.path.exists(first_frame):
				img = cv2.imread(first_frame)
				height, width, _ = img.shape
				
				# Créer la vidéo
				video_file = os.path.join(viz_dir, "simulation.mp4")
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				fps = 30
				video = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
				
				# Ajouter tous les frames
				frame_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
				for frame_file in frame_files:
					img = cv2.imread(os.path.join(frames_dir, frame_file))
					video.write(img)
				
				video.release()
				print(f"Vidéo créée: {video_file}")
			else:
				print("Impossible de créer la vidéo: aucun frame trouvé.")
		except ImportError:
			print("Impossible de créer la vidéo: OpenCV non installé.")
	
	# Si on a focalisé sur un satellite, générer des graphiques de ses données
	if args.focus_satellite is not None and focus_data["time"]:
		plots_dir = os.path.join(viz_dir, "plots")
		os.makedirs(plots_dir, exist_ok=True)
		
		# Tracer l'altitude
		plt.figure(figsize=(10, 6))
		plt.plot(focus_data["time"], focus_data["altitude"])
		plt.xlabel('Temps de simulation (s)')
		plt.ylabel('Altitude (km)')
		plt.title(f'Altitude du satellite #{args.focus_satellite}')
		plt.grid(True, alpha=0.3)
		plt.savefig(os.path.join(plots_dir, "altitude.png"), dpi=150)
		
		# Tracer le niveau de batterie
		plt.figure(figsize=(10, 6))
		plt.plot(focus_data["time"], focus_data["battery"])
		plt.xlabel('Temps de simulation (s)')
		plt.ylabel('Niveau de batterie (0-1)')
		plt.title(f'Niveau de batterie du satellite #{args.focus_satellite}')
		plt.grid(True, alpha=0.3)
		plt.savefig(os.path.join(plots_dir, "battery.png"), dpi=150)
		
		# Tracer la production solaire et les périodes d'éclipse
		plt.figure(figsize=(10, 6))
		plt.plot(focus_data["time"], focus_data["solar_output"], label='Production solaire (W)')
		
		# Ajouter les zones d'éclipse
		for i in range(len(focus_data["time"])):
			if focus_data["is_eclipsed"][i] > 0:
				plt.axvspan(focus_data["time"][i], focus_data["time"][i] + 1, alpha=0.2, color='gray')
		
		plt.xlabel('Temps de simulation (s)')
		plt.ylabel('Puissance (W)')
		plt.title(f'Production solaire du satellite #{args.focus_satellite}')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.savefig(os.path.join(plots_dir, "solar_output.png"), dpi=150)
		
		# Tracer la température
		plt.figure(figsize=(10, 6))
		plt.plot(focus_data["time"], focus_data["temperature"])
		plt.xlabel('Temps de simulation (s)')
		plt.ylabel('Température (K)')
		plt.title(f'Température du satellite #{args.focus_satellite}')
		plt.grid(True, alpha=0.3)
		plt.savefig(os.path.join(plots_dir, "temperature.png"), dpi=150)
		
		plt.close('all')
	
	print(f"Visualisation terminée. Résultats enregistrés dans {viz_dir}")

if __name__ == "__main__":
	args = parse_arguments()
	visualize(args)