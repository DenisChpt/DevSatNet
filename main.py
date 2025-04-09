import os
import argparse
import time
from typing import Dict, Any

from training.TrainingManager import TrainingManager

def parseArguments():
	"""
	Parse les arguments de ligne de commande.
	
	Returns:
		Arguments parsés
	"""
	parser = argparse.ArgumentParser(description='DeepSatNet - Apprentissage par renforcement pour les constellations de satellites')
	
	# Arguments généraux
	parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visualize'], 
						help='Mode d\'exécution (train, test, visualize)')
	parser.add_argument('--config', type=str, default='configs/default_config.json', 
						help='Chemin vers le fichier de configuration')
	parser.add_argument('--log-dir', type=str, default='logs', 
						help='Répertoire pour les journaux')
	parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
						help='Répertoire pour les points de contrôle')
	parser.add_argument('--seed', type=int, default=None, 
						help='Graine aléatoire pour la reproductibilité')
	parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
						help='Dispositif d\'exécution')
	
	# Arguments spécifiques à l'entraînement
	parser.add_argument('--total-timesteps', type=int, default=1000000, 
						help='Nombre total d\'étapes de temps pour l\'entraînement')
	parser.add_argument('--eval-freq', type=int, default=10000, 
						help='Fréquence d\'évaluation (en étapes de temps)')
	parser.add_argument('--log-freq', type=int, default=1000, 
						help='Fréquence de journalisation (en étapes de temps)')
	parser.add_argument('--save-freq', type=int, default=10000, 
						help='Fréquence de sauvegarde des points de contrôle (en étapes de temps)')
	parser.add_argument('--render', action='store_true', 
						help='Activer le rendu pendant l\'entraînement')
	
	# Arguments spécifiques au test
	parser.add_argument('--checkpoint', type=str, default='best', 
						help='Chemin ou mot-clé (latest, best) du point de contrôle à charger')
	parser.add_argument('--num-episodes', type=int, default=10, 
						help='Nombre d\'épisodes pour le test')
	parser.add_argument('--record-video', action='store_true', 
						help='Enregistrer des vidéos pendant le test')
	
	return parser.parse_args()

def train(args):
	"""
	Exécute l'entraînement.
	
	Args:
		args: Arguments de ligne de commande
	"""
	print(f"Démarrage de l'entraînement avec configuration: {args.config}")
	
	# Créer le gestionnaire d'entraînement
	trainingManager = TrainingManager(
		configPath=args.config,
		logDir=args.log_dir,
		checkpointDir=args.checkpoint_dir,
		render=args.render,
		renderEval=True,  # Toujours activer le rendu pendant l'évaluation
		seed=args.seed,
		deviceName=args.device
	)
	
	# Lancer l'entraînement
	trainingManager.train(
		totalTimeSteps=args.total_timesteps,
		evalFreq=args.eval_freq,
		logFreq=args.log_freq,
		saveFreq=args.save_freq
	)
	
	print("Entraînement terminé!")

def test(args):
	"""
	Exécute le test d'un agent entraîné.
	
	Args:
		args: Arguments de ligne de commande
	"""
	print(f"Démarrage du test avec configuration: {args.config}")
	print(f"Chargement du checkpoint: {args.checkpoint}")
	
	# Créer le gestionnaire d'entraînement
	trainingManager = TrainingManager(
		configPath=args.config,
		logDir=args.log_dir,
		checkpointDir=args.checkpoint_dir,
		render=False,
		renderEval=True,  # Toujours activer le rendu pour le test
		seed=args.seed,
		deviceName=args.device
	)
	
	# Charger l'agent
	trainingManager.loadAgent(args.checkpoint)
	
	# Exécuter le test
	testStats = trainingManager.testAgent(
		numEpisodes=args.num_episodes,
		recordVideo=args.record_video
	)
	
	print(f"Test terminé avec résultats: {testStats}")

def visualize(args):
	"""
	Exécute la visualisation de la constellation.
	
	Args:
		args: Arguments de ligne de commande
	"""
	print(f"Démarrage de la visualisation avec configuration: {args.config}")
	
	# Importer les modules nécessaires
	from environment.SatelliteEnv import SatelliteEnv
	from visualization.SimulationVisualizer import SimulationVisualizer
	
	# Créer l'environnement avec rendu 3D
	env = SatelliteEnv(
		configPath=args.config,
		renderMode="3d"
	)
	
	# Réinitialiser l'environnement
	env.reset()
	
	# Exécuter la visualisation en temps réel
	print("Appuyez sur Ctrl+C pour quitter la visualisation")
	
	try:
		while True:
			# Actions aléatoires pour visualiser le mouvement
			action = env.action_space.sample()
			
			# Si un checkpoint est spécifié, charger l'agent et utiliser ses actions
			if args.checkpoint != 'random':
				# Créer le gestionnaire d'entraînement
				trainingManager = TrainingManager(
					configPath=args.config,
					logDir=args.log_dir,
					checkpointDir=args.checkpoint_dir,
					render=False,
					renderEval=False,
					seed=args.seed,
					deviceName=args.device
				)
				
				# Charger l'agent
				trainingManager.loadAgent(args.checkpoint)
				
				# Utiliser l'agent pour sélectionner des actions
				observation = env.reset()
				action, _ = trainingManager.agent.selectAction(observation, deterministic=True)
			
			# Exécuter l'action dans l'environnement
			observation, reward, done, info = env.step(action)
			
			# Rendre la visualisation
			env.render()
			
			# Attendre un court moment pour une visualisation plus lente
			time.sleep(0.1)
			
			if done:
				observation = env.reset()
				
	except KeyboardInterrupt:
		print("Visualisation terminée par l'utilisateur")
	
	# Fermer l'environnement
	env.close()

def main():
	"""
	Point d'entrée principal du programme.
	"""
	# Parser les arguments
	args = parseArguments()
	
	# Créer les répertoires nécessaires s'ils n'existent pas
	os.makedirs(args.log_dir, exist_ok=True)
	os.makedirs(args.checkpoint_dir, exist_ok=True)
	
	# Exécuter le mode correspondant
	if args.mode == 'train':
		train(args)
	elif args.mode == 'test':
		test(args)
	elif args.mode == 'visualize':
		visualize(args)
	else:
		print(f"Mode non reconnu: {args.mode}")

if __name__ == "__main__":
	main()