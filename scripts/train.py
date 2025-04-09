#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
from datetime import datetime
import json

from environment.SatelliteEnv import SatelliteEnv
from models.PPOAgent import PPOAgent
from training.TrainingManager import TrainingManager
from training.Hyperparameters import HyperParameters
from training.MultiProcessTrainer import MultiProcessTrainer

def parse_arguments():
	"""
	Parse les arguments de ligne de commande.
	
	Returns:
		Arguments parsés
	"""
	parser = argparse.ArgumentParser(description='Entraînement de l\'agent DeepSatNet')
	
	# Arguments généraux
	parser.add_argument('--config', type=str, default='configs/default_config.json', 
						help='Chemin vers le fichier de configuration')
	parser.add_argument('--output-dir', type=str, default='results', 
						help='Répertoire pour enregistrer les résultats')
	parser.add_argument('--log-dir', type=str, default='logs', 
						help='Répertoire pour les journaux')
	parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', 
						help='Répertoire pour les points de contrôle')
	parser.add_argument('--seed', type=int, default=None, 
						help='Graine aléatoire pour la reproductibilité')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
						help='Dispositif d\'exécution')
	
	# Arguments spécifiques à l'entraînement
	parser.add_argument('--total-timesteps', type=int, default=1000000, 
						help='Nombre total d\'étapes de temps pour l\'entraînement')
	parser.add_argument('--num-envs', type=int, default=1, 
						help='Nombre d\'environnements parallèles')
	parser.add_argument('--eval-freq', type=int, default=10000, 
						help='Fréquence d\'évaluation (en étapes de temps)')
	parser.add_argument('--eval-episodes', type=int, default=5, 
						help='Nombre d\'épisodes pour l\'évaluation')
	parser.add_argument('--save-freq', type=int, default=10000, 
						help='Fréquence de sauvegarde des points de contrôle')
	parser.add_argument('--log-freq', type=int, default=1000, 
						help='Fréquence de journalisation')
	parser.add_argument('--render', action='store_true', 
						help='Activer le rendu pendant l\'entraînement')
	parser.add_argument('--resume', type=str, default=None, 
						help='Chemin vers le checkpoint pour reprendre l\'entraînement')
	parser.add_argument('--hyperparams', type=str, default=None, 
						help='Chemin vers un fichier d\'hyperparamètres personnalisé')
	
	return parser.parse_args()

def train(args):
	"""
	Entraîne l'agent DeepSatNet.
	
	Args:
		args: Arguments de ligne de commande
	"""
	# Créer les répertoires nécessaires
	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(args.log_dir, exist_ok=True)
	os.makedirs(args.checkpoint_dir, exist_ok=True)
	
	# Timestamp pour cette exécution
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Définir un chemin pour les résultats
	run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
	os.makedirs(run_dir, exist_ok=True)
	
	# Charger les hyperparamètres
	if args.hyperparams:
		hyperparams = HyperParameters.fromFile(args.hyperparams)
	else:
		# Essayer de charger depuis le fichier de configuration
		try:
			with open(args.config, 'r') as f:
				config = json.load(f)
				if "hyperparameters" in config:
					hyperparams = HyperParameters.fromDict(config["hyperparameters"])
				else:
					hyperparams = HyperParameters()
		except:
			hyperparams = HyperParameters()
	
	# Enregistrer les hyperparamètres utilisés
	hyperparams_file = os.path.join(run_dir, "hyperparameters.json")
	hyperparams.save(hyperparams_file)
	
	# Créer un environnement d'entraînement unique pour déterminer les dimensions
	env = SatelliteEnv(
		configPath=args.config,
		renderMode="none"
	)
	
	# Définir la graine aléatoire si spécifiée
	if args.seed is not None:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		env.seed(args.seed)
	
	# Créer l'agent
	agent = PPOAgent(
		observationSpace=env.observation_space,
		actionSpace=env.action_space,
		deviceName=args.device,
		satelliteFeaturesDim=hyperparams.satelliteFeaturesDim,
		globalFeaturesDim=hyperparams.globalFeaturesDim,
		hiddenDims=hyperparams.hiddenDims,
		learningRate=hyperparams.learningRate,
		batchSize=hyperparams.batchSize,
		numEpochs=hyperparams.numEpochs,
		clipRange=hyperparams.clipRange,
		valueLossCoef=hyperparams.valueLossCoef,
		entropyCoef=hyperparams.entropyCoef,
		maxGradNorm=hyperparams.maxGradNorm,
		targetKL=hyperparams.targetKL,
		gamma=hyperparams.gamma,
		lambd=hyperparams.lambd,
		bufferSize=hyperparams.bufferSize
	)
	
	# Charger le checkpoint pour reprendre l'entraînement si spécifié
	if args.resume:
		agent.load(args.resume)
		print(f"Entraînement repris à partir de {args.resume}")
	
	# Fermer l'environnement de référence
	env.close()
	
	if args.num_envs > 1:
		# Utiliser l'entraînement multi-processus
		trainer = MultiProcessTrainer(
			configPath=args.config,
			numEnvs=args.num_envs,
			logDir=args.log_dir,
			checkpointDir=args.checkpoint_dir,
			seed=args.seed,
			deviceName=args.device,
			hyperparams=hyperparams
		)
		
		# Commencer l'entraînement
		trainer.train(
			agent=agent,
			totalTimeSteps=args.total_timesteps,
			evalFreq=args.eval_freq,
			evalEpisodes=args.eval_episodes,
			saveFreq=args.save_freq,
			logFreq=args.log_freq
		)
	else:
		# Utiliser l'entraînement simple processus
		training_manager = TrainingManager(
			configPath=args.config,
			logDir=args.log_dir,
			checkpointDir=args.checkpoint_dir,
			render=args.render,
			renderEval=True,
			seed=args.seed,
			deviceName=args.device
		)
		
		# Si un agent est déjà chargé, l'utiliser
		if args.resume:
			training_manager.agent = agent
		
		# Commencer l'entraînement
		training_manager.train(
			totalTimeSteps=args.total_timesteps,
			evalFreq=args.eval_freq,
			logFreq=args.log_freq,
			saveFreq=args.save_freq
		)

if __name__ == "__main__":
	args = parse_arguments()
	train(args)