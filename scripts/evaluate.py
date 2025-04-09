#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from environment.SatelliteEnv import SatelliteEnv
from models.PPOAgent import PPOAgent
from training.TrainingManager import TrainingManager
from training.Hyperparameters import HyperParameters

def parse_arguments():
	"""
	Parse les arguments de ligne de commande.
	
	Returns:
		Arguments parsés
	"""
	parser = argparse.ArgumentParser(description='Évaluation de l\'agent DeepSatNet')
	
	# Arguments généraux
	parser.add_argument('--config', type=str, default='configs/default_config.json', 
						help='Chemin vers le fichier de configuration')
	parser.add_argument('--checkpoint', type=str, required=True, 
						help='Chemin vers le checkpoint de l\'agent')
	parser.add_argument('--output-dir', type=str, default='evaluation_results', 
						help='Répertoire pour enregistrer les résultats')
	parser.add_argument('--seed', type=int, default=None, 
						help='Graine aléatoire pour la reproductibilité')
	parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
						help='Dispositif d\'exécution')
	
	# Arguments spécifiques à l'évaluation
	parser.add_argument('--num-episodes', type=int, default=10, 
						help='Nombre d\'épisodes pour l\'évaluation')
	parser.add_argument('--render', action='store_true', 
						help='Activer le rendu pendant l\'évaluation')
	parser.add_argument('--record-video', action='store_true', 
						help='Enregistrer des vidéos de l\'évaluation')
	parser.add_argument('--plot-metrics', action='store_true', 
						help='Générer des graphiques des métriques')
	parser.add_argument('--save-trajectory', action='store_true', 
						help='Enregistrer les trajectoires des satellites')
	parser.add_argument('--compare-baseline', action='store_true', 
						help='Comparer avec une stratégie de référence')
	
	return parser.parse_args()

def evaluate(args):
	"""
	Évalue l'agent DeepSatNet.
	
	Args:
		args: Arguments de ligne de commande
	"""
	# Créer les répertoires nécessaires
	os.makedirs(args.output_dir, exist_ok=True)
	
	# Timestamp pour cette exécution
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	
	# Définir un chemin pour les résultats
	eval_dir = os.path.join(args.output_dir, f"eval_{timestamp}")
	os.makedirs(eval_dir, exist_ok=True)
	
	if args.record_video:
		video_dir = os.path.join(eval_dir, "videos")
		os.makedirs(video_dir, exist_ok=True)
	
	# Créer l'environnement
	env = SatelliteEnv(
		configPath=args.config,
		renderMode="3d" if args.render else "none"
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
		deviceName=args.device
	)
	
	# Charger le checkpoint
	agent.load(args.checkpoint)
	print(f"Agent chargé à partir de {args.checkpoint}")
	
	# Configuration pour l'enregistrement vidéo
	if args.record_video:
		import gym
		from gym.wrappers import RecordVideo
		
		env = RecordVideo(
			env,
			video_dir,
			episode_trigger=lambda x: True,
			name_prefix="satellite_constellation"
		)
	
	# Exécuter l'évaluation
	total_rewards = []
	episode_lengths = []
	all_metrics = []
	trajectories = []
	
	for episode in range(args.num_episodes):
		observation = env.reset()
		episode_reward = 0
		episode_length = 0
		episode_metrics = []
		episode_trajectory = []
		
		done = False
		while not done:
			# Sélectionner une action (mode déterministe)
			action, _ = agent.selectAction(observation, deterministic=True)
			
			# Exécuter l'action dans l'environnement
			next_observation, reward, done, info = env.step(action)
			
			# Mettre à jour l'observation
			observation = next_observation
			
			# Collecter les données
			episode_reward += reward
			episode_length += 1
			episode_metrics.append(info.get("metrics", {}))
			
			# Enregistrer la trajectoire si demandé
			if args.save_trajectory:
				satellites_positions = []
				for i, satellite in enumerate(env.constellation.satellites):
					pos = satellite.getPosition().tolist()
					state = {
						"position": pos,
						"batteryLevel": satellite.state.batteryLevel,
						"isEclipsed": satellite.isEclipsed
					}
					satellites_positions.append(state)
				
				trajectory_step = {
					"time": env.simulationTime,
					"satellites": satellites_positions,
					"metrics": info.get("metrics", {})
				}
				episode_trajectory.append(trajectory_step)
			
			# Afficher la progression
			if episode_length % 100 == 0:
				print(f"Épisode {episode+1}/{args.num_episodes}, Étape {episode_length}, Récompense cumulée: {episode_reward:.2f}")
		
		# Enregistrer les métriques de l'épisode
		total_rewards.append(episode_reward)
		episode_lengths.append(episode_length)
		
		# Calculer les statistiques des métriques
		episode_metric_stats = {}
		for metric_name in ["coverage", "data_throughput", "energy_efficiency", "network_resilience", "user_satisfaction"]:
			values = [m.get(metric_name, 0.0) for m in episode_metrics if metric_name in m]
			if values:
				episode_metric_stats[metric_name] = {
					"mean": float(np.mean(values)),
					"min": float(np.min(values)),
					"max": float(np.max(values)),
					"std": float(np.std(values))
				}
		
		all_metrics.append(episode_metric_stats)
		
		if args.save_trajectory:
			trajectories.append(episode_trajectory)
		
		print(f"Épisode {episode+1} terminé, Récompense: {episode_reward:.2f}, Longueur: {episode_length}")
	
	# Fermer l'environnement
	env.close()
	
	# Calculer les statistiques globales
	mean_reward = np.mean(total_rewards)
	std_reward = np.std(total_rewards)
	mean_length = np.mean(episode_lengths)
	
	# Agréger les métriques de tous les épisodes
	aggregated_metrics = {}
	for metric_name in ["coverage", "data_throughput", "energy_efficiency", "network_resilience", "user_satisfaction"]:
		values = [m[metric_name]["mean"] for m in all_metrics if metric_name in m]
		if values:
			aggregated_metrics[metric_name] = {
				"mean": float(np.mean(values)),
				"min": float(np.min(values)),
				"max": float(np.max(values)),
				"std": float(np.std(values))
			}
	
	# Enregistrer les résultats
	results = {
		"config": args.config,
		"checkpoint": args.checkpoint,
		"num_episodes": args.num_episodes,
		"seed": args.seed,
		"timestamp": timestamp,
		"mean_reward": float(mean_reward),
		"std_reward": float(std_reward),
		"mean_episode_length": float(mean_length),
		"rewards_per_episode": [float(r) for r in total_rewards],
		"lengths_per_episode": [int(l) for l in episode_lengths],
		"metrics": aggregated_metrics
	}
	
	# Enregistrer au format JSON
	results_file = os.path.join(eval_dir, "results.json")
	with open(results_file, 'w') as f:
		json.dump(results, f, indent=4)
	
	# Enregistrer les trajectoires si demandé
	if args.save_trajectory:
		trajectory_file = os.path.join(eval_dir, "trajectories.json")
		with open(trajectory_file, 'w') as f:
			json.dump(trajectories, f, indent=4)
	
	# Générer des graphiques si demandé
	if args.plot_metrics:
		plot_dir = os.path.join(eval_dir, "plots")
		os.makedirs(plot_dir, exist_ok=True)
		
		# Tracer les récompenses par épisode
		plt.figure(figsize=(10, 6))
		plt.plot(range(1, args.num_episodes + 1), total_rewards, 'b-', marker='o')
		plt.axhline(y=mean_reward, color='r', linestyle='--', label=f'Moyenne: {mean_reward:.2f}')
		plt.fill_between(
			range(1, args.num_episodes + 1),
			mean_reward - std_reward,
			mean_reward + std_reward,
			color='r', alpha=0.2
		)
		plt.xlabel('Épisode')
		plt.ylabel('Récompense totale')
		plt.title('Récompenses par épisode')
		plt.grid(True, alpha=0.3)
		plt.legend()
		plt.savefig(os.path.join(plot_dir, "rewards.png"), dpi=150)
		
		# Tracer les métriques clés
		for metric_name, values in aggregated_metrics.items():
			plt.figure(figsize=(10, 6))
			
			# Extraire les valeurs par épisode
			episode_values = [m[metric_name]["mean"] for m in all_metrics if metric_name in m]
			
			plt.plot(range(1, len(episode_values) + 1), episode_values, 'g-', marker='o')
			plt.axhline(y=values["mean"], color='r', linestyle='--', label=f'Moyenne: {values["mean"]:.2f}')
			plt.fill_between(
				range(1, len(episode_values) + 1),
				values["mean"] - values["std"],
				values["mean"] + values["std"],
				color='r', alpha=0.2
			)
			plt.xlabel('Épisode')
			plt.ylabel(f'{metric_name.replace("_", " ").title()}')
			plt.title(f'{metric_name.replace("_", " ").title()} par épisode')
			plt.grid(True, alpha=0.3)
			plt.legend()
			plt.savefig(os.path.join(plot_dir, f"{metric_name}.png"), dpi=150)
		
		plt.close('all')
	
	# Comparer avec la baseline si demandé
	if args.compare_baseline:
		# Créer un nouvel environnement pour la baseline
		baseline_env = SatelliteEnv(
			configPath=args.config,
			renderMode="none"
		)
		
		if args.seed is not None:
			baseline_env.seed(args.seed + 1000)  # Seed différent pour éviter la corrélation
		
		# Exécuter l'évaluation avec la stratégie de référence
		baseline_rewards = []
		baseline_metrics = []
		
		for episode in range(args.num_episodes):
			observation = baseline_env.reset()
			episode_reward = 0
			episode_metrics = []
			
			done = False
			while not done:
				# Stratégie de référence: actions aléatoires
				action = baseline_env.action_space.sample()
				
				# Exécuter l'action dans l'environnement
				next_observation, reward, done, info = baseline_env.step(action)
				
				# Mettre à jour l'observation
				observation = next_observation
				
				# Collecter les données
				episode_reward += reward
				episode_metrics.append(info.get("metrics", {}))
			
			# Enregistrer les métriques de l'épisode
			baseline_rewards.append(episode_reward)
			
			# Calculer les statistiques des métriques
			episode_metric_stats = {}
			for metric_name in ["coverage", "data_throughput", "energy_efficiency", "network_resilience", "user_satisfaction"]:
				values = [m.get(metric_name, 0.0) for m in episode_metrics if metric_name in m]
				if values:
					episode_metric_stats[metric_name] = {
						"mean": float(np.mean(values)),
						"min": float(np.min(values)),
						"max": float(np.max(values)),
						"std": float(np.std(values))
					}
			
			baseline_metrics.append(episode_metric_stats)
		
		baseline_env.close()
		
		# Calculer les statistiques de la baseline
		baseline_mean_reward = np.mean(baseline_rewards)
		baseline_std_reward = np.std(baseline_rewards)
		
		# Agréger les métriques de la baseline
		baseline_aggregated_metrics = {}
		for metric_name in ["coverage", "data_throughput", "energy_efficiency", "network_resilience", "user_satisfaction"]:
			values = [m[metric_name]["mean"] for m in baseline_metrics if metric_name in m]
			if values:
				baseline_aggregated_metrics[metric_name] = {
					"mean": float(np.mean(values)),
					"min": float(np.min(values)),
					"max": float(np.max(values)),
					"std": float(np.std(values))
				}
		
		# Enregistrer la comparaison
		comparison = {
			"agent": {
				"mean_reward": float(mean_reward),
				"std_reward": float(std_reward),
				"metrics": aggregated_metrics
			},
			"baseline": {
				"mean_reward": float(baseline_mean_reward),
				"std_reward": float(baseline_std_reward),
				"metrics": baseline_aggregated_metrics
			},
			"improvement": {
				"reward": float((mean_reward - baseline_mean_reward) / max(abs(baseline_mean_reward), 1e-6) * 100),
				"metrics": {}
			}
		}
		
		# Calculer l'amélioration pour chaque métrique
		for metric_name in aggregated_metrics:
			if metric_name in baseline_aggregated_metrics:
				agent_value = aggregated_metrics[metric_name]["mean"]
				baseline_value = baseline_aggregated_metrics[metric_name]["mean"]
				
				if baseline_value != 0:
					improvement = (agent_value - baseline_value) / abs(baseline_value) * 100
				else:
					improvement = 0.0 if agent_value == 0 else 100.0
				
				comparison["improvement"]["metrics"][metric_name] = float(improvement)
		
		# Enregistrer la comparaison au format JSON
		comparison_file = os.path.join(eval_dir, "comparison.json")
		with open(comparison_file, 'w') as f:
			json.dump(comparison, f, indent=4)
		
		# Générer un graphique de comparaison
		if args.plot_metrics:
			plt.figure(figsize=(12, 8))
			
			# Métriques à comparer
			metrics_to_plot = ["coverage", "data_throughput", "energy_efficiency", "network_resilience", "user_satisfaction"]
			metrics_to_plot = [m for m in metrics_to_plot if m in comparison["improvement"]["metrics"]]
			
			# Valeurs d'amélioration
			improvements = [comparison["improvement"]["metrics"][m] for m in metrics_to_plot]
			
			# Créer le graphique
			bars = plt.bar(metrics_to_plot, improvements, color=['green' if i > 0 else 'red' for i in improvements])
			
			# Ajouter les étiquettes
			for bar in bars:
				height = bar.get_height()
				plt.text(bar.get_x() + bar.get_width()/2., height + 1,
						f'{height:.1f}%', ha='center', va='bottom')
			
			# Configurer le graphique
			plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
			plt.ylabel('Amélioration par rapport à la baseline (%)')
			plt.title('Amélioration des métriques par rapport à la stratégie aléatoire')
			plt.xticks(rotation=45)
			plt.grid(True, axis='y', alpha=0.3)
			plt.tight_layout()
			
			# Enregistrer le graphique
			plt.savefig(os.path.join(plot_dir, "improvement.png"), dpi=150)
			plt.close()
	
	print(f"Évaluation terminée. Résultats enregistrés dans {eval_dir}")
	print(f"Récompense moyenne sur {args.num_episodes} épisodes: {mean_reward:.2f} ± {std_reward:.2f}")
	
	# Afficher l'amélioration si disponible
	if args.compare_baseline:
		reward_improvement = comparison["improvement"]["reward"]
		print(f"Amélioration de la récompense par rapport à la baseline: {reward_improvement:.1f}%")
		
		for metric_name, improvement in comparison["improvement"]["metrics"].items():
			print(f"Amélioration de {metric_name}: {improvement:.1f}%")

if __name__ == "__main__":
	args = parse_arguments()
	evaluate(args)