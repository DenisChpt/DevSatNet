import numpy as np
from typing import Dict, List, Any, Optional

class EvaluationMetrics:
	"""
	Classe pour collecter et analyser les métriques d'évaluation.
	"""
	
	def __init__(self):
		"""
		Initialise les métriques d'évaluation.
		"""
		self.reset()
	
	def reset(self) -> None:
		"""
		Réinitialise toutes les métriques.
		"""
		self.rewards: List[float] = []
		self.lengths: List[int] = []
		self.allMetrics: List[Dict[str, Any]] = []
	
	def addEpisode(self, reward: float, length: int, metrics: Dict[str, Any]) -> None:
		"""
		Ajoute les résultats d'un épisode.
		
		Args:
			reward: Récompense totale de l'épisode
			length: Longueur de l'épisode
			metrics: Métriques additionnelles
		"""
		self.rewards.append(reward)
		self.lengths.append(length)
		self.allMetrics.append(metrics)
	
	def computeStats(self) -> Dict[str, Any]:
		"""
		Calcule les statistiques à partir des métriques collectées.
		
		Returns:
			Dictionnaire des statistiques
		"""
		# Si aucun épisode n'a été enregistré, retourner des statistiques vides
		if not self.rewards:
			return {
				"num_episodes": 0,
				"mean_reward": 0.0,
				"std_reward": 0.0,
				"min_reward": 0.0,
				"max_reward": 0.0,
				"mean_length": 0,
				"metrics": {}
			}
		
		# Statistiques de récompense
		rewards = np.array(self.rewards)
		meanReward = np.mean(rewards)
		stdReward = np.std(rewards)
		minReward = np.min(rewards)
		maxReward = np.max(rewards)
		
		# Statistiques de longueur
		lengths = np.array(self.lengths)
		meanLength = np.mean(lengths)
		
		# Collecter toutes les clés de métriques présentes
		metricKeys = set()
		for metrics in self.allMetrics:
			metricKeys.update(metrics.keys())
		
		# Agréger les métriques
		avgMetrics = {}
		
		for key in metricKeys:
			# Extraire les valeurs pour cette métrique de tous les épisodes
			values = [metrics.get(key, 0.0) for metrics in self.allMetrics]
			
			# Calculer la moyenne
			if values:
				avgMetrics[key] = np.mean(values)
			else:
				avgMetrics[key] = 0.0
		
		# Extraire les métriques les plus importantes
		coverage = avgMetrics.get("coverage", 0.0)
		dataThroughput = avgMetrics.get("data_throughput", 0.0)
		energyEfficiency = avgMetrics.get("energy_efficiency", 0.0)
		networkResilience = avgMetrics.get("network_resilience", 0.0)
		userSatisfaction = avgMetrics.get("user_satisfaction", 0.0)
		
		# Construire le dictionnaire de résultats
		result = {
			"num_episodes": len(self.rewards),
			"mean_reward": float(meanReward),
			"std_reward": float(stdReward),
			"min_reward": float(minReward),
			"max_reward": float(maxReward),
			"mean_length": float(meanLength),
			"metrics": avgMetrics,
			"key_metrics": {
				"coverage": float(coverage),
				"data_throughput": float(dataThroughput),
				"energy_efficiency": float(energyEfficiency),
				"network_resilience": float(networkResilience),
				"user_satisfaction": float(userSatisfaction)
			}
		}
		
		return result
	
	def getPerEpisodeStats(self) -> Dict[str, List[float]]:
		"""
		Retourne les statistiques par épisode.
		
		Returns:
			Dictionnaire des métriques par épisode
		"""
		result = {
			"rewards": self.rewards,
			"lengths": self.lengths
		}
		
		# Ajouter les autres métriques par épisode
		if self.allMetrics:
			# Collecter toutes les clés de métriques présentes
			metricKeys = set()
			for metrics in self.allMetrics:
				metricKeys.update(metrics.keys())
			
			for key in metricKeys:
				# Extraire les valeurs pour cette métrique de tous les épisodes
				values = [metrics.get(key, 0.0) for metrics in self.allMetrics]
				result[key] = values
		
		return result