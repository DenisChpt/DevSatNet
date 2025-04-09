import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

class Logger:
	"""
	Classe pour journaliser les événements et métriques d'entraînement.
	"""
	
	def __init__(self, logPath: str):
		"""
		Initialise le logger.
		
		Args:
			logPath: Chemin du fichier de journal
		"""
		self.logPath = logPath
		self.startTime = time.time()
		
		# Créer le répertoire parent si nécessaire
		os.makedirs(os.path.dirname(self.logPath), exist_ok=True)
		
		# Initialiser le fichier de journal
		with open(self.logPath, 'w') as f:
			f.write(f"=== Session de journalisation démarrée à {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
	
	def log(self, message: str) -> None:
		"""
		Journalise un message.
		
		Args:
			message: Message à journaliser
		"""
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		elapsed = time.time() - self.startTime
		
		formattedMessage = f"[{timestamp}] [{self._formatElapsedTime(elapsed)}] {message}"
		
		# Écrire dans le fichier
		with open(self.logPath, 'a') as f:
			f.write(formattedMessage + "\n")
		
		# Afficher également dans la console
		print(formattedMessage)
	
	def _formatElapsedTime(self, seconds: float) -> str:
		"""
		Formate un temps écoulé en heures, minutes, secondes.
		
		Args:
			seconds: Temps écoulé en secondes
			
		Returns:
			Temps formaté
		"""
		hours, remainder = divmod(int(seconds), 3600)
		minutes, seconds = divmod(remainder, 60)
		return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
	
	def logHyperParameters(self, hyperParams: Any) -> None:
		"""
		Journalise les hyperparamètres.
		
		Args:
			hyperParams: Objet d'hyperparamètres avec méthode toDict()
		"""
		self.log("=== Hyperparamètres ===")
		self.log(json.dumps(hyperParams.toDict(), indent=2))
		self.log("=====================\n")
	
	def logEpisode(self, episode: int, reward: float, length: int, metrics: Dict[str, Any]) -> None:
		"""
		Journalise les résultats d'un épisode.
		
		Args:
			episode: Numéro de l'épisode
			reward: Récompense totale
			length: Longueur de l'épisode
			metrics: Métriques additionnelles
		"""
		# Formater les métriques les plus importantes
		coverage = metrics.get("coverage", 0.0) * 100  # Convertir en pourcentage
		dataThroughput = metrics.get("data_throughput", 0.0) * 100  # Convertir en pourcentage
		energyEfficiency = metrics.get("energy_efficiency", 0.0) * 100  # Convertir en pourcentage
		userSatisfaction = metrics.get("user_satisfaction", 0.0) * 100  # Convertir en pourcentage
		
		message = (
			f"Épisode {episode} - "
			f"Récompense: {reward:.2f}, "
			f"Longueur: {length}, "
			f"Couverture: {coverage:.1f}%, "
			f"Débit: {dataThroughput:.1f}%, "
			f"Efficacité: {energyEfficiency:.1f}%, "
			f"Satisfaction: {userSatisfaction:.1f}%"
		)
		
		self.log(message)
	
	def logTrainingStats(self, iteration: int, timeSteps: int, stepsPerSecond: float, trainStats: Dict[str, float]) -> None:
		"""
		Journalise les statistiques d'entraînement.
		
		Args:
			iteration: Itération d'entraînement
			timeSteps: Nombre total d'étapes de temps
			stepsPerSecond: Étapes par seconde
			trainStats: Statistiques d'entraînement
		"""
		# Extraire les statistiques importantes
		policyLoss = trainStats.get("policy_loss", 0.0)
		valueLoss = trainStats.get("value_loss", 0.0)
		entropyLoss = trainStats.get("entropy_loss", 0.0)
		approxKL = trainStats.get("approx_kl", 0.0)
		explainedVar = trainStats.get("explained_variance", 0.0)
		
		message = (
			f"Entraînement #{iteration} - "
			f"Étapes: {timeSteps}, "
			f"Vitesse: {stepsPerSecond:.1f} étapes/s, "
			f"Policy loss: {policyLoss:.4f}, "
			f"Value loss: {valueLoss:.4f}, "
			f"Entropy: {entropyLoss:.4f}, "
			f"KL: {approxKL:.4f}, "
			f"Var expliquée: {explainedVar:.2f}"
		)
		
		self.log(message)
	
	def logEvaluation(self, timeSteps: int, evalStats: Dict[str, Any]) -> None:
		"""
		Journalise les résultats d'une évaluation.
		
		Args:
			timeSteps: Nombre d'étapes de temps avant l'évaluation
			evalStats: Statistiques d'évaluation
		"""
		# Extraire les statistiques importantes
		meanReward = evalStats.get("mean_reward", 0.0)
		stdReward = evalStats.get("std_reward", 0.0)
		minReward = evalStats.get("min_reward", 0.0)
		maxReward = evalStats.get("max_reward", 0.0)
		
		# Formater les métriques moyennes importantes
		metrics = evalStats.get("metrics", {})
		coverage = metrics.get("coverage", 0.0) * 100  # Convertir en pourcentage
		dataThroughput = metrics.get("data_throughput", 0.0) * 100  # Convertir en pourcentage
		energyEfficiency = metrics.get("energy_efficiency", 0.0) * 100  # Convertir en pourcentage
		userSatisfaction = metrics.get("user_satisfaction", 0.0) * 100  # Convertir en pourcentage
		
		message = (
			f"Évaluation à {timeSteps} étapes - "
			f"Récompense: {meanReward:.2f} ± {stdReward:.2f} (min: {minReward:.2f}, max: {maxReward:.2f}), "
			f"Couverture: {coverage:.1f}%, "
			f"Débit: {dataThroughput:.1f}%, "
			f"Efficacité: {energyEfficiency:.1f}%, "
			f"Satisfaction: {userSatisfaction:.1f}%"
		)
		
		self.log(message)
		
		# Journaliser les métriques détaillées
		self.log(f"Métriques d'évaluation détaillées: {json.dumps(metrics, indent=2)}")
	
	def logException(self, exception: Exception) -> None:
		"""
		Journalise une exception.
		
		Args:
			exception: L'exception à journaliser
		"""
		self.log(f"EXCEPTION: {type(exception).__name__}: {str(exception)}")
		import traceback
		self.log(traceback.format_exc())