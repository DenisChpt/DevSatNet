import os
import time
import numpy as np
import torch
import gym
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

from environment.SatelliteEnv import SatelliteEnv
from models.PPOAgent import PPOAgent
from training.Hyperparameters import HyperParameters
from training.Logger import Logger
from training.CheckpointManager import CheckpointManager
from training.EvaluationMetrics import EvaluationMetrics

class TrainingManager:
	"""
	Gère l'entraînement complet de l'agent dans l'environnement de constellation de satellites.
	"""
	
	def __init__(
		self,
		configPath: Optional[str] = None,
		logDir: str = "logs",
		checkpointDir: str = "checkpoints",
		render: bool = False,
		renderEval: bool = True,
		seed: Optional[int] = None,
		deviceName: str = "cuda" if torch.cuda.is_available() else "cpu"
	):
		"""
		Initialise le gestionnaire d'entraînement.
		
		Args:
			configPath: Chemin vers le fichier de configuration
			logDir: Répertoire pour les journaux
			checkpointDir: Répertoire pour les points de contrôle
			render: Activer le rendu pendant l'entraînement
			renderEval: Activer le rendu pendant l'évaluation
			seed: Graine aléatoire pour la reproductibilité
			deviceName: Dispositif d'exécution ('cuda' ou 'cpu')
		"""
		self.configPath = configPath
		self.logDir = logDir
		self.checkpointDir = checkpointDir
		self.render = render
		self.renderEval = renderEval
		self.seed = seed
		self.device = torch.device(deviceName)
		
		# Charger les hyperparamètres
		if configPath is not None:
			self.hyperParams = HyperParameters.fromFile(configPath)
		else:
			self.hyperParams = HyperParameters()
		
		# Créer les répertoires nécessaires
		os.makedirs(logDir, exist_ok=True)
		os.makedirs(checkpointDir, exist_ok=True)
		
		# Timestamp pour cette session d'entraînement
		self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		
		# Initialiser le logger
		self.logger = Logger(os.path.join(logDir, f"training_{self.timestamp}.log"))
		
		# Initialiser le gestionnaire de points de contrôle
		self.checkpointManager = CheckpointManager(checkpointDir, maxToKeep=5)
		
		# Créer l'environnement d'entraînement
		self.envTrain = SatelliteEnv(
			configPath=configPath,
			renderMode="none" if not render else "3d"
		)
		
		# Créer l'environnement d'évaluation
		self.envEval = SatelliteEnv(
			configPath=configPath,
			renderMode="none" if not renderEval else "3d"
		)
		
		# Définir les graines aléatoires si spécifiées
		if seed is not None:
			np.random.seed(seed)
			torch.manual_seed(seed)
			self.envTrain.seed(seed)
			self.envEval.seed(seed + 1000)  # Graine différente pour l'évaluation
		
		# Créer l'agent
		self.agent = PPOAgent(
			observationSpace=self.envTrain.observation_space,
			actionSpace=self.envTrain.action_space,
			deviceName=deviceName,
			satelliteFeaturesDim=self.hyperParams.satelliteFeaturesDim,
			globalFeaturesDim=self.hyperParams.globalFeaturesDim,
			hiddenDims=self.hyperParams.hiddenDims,
			learningRate=self.hyperParams.learningRate,
			batchSize=self.hyperParams.batchSize,
			numEpochs=self.hyperParams.numEpochs,
			clipRange=self.hyperParams.clipRange,
			valueLossCoef=self.hyperParams.valueLossCoef,
			entropyCoef=self.hyperParams.entropyCoef,
			maxGradNorm=self.hyperParams.maxGradNorm,
			targetKL=self.hyperParams.targetKL,
			gamma=self.hyperParams.gamma,
			lambd=self.hyperParams.lambd,
			bufferSize=self.hyperParams.bufferSize
		)
		
		# Métriques d'évaluation
		self.evaluationMetrics = EvaluationMetrics()
		
		# Statistiques d'entraînement
		self.episodeRewards: List[float] = []
		self.episodeLengths: List[int] = []
		self.trainIterations: int = 0
		self.totalTimeSteps: int = 0
		self.startTime: float = 0.0
		self.bestReward: float = -float('inf')
	
	def train(self, totalTimeSteps: int, evalFreq: int = 10000, logFreq: int = 1000, saveFreq: int = 10000) -> None:
		"""
		Entraîne l'agent pour un nombre spécifié d'étapes de temps.
		
		Args:
			totalTimeSteps: Nombre total d'étapes de temps pour l'entraînement
			evalFreq: Fréquence d'évaluation (en étapes de temps)
			logFreq: Fréquence de journalisation (en étapes de temps)
			saveFreq: Fréquence de sauvegarde des points de contrôle (en étapes de temps)
		"""
		# Journaliser les hyperparamètres
		self.logger.logHyperParameters(self.hyperParams)
		
		# Initialiser le temps
		self.startTime = time.time()
		
		# Réinitialiser l'environnement
		observation = self.envTrain.reset()
		
		# Variables d'état pour l'épisode en cours
		episodeReward = 0.0
		episodeLength = 0
		
		# Boucle principale d'entraînement
		for step in range(1, totalTimeSteps + 1):
			# Sélectionner une action
			action, extraInfo = self.agent.selectAction(observation)
			
			# Exécuter l'action dans l'environnement
			nextObservation, reward, done, info = self.envTrain.step(action)
			
			# Stocker la transition
			self.agent.storeTransition(
				observation=observation,
				action=action,
				reward=reward,
				nextObservation=nextObservation,
				done=done,
				info=info,
				extraInfo=extraInfo
			)
			
			# Mettre à jour l'observation
			observation = nextObservation
			
			# Mettre à jour les métriques de l'épisode
			episodeReward += reward
			episodeLength += 1
			
			# Fin d'épisode
			if done:
				# Enregistrer les métriques de l'épisode
				self.episodeRewards.append(episodeReward)
				self.episodeLengths.append(episodeLength)
				
				# Journaliser les métriques
				self.logger.logEpisode(
					episode=len(self.episodeRewards),
					reward=episodeReward,
					length=episodeLength,
					metrics=info.get("metrics", {})
				)
				
				# Réinitialiser les variables d'état pour le prochain épisode
				observation = self.envTrain.reset()
				episodeReward = 0.0
				episodeLength = 0
			
			# Entraîner l'agent si son tampon est prêt
			if step % self.hyperParams.trainingFrequency == 0:
				trainStats = self.agent.train()
				self.trainIterations += 1
				
				# Journaliser les statistiques d'entraînement
				if self.trainIterations % (logFreq // self.hyperParams.trainingFrequency) == 0:
					timeElapsed = time.time() - self.startTime
					stepsPerSecond = step / timeElapsed
					
					self.logger.logTrainingStats(
						iteration=self.trainIterations,
						timeSteps=step,
						stepsPerSecond=stepsPerSecond,
						trainStats=trainStats
					)
			
			# Évaluer l'agent périodiquement
			if step % evalFreq == 0:
				evalStats = self.evaluate(numEpisodes=5)
				
				# Journaliser les statistiques d'évaluation
				self.logger.logEvaluation(
					timeSteps=step,
					evalStats=evalStats
				)
				
				# Sauvegarder le meilleur modèle
				if evalStats["mean_reward"] > self.bestReward:
					self.bestReward = evalStats["mean_reward"]
					self.checkpointManager.save(
						self.agent,
						step,
						{
							"best_reward": self.bestReward,
							"eval_stats": evalStats
						},
						isBest=True
					)
			
			# Sauvegarder le modèle périodiquement
			if step % saveFreq == 0:
				self.checkpointManager.save(
					self.agent,
					step,
					{
						"best_reward": self.bestReward,
						"episode_rewards": self.episodeRewards,
						"episode_lengths": self.episodeLengths
					},
					isBest=False
				)
			
			# Mettre à jour le nombre total d'étapes
			self.totalTimeSteps = step
		
		# Sauvegarder le modèle final
		self.checkpointManager.save(
			self.agent,
			self.totalTimeSteps,
			{
				"best_reward": self.bestReward,
				"episode_rewards": self.episodeRewards,
				"episode_lengths": self.episodeLengths
			},
			isBest=False
		)
		
		# Enregistrer les résultats finaux
		self._saveResults()
	
	def evaluate(self, numEpisodes: int = 10) -> Dict[str, Any]:
		"""
		Évalue l'agent sur plusieurs épisodes.
		
		Args:
			numEpisodes: Nombre d'épisodes pour l'évaluation
			
		Returns:
			Statistiques d'évaluation
		"""
		# Réinitialiser les métriques d'évaluation
		self.evaluationMetrics.reset()
		
		# Évaluer sur plusieurs épisodes
		for episode in range(numEpisodes):
			# Réinitialiser l'environnement
			observation = self.envEval.reset()
			done = False
			episodeReward = 0.0
			episodeLength = 0
			
			# Boucle d'épisode
			while not done:
				# Sélectionner une action (mode déterministe)
				action, _ = self.agent.selectAction(observation, deterministic=True)
				
				# Exécuter l'action dans l'environnement
				nextObservation, reward, done, info = self.envEval.step(action)
				
				# Mettre à jour l'observation
				observation = nextObservation
				
				# Mettre à jour les métriques de l'épisode
				episodeReward += reward
				episodeLength += 1
				
				# Rendre si demandé
				if self.renderEval:
					self.envEval.render()
			
			# Enregistrer les métriques de l'épisode
			self.evaluationMetrics.addEpisode(
				reward=episodeReward,
				length=episodeLength,
				metrics=info.get("metrics", {})
			)
		
		# Calculer les statistiques d'évaluation
		return self.evaluationMetrics.computeStats()
	
	def _saveResults(self) -> None:
		"""
		Enregistre les résultats finaux de l'entraînement.
		"""
		# Créer le répertoire des résultats
		resultsDir = os.path.join(self.logDir, f"results_{self.timestamp}")
		os.makedirs(resultsDir, exist_ok=True)
		
		# Enregistrer les récompenses et longueurs d'épisode
		np.save(os.path.join(resultsDir, "episode_rewards.npy"), np.array(self.episodeRewards))
		np.save(os.path.join(resultsDir, "episode_lengths.npy"), np.array(self.episodeLengths))
		
		# Tracer et enregistrer les graphiques
		self._plotAndSaveMetrics(resultsDir)
		
		# Enregistrer un résumé des résultats
		summary = {
			"total_time_steps": self.totalTimeSteps,
			"train_iterations": self.trainIterations,
			"best_reward": self.bestReward,
			"last_10_episodes_avg_reward": np.mean(self.episodeRewards[-10:]) if len(self.episodeRewards) > 0 else 0.0,
			"total_episodes": len(self.episodeRewards),
			"training_time": time.time() - self.startTime,
		}
		
		with open(os.path.join(resultsDir, "summary.json"), "w") as f:
			json.dump(summary, f, indent=4)
	
	def _plotAndSaveMetrics(self, resultsDir: str) -> None:
		"""
		Trace et enregistre des graphiques des métriques d'entraînement.
		
		Args:
			resultsDir: Répertoire pour enregistrer les graphiques
		"""
		if len(self.episodeRewards) == 0:
			return
		
		# Tracer les récompenses par épisode
		plt.figure(figsize=(10, 6))
		plt.plot(self.episodeRewards)
		plt.xlabel("Épisode")
		plt.ylabel("Récompense")
		plt.title("Récompenses par épisode")
		plt.savefig(os.path.join(resultsDir, "episode_rewards.png"))
		plt.close()
		
		# Tracer les longueurs d'épisode
		plt.figure(figsize=(10, 6))
		plt.plot(self.episodeLengths)
		plt.xlabel("Épisode")
		plt.ylabel("Longueur")
		plt.title("Longueurs d'épisode")
		plt.savefig(os.path.join(resultsDir, "episode_lengths.png"))
		plt.close()
		
		# Tracer les récompenses moyennes (fenêtre glissante)
		if len(self.episodeRewards) >= 10:
			plt.figure(figsize=(10, 6))
			window_size = 10
			avg_rewards = [np.mean(self.episodeRewards[max(0, i - window_size):i+1]) for i in range(len(self.episodeRewards))]
			plt.plot(avg_rewards)
			plt.xlabel("Épisode")
			plt.ylabel("Récompense moyenne (fenêtre de 10)")
			plt.title("Récompense moyenne sur 10 épisodes")
			plt.savefig(os.path.join(resultsDir, "avg_episode_rewards.png"))
			plt.close()
	
	def loadAgent(self, checkpointPath: str) -> None:
		"""
		Charge un agent à partir d'un point de contrôle.
		
		Args:
			checkpointPath: Chemin vers le point de contrôle
		"""
		self.agent.load(checkpointPath)
		
		# Journaliser le chargement
		self.logger.log(f"Agent chargé à partir de {checkpointPath}")
	
	def testAgent(self, numEpisodes: int = 10, recordVideo: bool = False, videoPrefix: str = "satellite_") -> Dict[str, Any]:
		"""
		Teste l'agent en mode évaluation et enregistre éventuellement des vidéos.
		
		Args:
			numEpisodes: Nombre d'épisodes pour le test
			recordVideo: Activer l'enregistrement de vidéos
			videoPrefix: Préfixe pour les fichiers vidéo
			
		Returns:
			Statistiques de test
		"""
		# Configurer l'enregistrement vidéo si demandé
		if recordVideo:
			videoDir = os.path.join(self.logDir, f"videos_{self.timestamp}")
			os.makedirs(videoDir, exist_ok=True)
			self.envEval = gym.wrappers.RecordVideo(
				self.envEval,
				videoDir,
				episode_trigger=lambda x: True,
				name_prefix=videoPrefix
			)
		
		# Exécuter l'évaluation
		testStats = self.evaluate(numEpisodes)
		
		# Journaliser les résultats du test
		self.logger.log(f"Résultats du test: {json.dumps(testStats, indent=2)}")
		
		return testStats