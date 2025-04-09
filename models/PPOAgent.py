import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import gym
from gym import spaces
import time
import os

from models.ActorNetwork import ActorNetwork
from models.CriticNetwork import CriticNetwork
from models.MemoryBuffer import MemoryBuffer
from models.SharedFeatureExtractor import SharedFeatureExtractor
from models.ModelUtils import explained_variance

class PPOAgent:
	"""
	Agent utilisant l'algorithme Proximal Policy Optimization (PPO) pour
	l'apprentissage par renforcement dans l'environnement de constellation de satellites.
	"""
	
	def __init__(
		self,
		observationSpace: spaces.Dict,
		actionSpace: spaces.Box,
		deviceName: str = "cuda" if torch.cuda.is_available() else "cpu",
		satelliteFeaturesDim: int = 128,
		globalFeaturesDim: int = 64,
		hiddenDims: List[int] = [256, 256],
		learningRate: float = 3e-4,
		batchSize: int = 64,
		numEpochs: int = 10,
		clipRange: float = 0.2,
		valueLossCoef: float = 0.5,
		entropyCoef: float = 0.01,
		maxGradNorm: float = 0.5,
		targetKL: float = 0.01,
		gamma: float = 0.99,
		lambd: float = 0.95,
		bufferSize: int = 2048
	):
		"""
		Initialise l'agent PPO.
		
		Args:
			observationSpace: Espace d'observation de l'environnement
			actionSpace: Espace d'action de l'environnement
			deviceName: Dispositif d'exécution ('cuda' ou 'cpu')
			satelliteFeaturesDim: Dimension des caractéristiques extraites par satellite
			globalFeaturesDim: Dimension des caractéristiques globales extraites
			hiddenDims: Dimensions des couches cachées pour l'acteur et le critique
			learningRate: Taux d'apprentissage pour l'optimiseur
			batchSize: Taille des lots pour l'entraînement
			numEpochs: Nombre d'époques d'entraînement par lot
			clipRange: Paramètre de clip pour PPO
			valueLossCoef: Coefficient pour la perte de la fonction de valeur
			entropyCoef: Coefficient pour la perte d'entropie
			maxGradNorm: Norme maximale du gradient pour le clipping
			targetKL: Divergence KL cible pour early stopping
			gamma: Facteur d'actualisation pour les récompenses
			lambd: Paramètre lambda pour l'avantage généralisé
			bufferSize: Taille du tampon de mémoire
		"""
		self.observationSpace = observationSpace
		self.actionSpace = actionSpace
		self.device = torch.device(deviceName)
		
		# Hyperparamètres
		self.learningRate = learningRate
		self.batchSize = batchSize
		self.numEpochs = numEpochs
		self.clipRange = clipRange
		self.valueLossCoef = valueLossCoef
		self.entropyCoef = entropyCoef
		self.maxGradNorm = maxGradNorm
		self.targetKL = targetKL
		self.gamma = gamma
		self.lambd = lambd
		
		# Analyser l'espace d'observation
		self.numSatellites = 0
		self.satelliteObsDim = 0
		self.globalObsDim = 0
		
		for key, space in observationSpace.spaces.items():
			if key.startswith("satellite_"):
				self.numSatellites += 1
				self.satelliteObsDim = space.shape[0]
			elif key == "global":
				self.globalObsDim = space.shape[0]
		
		# Dimensions des caractéristiques
		self.satelliteFeaturesDim = satelliteFeaturesDim
		self.globalFeaturesDim = globalFeaturesDim
		
		# Dimensions de l'action
		self.actionDim = int(np.prod(actionSpace.shape[1:]))
		
		# Créer le réseau partagé d'extraction de caractéristiques
		self.featureExtractor = SharedFeatureExtractor(
			satelliteObsDim=self.satelliteObsDim,
			globalObsDim=self.globalObsDim,
			satelliteFeaturesDim=self.satelliteFeaturesDim,
			globalFeaturesDim=self.globalFeaturesDim,
			numSatellites=self.numSatellites
		).to(self.device)
		
		# Créer le réseau de politique (acteur)
		self.actor = ActorNetwork(
			satelliteFeaturesDim=self.satelliteFeaturesDim,
			globalFeaturesDim=self.globalFeaturesDim,
			actionDim=self.actionDim,
			hiddenDims=hiddenDims,
			numSatellites=self.numSatellites
		).to(self.device)
		
		# Créer le réseau de valeur (critique)
		self.critic = CriticNetwork(
			satelliteFeaturesDim=self.satelliteFeaturesDim,
			globalFeaturesDim=self.globalFeaturesDim,
			hiddenDims=hiddenDims
		).to(self.device)
		
		# Optimiseur
		self.optimizer = optim.Adam(
			list(self.featureExtractor.parameters()) + 
			list(self.actor.parameters()) + 
			list(self.critic.parameters()),
			lr=self.learningRate
		)
		
		# Tampon de mémoire
		self.buffer = MemoryBuffer(
			bufferSize=bufferSize,
			observationSpace=observationSpace,
			actionSpace=actionSpace,
			device=self.device,
			gamma=self.gamma,
			lambd=self.lambd
		)
		
		# Compteurs de statistiques
		self.trainingIterations = 0
		self.trainTime = 0.0
		self.explainedVariance = 0.0
		self.policyLoss = 0.0
		self.valueLoss = 0.0
		self.entropyLoss = 0.0
		self.approxKL = 0.0
		self.clipFraction = 0.0
	
	def selectAction(self, observation: Dict[str, np.ndarray], deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
		"""
		Sélectionne une action en fonction de l'observation actuelle.
		
		Args:
			observation: Dictionnaire d'observation
			deterministic: Si vrai, sélectionne l'action déterministe (mode évaluation)
			
		Returns:
			Tuple (action, extra_info)
		"""
		# Convertir l'observation en tenseurs et déplacer vers le périphérique
		obsDict = {}
		
		for key, value in observation.items():
			obsDict[key] = torch.tensor(value, dtype=torch.float32).unsqueeze(0).to(self.device)
		
		with torch.no_grad():
			# Extraire les caractéristiques
			satelliteFeatures, globalFeatures = self.featureExtractor(obsDict)
			
			# Prédire la distribution de politique
			actionMean, actionLogStd = self.actor(satelliteFeatures, globalFeatures)
			actionStd = torch.exp(actionLogStd)
			
			# Prédire la valeur
			value = self.critic(satelliteFeatures, globalFeatures)
		
		# Échantillonner l'action ou prendre le mode
		if deterministic:
			action = actionMean
		else:
			# Échantillonner à partir d'une distribution normale
			action = torch.normal(actionMean, actionStd)
			
			# Clipper l'action dans la plage [-1, 1]
			action = torch.clamp(action, -1.0, 1.0)
		
		# Convertir en numpy pour l'environnement
		actionNumpy = action.cpu().numpy()[0]
		
		# Informations supplémentaires pour l'entraînement
		extraInfo = {
			"action_mean": actionMean.cpu().numpy()[0],
			"action_std": actionStd.cpu().numpy()[0],
			"value": value.cpu().numpy()[0]
		}
		
		return actionNumpy, extraInfo
	
	def storeTransition(
		self,
		observation: Dict[str, np.ndarray],
		action: np.ndarray,
		reward: float,
		nextObservation: Dict[str, np.ndarray],
		done: bool,
		info: Dict[str, Any],
		extraInfo: Dict[str, Any]
	) -> None:
		"""
		Stocke une transition dans le tampon de mémoire.
		
		Args:
			observation: Observation actuelle
			action: Action prise
			reward: Récompense reçue
			nextObservation: Observation suivante
			done: Indicateur de fin d'épisode
			info: Informations supplémentaires de l'environnement
			extraInfo: Informations supplémentaires de l'agent
		"""
		self.buffer.add(
			observation=observation,
			action=action,
			reward=reward,
			nextObservation=nextObservation,
			done=done,
			value=extraInfo["value"],
			actionLogProb=self._computeLogProb(action, extraInfo["action_mean"], extraInfo["action_std"])
		)
	
	def _computeLogProb(self, action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
		"""
		Calcule le log de la probabilité de l'action.
		
		Args:
			action: Action prise
			mean: Moyenne de la distribution
			std: Écart-type de la distribution
			
		Returns:
			Log probabilité
		"""
		# Convertir en tenseurs
		action_tensor = torch.tensor(action, dtype=torch.float32)
		mean_tensor = torch.tensor(mean, dtype=torch.float32)
		std_tensor = torch.tensor(std, dtype=torch.float32)
		
		# Calcul de la log probabilité pour une distribution normale
		var = std_tensor ** 2
		log_prob = -((action_tensor - mean_tensor) ** 2) / (2 * var) - torch.log(torch.sqrt(2 * np.pi * var))
		
		# Somme sur toutes les dimensions de l'action
		return log_prob.sum().item()
	
	def train(self) -> Dict[str, float]:
		"""
		Entraîne l'agent sur les données collectées dans le tampon.
		
		Returns:
			Dictionnaire des métriques d'entraînement
		"""
		# Vérifier si le tampon est prêt pour l'entraînement
		if not self.buffer.isReady():
			return {
				"policy_loss": 0.0,
				"value_loss": 0.0,
				"entropy_loss": 0.0,
				"total_loss": 0.0,
				"approx_kl": 0.0,
				"clip_fraction": 0.0,
				"explained_variance": 0.0,
				"train_time": 0.0
			}
		
		# Chronométrer le temps d'entraînement
		startTime = time.time()
		
		# Calculer les avantages et les retours
		self.buffer.computeReturnsAndAdvantages()
		
		# Statistiques d'entraînement
		policyLosses = []
		valueLosses = []
		entropyLosses = []
		totalLosses = []
		approxKLs = []
		clipFractions = []
		
		# Entraînement sur plusieurs époques
		for epoch in range(self.numEpochs):
			# Générer des lots aléatoires
			for batch_tensors in self.buffer.getBatchGenerator(self.batchSize):
				# Déballer les tenseurs du lot
				obsBatch, actionBatch, oldValuesBatch, oldLogProbBatch, advantageBatch, returnBatch = batch_tensors
				
				# Forward pass
				# Extraire les caractéristiques
				satelliteFeatures, globalFeatures = self.featureExtractor(obsBatch)
				
				# Prédire la distribution de politique
				actionMean, actionLogStd = self.actor(satelliteFeatures, globalFeatures)
				actionStd = torch.exp(actionLogStd)
				
				# Prédire la valeur
				values = self.critic(satelliteFeatures, globalFeatures)
				
				# Calculer la log probabilité des actions
				actionDistribution = torch.distributions.Normal(actionMean, actionStd)
				logProbs = actionDistribution.log_prob(actionBatch).sum(2)
				entropy = actionDistribution.entropy().sum(2).mean()
				
				# Ratio pour le clipping PPO
				ratios = torch.exp(logProbs - oldLogProbBatch)
				
				# Calcul de la perte de politique avec clipping
				advNormalized = (advantageBatch - advantageBatch.mean()) / (advantageBatch.std() + 1e-8)
				
				# Première partie de la perte de politique (sans clipping)
				policySurr1 = ratios * advNormalized
				
				# Deuxième partie de la perte de politique (avec clipping)
				policySurr2 = torch.clamp(ratios, 1.0 - self.clipRange, 1.0 + self.clipRange) * advNormalized
				
				# Prendre le minimum des deux parties pour le clipping
				policyLoss = -torch.min(policySurr1, policySurr2).mean()
				
				# Calcul de la perte de valeur avec clipping
				valuePred = values.squeeze()
				valueLoss = F.mse_loss(valuePred, returnBatch)
				
				# Perte d'entropie
				entropyLoss = -entropy
				
				# Perte totale
				totalLoss = policyLoss + self.valueLossCoef * valueLoss + self.entropyCoef * entropyLoss
				
				# Backward pass et optimisation
				self.optimizer.zero_grad()
				totalLoss.backward()
				
				# Gradient clipping
				torch.nn.utils.clip_grad_norm_(
					list(self.featureExtractor.parameters()) + 
					list(self.actor.parameters()) + 
					list(self.critic.parameters()),
					self.maxGradNorm
				)
				
				self.optimizer.step()
				
				# Calculer les métriques supplémentaires
				with torch.no_grad():
					# Approximation de la divergence KL
					logRatio = logProbs - oldLogProbBatch
					approxKL = ((torch.exp(logRatio) - 1) - logRatio).mean().item()
					
					# Fraction de clipping
					clipFraction = ((ratios < 1.0 - self.clipRange) | (ratios > 1.0 + self.clipRange)).float().mean().item()
				
				# Collecter les statistiques
				policyLosses.append(policyLoss.item())
				valueLosses.append(valueLoss.item())
				entropyLosses.append(entropyLoss.item())
				totalLosses.append(totalLoss.item())
				approxKLs.append(approxKL)
				clipFractions.append(clipFraction)
				
				# Early stopping basé sur la divergence KL
				if approxKL > 1.5 * self.targetKL:
					break
			
			# Early stopping sur les époques
			if approxKL > 1.5 * self.targetKL:
				break
		
		# Calculer la variance expliquée
		y_pred = self.buffer.values.cpu().numpy()
		y_true = self.buffer.returns.cpu().numpy()
		explVarValue = explained_variance(y_true, y_pred)
		
		# Mettre à jour les compteurs
		self.trainingIterations += 1
		self.trainTime = time.time() - startTime
		self.explainedVariance = explVarValue
		self.policyLoss = np.mean(policyLosses)
		self.valueLoss = np.mean(valueLosses)
		self.entropyLoss = np.mean(entropyLosses)
		self.approxKL = np.mean(approxKLs)
		self.clipFraction = np.mean(clipFractions)
		
		# Vider le tampon après l'entraînement
		self.buffer.clear()
		
		# Retourner les statistiques d'entraînement
		return {
			"policy_loss": self.policyLoss,
			"value_loss": self.valueLoss,
			"entropy_loss": self.entropyLoss,
			"total_loss": np.mean(totalLosses),
			"approx_kl": self.approxKL,
			"clip_fraction": self.clipFraction,
			"explained_variance": self.explainedVariance,
			"train_time": self.trainTime
		}
	
	def save(self, path: str) -> None:
		"""
		Sauvegarde les modèles dans un fichier.
		
		Args:
			path: Chemin pour la sauvegarde
		"""
		os.makedirs(os.path.dirname(path), exist_ok=True)
		
		torch.save({
			"feature_extractor": self.featureExtractor.state_dict(),
			"actor": self.actor.state_dict(),
			"critic": self.critic.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"training_iterations": self.trainingIterations,
			"hyperparams": {
				"learning_rate": self.learningRate,
				"batch_size": self.batchSize,
				"num_epochs": self.numEpochs,
				"clip_range": self.clipRange,
				"value_loss_coef": self.valueLossCoef,
				"entropy_coef": self.entropyCoef,
				"max_grad_norm": self.maxGradNorm,
				"target_kl": self.targetKL,
				"gamma": self.gamma,
				"lambd": self.lambd
			}
		}, path)
	
	def load(self, path: str) -> None:
		"""
		Charge les modèles à partir d'un fichier.
		
		Args:
			path: Chemin vers le fichier de sauvegarde
		"""
		if not os.path.exists(path):
			print(f"Le fichier {path} n'existe pas.")
			return
		
		checkpoint = torch.load(path, map_location=self.device)
		
		self.featureExtractor.load_state_dict(checkpoint["feature_extractor"])
		self.actor.load_state_dict(checkpoint["actor"])
		self.critic.load_state_dict(checkpoint["critic"])
		self.optimizer.load_state_dict(checkpoint["optimizer"])
		self.trainingIterations = checkpoint["training_iterations"]
		
		# Mettre à jour les hyperparamètres si présents
		if "hyperparams" in checkpoint:
			hyperparams = checkpoint["hyperparams"]
			self.learningRate = hyperparams.get("learning_rate", self.learningRate)
			self.batchSize = hyperparams.get("batch_size", self.batchSize)
			self.numEpochs = hyperparams.get("num_epochs", self.numEpochs)
			self.clipRange = hyperparams.get("clip_range", self.clipRange)
			self.valueLossCoef = hyperparams.get("value_loss_coef", self.valueLossCoef)
			self.entropyCoef = hyperparams.get("entropy_coef", self.entropyCoef)
			self.maxGradNorm = hyperparams.get("max_grad_norm", self.maxGradNorm)
			self.targetKL = hyperparams.get("target_kl", self.targetKL)
			self.gamma = hyperparams.get("gamma", self.gamma)
			self.lambd = hyperparams.get("lambd", self.lambd)