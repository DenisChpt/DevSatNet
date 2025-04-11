# learning/trainers/a2c_trainer.py
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uuid
import time
from collections import deque

from learning.models.policy_network import PolicyNetwork
from learning.models.value_network import ValueNetwork
from learning.trainers.experience_buffer import ExperienceBuffer
from utils.serialization import Serializable


class A2CTrainer(Serializable):
	"""
	Implémentation de l'algorithme Advantage Actor-Critic (A2C).
	
	A2C est un algorithme d'apprentissage par renforcement qui combine 
	une politique (actor) et une fonction de valeur (critic) pour apprendre
	efficacement un comportement optimal.
	"""
	
	def __init__(
		self,
		stateDim: int,
		actionDim: int,
		continuousAction: bool = True,
		policyLr: float = 3e-4,
		valueLr: float = 1e-3,
		gamma: float = 0.99,
		entropyCoef: float = 0.01,
		valueLossCoef: float = 0.5,
		maxGradNorm: float = 0.5,
		numSteps: int = 5,
		batchSize: int = 64,
		useGae: bool = True,
		gaeParam: float = 0.95,
		policyHiddenDims: List[int] = [128, 64],
		valueHiddenDims: List[int] = [128, 64],
		activation: str = "tanh",
		stateNormalization: bool = True,
		device: str = "cpu",
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise l'entraîneur A2C.
		
		Args:
			stateDim: Dimension de l'espace d'état
			actionDim: Dimension de l'espace d'action
			continuousAction: Si True, actions continues, sinon actions discrètes
			policyLr: Taux d'apprentissage pour le réseau de politique
			valueLr: Taux d'apprentissage pour le réseau de valeur
			gamma: Facteur d'actualisation pour les récompenses futures
			entropyCoef: Coefficient pour l'entropie dans la fonction objectif
			valueLossCoef: Coefficient pour la perte de la fonction de valeur
			maxGradNorm: Norme maximale du gradient pour le clipping
			numSteps: Nombre d'étapes pour chaque mise à jour A2C
			batchSize: Taille des mini-batchs pour l'apprentissage
			useGae: Utiliser Generalized Advantage Estimation
			gaeParam: Paramètre lambda pour GAE
			policyHiddenDims: Dimensions des couches cachées du réseau de politique
			valueHiddenDims: Dimensions des couches cachées du réseau de valeur
			activation: Fonction d'activation pour les réseaux
			stateNormalization: Normaliser les états d'entrée
			device: Dispositif pour les calculs ("cpu" ou "cuda")
			seed: Graine pour le générateur de nombres aléatoires
		"""
		self.id: str = str(uuid.uuid4())
		self.stateDim: int = stateDim
		self.actionDim: int = actionDim
		self.continuousAction: bool = continuousAction
		self.policyLr: float = policyLr
		self.valueLr: float = valueLr
		self.gamma: float = gamma
		self.entropyCoef: float = entropyCoef
		self.valueLossCoef: float = valueLossCoef
		self.maxGradNorm: float = maxGradNorm
		self.numSteps: int = numSteps
		self.batchSize: int = batchSize
		self.useGae: bool = useGae
		self.gaeParam: float = gaeParam
		self.policyHiddenDims: List[int] = policyHiddenDims
		self.valueHiddenDims: List[int] = valueHiddenDims
		self.activation: str = activation
		self.stateNormalization: bool = stateNormalization
		self.device: str = device
		self.seed: int = seed if seed is not None else int(time.time())
		
		# Définir le dispositif (CPU ou GPU)
		self.device = torch.device(device if torch.cuda.is_available() else "cpu")
		
		# Définir la graine pour la reproductibilité
		torch.manual_seed(self.seed)
		np.random.seed(self.seed)
		
		# Créer les réseaux de politique et de valeur
		self.policyNetwork = PolicyNetwork(
			stateDim=stateDim,
			actionDim=actionDim,
			continuousAction=continuousAction,
			hiddenDims=policyHiddenDims,
			activation=activation,
			stateNormalization=stateNormalization,
			entropyCoef=entropyCoef
		).to(self.device)
		
		self.valueNetwork = ValueNetwork(
			stateDim=stateDim,
			hiddenDims=valueHiddenDims,
			activation=activation,
			stateNormalization=stateNormalization
		).to(self.device)
		
		# Créer les optimiseurs
		self.policyOptimizer = optim.Adam(self.policyNetwork.parameters(), lr=policyLr)
		self.valueOptimizer = optim.Adam(self.valueNetwork.parameters(), lr=valueLr)
		
		# Buffer d'expérience
		self.experienceBuffer = ExperienceBuffer(capacity=numSteps * batchSize)
		
		# Statistiques d'entraînement
		self.trainingStats = {
			"totalSteps": 0,
			"episodesCompleted": 0,
			"policyLosses": deque(maxlen=100),
			"valueLosses": deque(maxlen=100),
			"entropyLosses": deque(maxlen=100),
			"totalLosses": deque(maxlen=100),
			"returns": deque(maxlen=100),
			"episodeLengths": deque(maxlen=100),
			"lastUpdate": 0
		}
		
		# État actuel de l'entraînement
		self.isTraining: bool = False
	
	def selectAction(self, state: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
		"""
		Sélectionne une action selon la politique actuelle.
		
		Args:
			state: État observé
			deterministic: Si True, utilise la valeur moyenne au lieu d'échantillonner
			
		Returns:
			Tuple contenant l'action sélectionnée et le log de sa probabilité
		"""
		# Convertir l'état en tensor
		stateTensor = torch.FloatTensor(state).to(self.device)
		
		# Obtenir l'action de la politique
		with torch.no_grad():
			action, logProb, _ = self.policyNetwork.sampleAction(stateTensor, deterministic)
			
		# Convertir en numpy pour l'interaction avec l'environnement
		actionNp = action.cpu().numpy()
		logProbNp = logProb.cpu().numpy()
		
		return actionNp, logProbNp
	
	def storeExperience(
		self,
		state: np.ndarray,
		action: np.ndarray,
		reward: float,
		nextState: np.ndarray,
		done: bool,
		info: Dict[str, Any] = {}
	) -> None:
		"""
		Stocke une expérience dans le buffer.
		
		Args:
			state: État observé
			action: Action effectuée
			reward: Récompense reçue
			nextState: État suivant
			done: Indicateur de fin d'épisode
			info: Informations supplémentaires
		"""
		self.experienceBuffer.add(state, action, reward, nextState, done, info)
		
		# Mettre à jour les statistiques
		self.trainingStats["totalSteps"] += 1
		
		if done:
			self.trainingStats["episodesCompleted"] += 1
			
			# Ajouter la longueur de l'épisode aux statistiques
			if "episode_length" in info:
				self.trainingStats["episodeLengths"].append(info["episode_length"])
				
			# Ajouter le retour de l'épisode aux statistiques
			if "episode_return" in info:
				self.trainingStats["returns"].append(info["episode_return"])
	
	def update(self) -> Dict[str, float]:
		"""
		Effectue une mise à jour des paramètres selon l'algorithme A2C.
		
		Returns:
			Dictionnaire des métriques d'apprentissage
		"""
		# Vérifier s'il y a assez de données dans le buffer
		if len(self.experienceBuffer) < self.batchSize:
			return {
				"policy_loss": 0.0,
				"value_loss": 0.0,
				"entropy_loss": 0.0,
				"total_loss": 0.0
			}
			
		# Échantillonner des expériences du buffer
		batch = self.experienceBuffer.sample(self.batchSize)
		
		# Convertir les données en tensors
		states = torch.FloatTensor(batch.states).to(self.device)
		actions = torch.FloatTensor(batch.actions).to(self.device) if self.continuousAction else torch.LongTensor(batch.actions).to(self.device)
		rewards = torch.FloatTensor(batch.rewards).to(self.device)
		nextStates = torch.FloatTensor(batch.nextStates).to(self.device)
		dones = torch.FloatTensor(batch.dones).to(self.device)
		
		# Mettre à jour les statistiques d'état si la normalisation est activée
		if self.stateNormalization:
			self.policyNetwork.updateStateStats(states)
			self.valueNetwork.updateStateStats(states)
		
		# Calculer les avantages et les retours
		if self.useGae:
			advantages, returns = self.valueNetwork.calculateAdvantages(
				states, nextStates, rewards, dones, self.gamma, self.gaeParam
			)
		else:
			# Calculer les valeurs des états actuels
			values = self.valueNetwork(states)
			
			# Calculer les valeurs des états suivants
			nextValues = self.valueNetwork(nextStates)
			
			# Calculer les cibles TD: r + gamma * V(s') * (1 - done)
			returns = rewards + self.gamma * nextValues * (1.0 - dones)
			
			# Calculer les avantages: TD target - V(s)
			advantages = returns - values
		
		# Calculer les probabilités d'action selon la politique actuelle
		logProbs, entropy = self.policyNetwork.evaluateActions(states, actions)
		
		# Normaliser les avantages (pour stabilité numérique)
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
		
		# Calculer la perte de la politique: -log_prob * advantage
		policyLoss = -(logProbs * advantages.detach()).mean()
		
		# Calculer la perte d'entropie (négative car on veut maximiser l'entropie)
		entropyLoss = -entropy * self.entropyCoef
		
		# Calculer la perte du critique: MSE entre valeurs prédites et retours
		valuePredictions = self.valueNetwork(states)
		valueLoss = self.valueLossCoef * nn.functional.mse_loss(valuePredictions, returns.detach())
		
		# Perte totale
		totalLoss = policyLoss + valueLoss + entropyLoss
		
		# Optimisation de la politique (actor)
		self.policyOptimizer.zero_grad()
		policyLoss.backward()
		# Limiter la norme du gradient pour la stabilité
		if self.maxGradNorm > 0:
			nn.utils.clip_grad_norm_(self.policyNetwork.parameters(), self.maxGradNorm)
		self.policyOptimizer.step()
		
		# Optimisation de la fonction de valeur (critic)
		self.valueOptimizer.zero_grad()
		valueLoss.backward()
		# Limiter la norme du gradient pour la stabilité
		if self.maxGradNorm > 0:
			nn.utils.clip_grad_norm_(self.valueNetwork.parameters(), self.maxGradNorm)
		self.valueOptimizer.step()
		
		# Mettre à jour les statistiques d'entraînement
		self.trainingStats["policyLosses"].append(policyLoss.item())
		self.trainingStats["valueLosses"].append(valueLoss.item())
		self.trainingStats["entropyLosses"].append(entropyLoss.item())
		self.trainingStats["totalLosses"].append(totalLoss.item())
		self.trainingStats["lastUpdate"] = self.trainingStats["totalSteps"]
		
		# Retourner les métriques d'apprentissage
		return {
			"policy_loss": policyLoss.item(),
			"value_loss": valueLoss.item(),
			"entropy_loss": entropyLoss.item(),
			"total_loss": totalLoss.item()
		}
	
	def train(
		self,
		environmentInteraction: Callable[[np.ndarray, bool], Tuple[np.ndarray, float, bool, Dict[str, Any]]],
		totalSteps: int,
		updateInterval: int = 100,
		evaluationInterval: int = 1000,
		evaluationEpisodes: int = 5,
		saveInterval: int = 5000,
		savePath: Optional[str] = None,
		callbackFn: Optional[Callable[[Dict[str, Any]], None]] = None
	) -> Dict[str, Any]:
		"""
		Entraîne l'agent A2C sur l'environnement spécifié.
		
		Args:
			environmentInteraction: Fonction qui interagit avec l'environnement.
				Elle prend une action et un booléen déterministe, et retourne (nextState, reward, done, info)
			totalSteps: Nombre total d'étapes d'entraînement
			updateInterval: Nombre d'étapes entre les mises à jour
			evaluationInterval: Nombre d'étapes entre les évaluations
			evaluationEpisodes: Nombre d'épisodes pour chaque évaluation
			saveInterval: Nombre d'étapes entre les sauvegardes
			savePath: Chemin de sauvegarde du modèle
			callbackFn: Fonction de rappel appelée après chaque mise à jour
			
		Returns:
			Dictionnaire des statistiques d'entraînement
		"""
		self.isTraining = True
		
		# Initialiser l'état
		state, _, _, _ = environmentInteraction(None, False)  # None indique une réinitialisation
		
		# Statistiques d'entraînement supplémentaires
		trainingProgress = {
			"steps": [],
			"rewards": [],
			"policyLosses": [],
			"valueLosses": [],
			"entropyLosses": [],
			"totalLosses": [],
			"evaluations": []
		}
		
		# Stocker le début de l'heure d'entraînement
		startTime = time.time()
		
		# Boucle d'entraînement principale
		for step in range(totalSteps):
			# Sélectionner une action
			action, _ = self.selectAction(state)
			
			# Interagir avec l'environnement
			nextState, reward, done, info = environmentInteraction(action, False)
			
			# Stocker l'expérience
			self.storeExperience(state, action, reward, nextState, done, info)
			
			# Mettre à jour l'état
			state = nextState if not done else environmentInteraction(None, False)[0]
			
			# Mettre à jour les réseaux périodiquement
			if (step + 1) % updateInterval == 0:
				metrics = self.update()
				
				# Mettre à jour les statistiques d'entraînement
				trainingProgress["steps"].append(step + 1)
				trainingProgress["policyLosses"].append(metrics["policy_loss"])
				trainingProgress["valueLosses"].append(metrics["value_loss"])
				trainingProgress["entropyLosses"].append(metrics["entropy_loss"])
				trainingProgress["totalLosses"].append(metrics["total_loss"])
				
				# Récompense moyenne récente
				if self.trainingStats["returns"]:
					trainingProgress["rewards"].append(np.mean(self.trainingStats["returns"]))
				else:
					trainingProgress["rewards"].append(0.0)
				
				# Appeler la fonction de rappel si fournie
				if callbackFn is not None:
					callbackFn({
						"step": step + 1,
						"metrics": metrics,
						"stats": self.getStats(),
						"progress": trainingProgress
					})
			
			# Évaluer périodiquement
			if (step + 1) % evaluationInterval == 0:
				evaluationResult = self.evaluate(environmentInteraction, evaluationEpisodes)
				trainingProgress["evaluations"].append({
					"step": step + 1,
					"mean_return": evaluationResult["mean_return"],
					"std_return": evaluationResult["std_return"],
					"min_return": evaluationResult["min_return"],
					"max_return": evaluationResult["max_return"],
					"mean_length": evaluationResult["mean_length"]
				})
				
				# Appeler la fonction de rappel avec les résultats d'évaluation
				if callbackFn is not None:
					callbackFn({
						"step": step + 1,
						"evaluation": evaluationResult,
						"stats": self.getStats(),
						"progress": trainingProgress
					})
			
			# Sauvegarder périodiquement
			if savePath is not None and (step + 1) % saveInterval == 0:
				self.save(f"{savePath}_step_{step+1}.pt")
		
		# Calcul de la durée totale d'entraînement
		trainingTime = time.time() - startTime
		
		# Statistiques finales
		finalStats = self.getStats()
		finalStats["training_time"] = trainingTime
		finalStats["progress"] = trainingProgress
		
		self.isTraining = False
		
		return finalStats
	
	def evaluate(
		self,
		environmentInteraction: Callable[[np.ndarray, bool], Tuple[np.ndarray, float, bool, Dict[str, Any]]],
		numEpisodes: int = 5
	) -> Dict[str, float]:
		"""
		Évalue l'agent sur plusieurs épisodes.
		
		Args:
			environmentInteraction: Fonction qui interagit avec l'environnement
			numEpisodes: Nombre d'épisodes d'évaluation
			
		Returns:
			Dictionnaire des statistiques d'évaluation
		"""
		wasTraining = self.isTraining
		self.isTraining = False
		
		episodeReturns = []
		episodeLengths = []
		
		# Évaluer sur plusieurs épisodes
		for _ in range(numEpisodes):
			state, _, _, _ = environmentInteraction(None, True)  # Réinitialisation
			episodeReturn = 0.0
			episodeLength = 0
			done = False
			
			while not done:
				# Sélectionner l'action de manière déterministe
				action, _ = self.selectAction(state, deterministic=True)
				
				# Interagir avec l'environnement
				nextState, reward, done, _ = environmentInteraction(action, True)
				
				# Mettre à jour les compteurs
				episodeReturn += reward
				episodeLength += 1
				
				# Mettre à jour l'état
				state = nextState
			
			# Stocker les statistiques de l'épisode
			episodeReturns.append(episodeReturn)
			episodeLengths.append(episodeLength)
		
		# Restaurer l'état d'entraînement
		self.isTraining = wasTraining
		
		# Calculer les statistiques
		meanReturn = np.mean(episodeReturns)
		stdReturn = np.std(episodeReturns)
		minReturn = np.min(episodeReturns)
		maxReturn = np.max(episodeReturns)
		meanLength = np.mean(episodeLengths)
		
		return {
			"mean_return": meanReturn,
			"std_return": stdReturn,
			"min_return": minReturn,
			"max_return": maxReturn,
			"mean_length": meanLength,
			"num_episodes": numEpisodes,
			"returns": episodeReturns,
			"lengths": episodeLengths
		}
	
	def getStats(self) -> Dict[str, Any]:
		"""
		Retourne les statistiques actuelles d'entraînement.
		
		Returns:
			Dictionnaire des statistiques
		"""
		avgPolicyLoss = np.mean(self.trainingStats["policyLosses"]) if self.trainingStats["policyLosses"] else 0.0
		avgValueLoss = np.mean(self.trainingStats["valueLosses"]) if self.trainingStats["valueLosses"] else 0.0
		avgEntropyLoss = np.mean(self.trainingStats["entropyLosses"]) if self.trainingStats["entropyLosses"] else 0.0
		avgTotalLoss = np.mean(self.trainingStats["totalLosses"]) if self.trainingStats["totalLosses"] else 0.0
		avgReturn = np.mean(self.trainingStats["returns"]) if self.trainingStats["returns"] else 0.0
		avgEpisodeLength = np.mean(self.trainingStats["episodeLengths"]) if self.trainingStats["episodeLengths"] else 0.0
		
		return {
			"total_steps": self.trainingStats["totalSteps"],
			"episodes_completed": self.trainingStats["episodesCompleted"],
			"avg_policy_loss": avgPolicyLoss,
			"avg_value_loss": avgValueLoss,
			"avg_entropy_loss": avgEntropyLoss,
			"avg_total_loss": avgTotalLoss,
			"avg_return": avgReturn,
			"avg_episode_length": avgEpisodeLength,
			"last_update": self.trainingStats["lastUpdate"]
		}
	
	def save(self, path: str) -> None:
		"""
		Sauvegarde le modèle entraîné.
		
		Args:
			path: Chemin de sauvegarde
		"""
		torch.save({
			"policy_state_dict": self.policyNetwork.state_dict(),
			"value_state_dict": self.valueNetwork.state_dict(),
			"policy_optimizer": self.policyOptimizer.state_dict(),
			"value_optimizer": self.valueOptimizer.state_dict(),
			"stats": self.trainingStats,
			"config": {
				"state_dim": self.stateDim,
				"action_dim": self.actionDim,
				"continuous_action": self.continuousAction,
				"policy_lr": self.policyLr,
				"value_lr": self.valueLr,
				"gamma": self.gamma,
				"entropy_coef": self.entropyCoef,
				"value_loss_coef": self.valueLossCoef,
				"max_grad_norm": self.maxGradNorm,
				"num_steps": self.numSteps,
				"batch_size": self.batchSize,
				"use_gae": self.useGae,
				"gae_param": self.gaeParam,
				"policy_hidden_dims": self.policyHiddenDims,
				"value_hidden_dims": self.valueHiddenDims,
				"activation": self.activation,
				"state_normalization": self.stateNormalization,
				"device": self.device,
				"seed": self.seed
			}
		}, path)
	
	def load(self, path: str) -> None:
		"""
		Charge un modèle sauvegardé.
		
		Args:
			path: Chemin vers le fichier de sauvegarde
		"""
		checkpoint = torch.load(path, map_location=self.device)
		
		# Charger les états des réseaux
		self.policyNetwork.load_state_dict(checkpoint["policy_state_dict"])
		self.valueNetwork.load_state_dict(checkpoint["value_state_dict"])
		
		# Charger les états des optimiseurs
		self.policyOptimizer.load_state_dict(checkpoint["policy_optimizer"])
		self.valueOptimizer.load_state_dict(checkpoint["value_optimizer"])
		
		# Charger les statistiques
		self.trainingStats = checkpoint["stats"]
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet A2CTrainer en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de l'entraîneur A2C
		"""
		return {
			"id": self.id,
			"state_dim": self.stateDim,
			"action_dim": self.actionDim,
			"continuous_action": self.continuousAction,
			"policy_lr": self.policyLr,
			"value_lr": self.valueLr,
			"gamma": self.gamma,
			"entropy_coef": self.entropyCoef,
			"value_loss_coef": self.valueLossCoef,
			"max_grad_norm": self.maxGradNorm,
			"num_steps": self.numSteps,
			"batch_size": self.batchSize,
			"use_gae": self.useGae,
			"gae_param": self.gaeParam,
			"policy_hidden_dims": self.policyHiddenDims,
			"value_hidden_dims": self.valueHiddenDims,
			"activation": self.activation,
			"state_normalization": self.stateNormalization,
			"device": self.device,
			"seed": self.seed,
			"policy_network": self.policyNetwork.toDict(),
			"value_network": self.valueNetwork.toDict(),
			"training_stats": {
				"total_steps": self.trainingStats["totalSteps"],
				"episodes_completed": self.trainingStats["episodesCompleted"],
				"policy_losses": list(self.trainingStats["policyLosses"]),
				"value_losses": list(self.trainingStats["valueLosses"]),
				"entropy_losses": list(self.trainingStats["entropyLosses"]),
				"total_losses": list(self.trainingStats["totalLosses"]),
				"returns": list(self.trainingStats["returns"]),
				"episode_lengths": list(self.trainingStats["episodeLengths"]),
				"last_update": self.trainingStats["lastUpdate"]
			},
			"is_training": self.isTraining
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'A2CTrainer':
		"""
		Crée une instance de A2CTrainer à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de l'entraîneur
			
		Returns:
			Instance de A2CTrainer reconstruite
		"""
		# Créer l'entraîneur avec les hyperparamètres
		trainer = cls(
			stateDim=data["state_dim"],
			actionDim=data["action_dim"],
			continuousAction=data["continuous_action"],
			policyLr=data["policy_lr"],
			valueLr=data["value_lr"],
			gamma=data["gamma"],
			entropyCoef=data["entropy_coef"],
			valueLossCoef=data["value_loss_coef"],
			maxGradNorm=data["max_grad_norm"],
			numSteps=data["num_steps"],
			batchSize=data["batch_size"],
			useGae=data["use_gae"],
			gaeParam=data["gae_param"],
			policyHiddenDims=data["policy_hidden_dims"],
			valueHiddenDims=data["value_hidden_dims"],
			activation=data["activation"],
			stateNormalization=data["state_normalization"],
			device=data["device"],
			seed=data["seed"]
		)
		
		# Restaurer l'ID
		trainer.id = data["id"]
		
		# Restaurer les réseaux
		from learning.models.policy_network import PolicyNetwork
		from learning.models.value_network import ValueNetwork
		
		trainer.policyNetwork = PolicyNetwork.fromDict(data["policy_network"]).to(trainer.device)
		trainer.valueNetwork = ValueNetwork.fromDict(data["value_network"]).to(trainer.device)
		
		# Recréer les optimiseurs
		trainer.policyOptimizer = optim.Adam(trainer.policyNetwork.parameters(), lr=trainer.policyLr)
		trainer.valueOptimizer = optim.Adam(trainer.valueNetwork.parameters(), lr=trainer.valueLr)
		
		# Restaurer les statistiques d'entraînement
		stats = data["training_stats"]
		trainer.trainingStats["totalSteps"] = stats["total_steps"]
		trainer.trainingStats["episodesCompleted"] = stats["episodes_completed"]
		trainer.trainingStats["policyLosses"] = deque(stats["policy_losses"], maxlen=100)
		trainer.trainingStats["valueLosses"] = deque(stats["value_losses"], maxlen=100)
		trainer.trainingStats["entropyLosses"] = deque(stats["entropy_losses"], maxlen=100)
		trainer.trainingStats["totalLosses"] = deque(stats["total_losses"], maxlen=100)
		trainer.trainingStats["returns"] = deque(stats["returns"], maxlen=100)
		trainer.trainingStats["episodeLengths"] = deque(stats["episode_lengths"], maxlen=100)
		trainer.trainingStats["lastUpdate"] = stats["last_update"]
		
		# Restaurer l'état d'entraînement
		trainer.isTraining = data["is_training"]
		
		return trainer