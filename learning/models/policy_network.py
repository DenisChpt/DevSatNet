# learning/models/policy_network.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import uuid

from utils.serialization import Serializable


class PolicyNetwork(nn.Module, Serializable):
	"""
	Réseau de politique pour l'apprentissage par renforcement.
	Détermine quelles actions prendre en fonction de l'état observé.
	"""
	
	def __init__(
		self,
		stateDim: int,
		actionDim: int,
		continuousAction: bool = True,
		hiddenDims: List[int] = [128, 64],
		activation: str = "tanh",
		stateNormalization: bool = True,
		actionScaling: float = 1.0,
		entropyCoef: float = 0.01,
		initStd: float = 0.1
	) -> None:
		"""
		Initialise le réseau de politique.
		
		Args:
			stateDim: Dimension de l'espace d'état
			actionDim: Dimension de l'espace d'action
			continuousAction: Si True, actions continues, sinon actions discrètes
			hiddenDims: Liste des dimensions des couches cachées
			activation: Fonction d'activation à utiliser
			stateNormalization: Normaliser les états d'entrée
			actionScaling: Facteur d'échelle pour les actions continues
			entropyCoef: Coefficient pour le terme d'entropie dans la fonction objectif
			initStd: Écart-type pour l'initialisation des poids
		"""
		super(PolicyNetwork, self).__init__()
		
		self.id: str = str(uuid.uuid4())
		self.stateDim: int = stateDim
		self.actionDim: int = actionDim
		self.continuousAction: bool = continuousAction
		self.hiddenDims: List[int] = hiddenDims
		self.activationName: str = activation
		self.stateNormalization: bool = stateNormalization
		self.actionScaling: float = actionScaling
		self.entropyCoef: float = entropyCoef
		self.initStd: float = initStd
		
		# Statistiques pour la normalisation des états
		self.stateStats = {
			"mean": torch.zeros(stateDim),
			"std": torch.ones(stateDim),
			"count": 0
		}
		
		# Définir la fonction d'activation
		if activation == "relu":
			self.activation = nn.ReLU()
		elif activation == "tanh":
			self.activation = nn.Tanh()
		elif activation == "sigmoid":
			self.activation = nn.Sigmoid()
		else:
			self.activation = nn.Tanh()  # Par défaut
			
		# Construire l'architecture du réseau
		self._buildNetwork()
		
		# Initialiser les poids
		self._initializeWeights()
		
		# Compteur d'actions
		self.actionCount: int = 0
	
	def _buildNetwork(self) -> None:
		"""
		Construit l'architecture du réseau de politique.
		"""
		# Liste des couches pour la partie commune
		layers = []
		
		# Première couche cachée à partir de l'entrée
		layers.append(nn.Linear(self.stateDim, self.hiddenDims[0]))
		layers.append(self.activation)
		
		# Couches cachées suivantes
		for i in range(len(self.hiddenDims) - 1):
			layers.append(nn.Linear(self.hiddenDims[i], self.hiddenDims[i+1]))
			layers.append(self.activation)
			
		# Réseau principal (partie commune)
		self.backbone = nn.Sequential(*layers)
		
		# Couche de sortie pour les actions
		if self.continuousAction:
			# Pour les actions continues, on produit la moyenne et l'écart-type
			self.muHead = nn.Linear(self.hiddenDims[-1], self.actionDim)
			self.logStdHead = nn.Linear(self.hiddenDims[-1], self.actionDim)
			# Bornes pour le log de l'écart-type (pour stabilité numérique)
			self.minLogStd = -20
			self.maxLogStd = 2
		else:
			# Pour les actions discrètes, on produit des logits
			self.categoricalHead = nn.Linear(self.hiddenDims[-1], self.actionDim)
	
	def _initializeWeights(self) -> None:
		"""
		Initialise les poids du réseau avec une distribution normale.
		"""
		for module in self.modules():
			if isinstance(module, nn.Linear):
				# Initialisation des poids
				nn.init.normal_(module.weight, mean=0.0, std=self.initStd)
				# Initialisation des biais à zéro
				if module.bias is not None:
					nn.init.zeros_(module.bias)
	
	def normalizeState(self, state: torch.Tensor) -> torch.Tensor:
		"""
		Normalise un état d'entrée en utilisant les statistiques accumulées.
		
		Args:
			state: Tensor d'état à normaliser
			
		Returns:
			État normalisé
		"""
		if not self.stateNormalization:
			return state
			
		# Utiliser les statistiques pour normaliser
		mean = self.stateStats["mean"].to(state.device)
		std = self.stateStats["std"].to(state.device)
		
		# Éviter la division par zéro
		std = torch.clamp(std, min=1e-6)
		
		# Normalisation
		normalizedState = (state - mean) / std
		
		return normalizedState
	
	def updateStateStats(self, states: torch.Tensor) -> None:
		"""
		Met à jour les statistiques pour la normalisation des états.
		
		Args:
			states: Batch d'états observés
		"""
		if not self.stateNormalization:
			return
			
		batchSize = states.size(0)
		batchMean = torch.mean(states, dim=0)
		batchVar = torch.var(states, dim=0, unbiased=False)
		
		# Mise à jour incrémentale des statistiques
		if self.stateStats["count"] == 0:
			# Premier batch
			self.stateStats["mean"] = batchMean
			self.stateStats["std"] = torch.sqrt(batchVar + 1e-6)
		else:
			# Mettre à jour la moyenne et l'écart-type de manière incrémentale
			oldCount = self.stateStats["count"]
			newCount = oldCount + batchSize
			
			# Mise à jour de la moyenne
			newMean = (self.stateStats["mean"] * oldCount + batchMean * batchSize) / newCount
			
			# Mise à jour de la variance
			oldM = self.stateStats["mean"]
			newM = newMean
			oldS = self.stateStats["std"] ** 2 * oldCount
			newS = batchVar * batchSize
			
			# Formule pour combiner les variances
			combinedVar = (oldS + newS) / newCount + oldCount * batchSize * ((oldM - newM) ** 2) / (newCount ** 2)
			
			# Mettre à jour les statistiques
			self.stateStats["mean"] = newMean
			self.stateStats["std"] = torch.sqrt(combinedVar + 1e-6)
			
		# Mettre à jour le compteur
		self.stateStats["count"] += batchSize
	
	def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
		"""
		Propagation avant dans le réseau de politique.
		
		Args:
			state: Tensor d'état d'entrée
			
		Returns:
			Tuple contenant:
			- Pour actions continues: (moyennes, écarts-types)
			- Pour actions discrètes: (logits, None)
		"""
		# Normaliser l'état si nécessaire
		if self.stateNormalization:
			state = self.normalizeState(state)
			
		# Passage à travers le réseau principal
		features = self.backbone(state)
		
		if self.continuousAction:
			# Pour les actions continues
			mu = self.muHead(features)
			log_std = self.logStdHead(features)
			
			# Limite le log_std pour stabilité numérique
			log_std = torch.clamp(log_std, self.minLogStd, self.maxLogStd)
			
			# Calculer l'écart-type
			std = torch.exp(log_std)
			
			return mu, std
		else:
			# Pour les actions discrètes
			logits = self.categoricalHead(features)
			
			return logits, None
	
	def sampleAction(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[distributions.Distribution]]:
		"""
		Échantillonne une action à partir de la politique.
		
		Args:
			state: Tensor d'état d'entrée
			deterministic: Si True, retourne l'action la plus probable au lieu d'échantillonner
			
		Returns:
			Tuple contenant:
			- L'action échantillonnée
			- Le log de la probabilité de l'action
			- La distribution de probabilité (pour le calcul d'entropie)
		"""
		# Mettre à jour le compteur d'actions
		self.actionCount += 1
		
		# Passage à travers le réseau
		if self.continuousAction:
			mu, std = self.forward(state)
			
			# Créer la distribution normale
			dist = distributions.Normal(mu, std)
			
			if deterministic:
				# Action déterministe: utiliser la moyenne
				action = mu
			else:
				# Échantillonner l'action
				action = dist.sample()
				
			# Appliquer l'échelle et les limites
			scaledAction = torch.tanh(action) * self.actionScaling
			
			# Log de probabilité de l'action
			# Note: nous calculons le log_prob de l'action avant mise à l'échelle
			log_prob = dist.log_prob(action).sum(dim=-1)
			
			return scaledAction, log_prob, dist
		else:
			logits, _ = self.forward(state)
			
			# Créer la distribution catégorielle
			dist = distributions.Categorical(logits=logits)
			
			if deterministic:
				# Action déterministe: choisir l'action la plus probable
				action = torch.argmax(logits, dim=-1)
			else:
				# Échantillonner l'action
				action = dist.sample()
				
			# Log de probabilité de l'action
			log_prob = dist.log_prob(action)
			
			return action, log_prob, dist
	
	def evaluateActions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Évalue les actions données par rapport à la politique actuelle.
		
		Args:
			states: Batch d'états
			actions: Batch d'actions correspondantes
			
		Returns:
			Tuple contenant:
			- Les log-probabilités des actions
			- L'entropie de la politique
		"""
		if self.continuousAction:
			# Pour les actions continues
			mu, std = self.forward(states)
			
			# Convertir les actions mises à l'échelle en actions brutes
			# Inverse de la transformation tanh
			unscaledActions = torch.atanh(torch.clamp(actions / self.actionScaling, -0.999, 0.999))
			
			# Créer la distribution normale
			dist = distributions.Normal(mu, std)
			
			# Log-probabilités des actions
			log_probs = dist.log_prob(unscaledActions).sum(dim=-1)
			
			# Entropie de la politique
			entropy = dist.entropy().sum(dim=-1).mean()
		else:
			# Pour les actions discrètes
			logits, _ = self.forward(states)
			
			# Créer la distribution catégorielle
			dist = distributions.Categorical(logits=logits)
			
			# Log-probabilités des actions
			log_probs = dist.log_prob(actions)
			
			# Entropie de la politique
			entropy = dist.entropy().mean()
			
		return log_probs, entropy
	
	def getEntropyLoss(self, dist: Optional[distributions.Distribution] = None, states: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""
		Calcule la perte d'entropie pour encourager l'exploration.
		
		Args:
			dist: Distribution de probabilité des actions
			states: Batch d'états (si distribution non fournie)
			
		Returns:
			Perte d'entropie
		"""
		if dist is None and states is not None:
			# Calculer la distribution à partir des états
			if self.continuousAction:
				mu, std = self.forward(states)
				dist = distributions.Normal(mu, std)
			else:
				logits, _ = self.forward(states)
				dist = distributions.Categorical(logits=logits)
		
		if dist is None:
			return torch.tensor(0.0)
			
		# Calculer l'entropie
		if isinstance(dist, distributions.Normal):
			entropy = dist.entropy().sum(dim=-1).mean()
		else:
			entropy = dist.entropy().mean()
			
		# Retourner la perte d'entropie négative (à maximiser)
		return -self.entropyCoef * entropy
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet PolicyNetwork en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du réseau de politique
		"""
		# Sérialiser les paramètres du réseau
		state_dict = {k: v.cpu().numpy() for k, v in self.state_dict().items()}
		
		return {
			"id": self.id,
			"stateDim": self.stateDim,
			"actionDim": self.actionDim,
			"continuousAction": self.continuousAction,
			"hiddenDims": self.hiddenDims,
			"activationName": self.activationName,
			"stateNormalization": self.stateNormalization,
			"actionScaling": self.actionScaling,
			"entropyCoef": self.entropyCoef,
			"initStd": self.initStd,
			"state_dict": state_dict,
			"stateStats": {
				"mean": self.stateStats["mean"].cpu().numpy(),
				"std": self.stateStats["std"].cpu().numpy(),
				"count": self.stateStats["count"]
			},
			"actionCount": self.actionCount
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'PolicyNetwork':
		"""
		Crée une instance de PolicyNetwork à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du réseau
			
		Returns:
			Instance de PolicyNetwork reconstruite
		"""
		# Créer le réseau avec les hyperparamètres
		network = cls(
			stateDim=data["stateDim"],
			actionDim=data["actionDim"],
			continuousAction=data["continuousAction"],
			hiddenDims=data["hiddenDims"],
			activation=data["activationName"],
			stateNormalization=data["stateNormalization"],
			actionScaling=data["actionScaling"],
			entropyCoef=data["entropyCoef"],
			initStd=data["initStd"]
		)
		
		# Restaurer l'ID
		network.id = data["id"]
		
		# Convertir les paramètres du réseau de numpy à torch
		state_dict = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
		network.load_state_dict(state_dict)
		
		# Restaurer les statistiques d'état
		network.stateStats["mean"] = torch.tensor(data["stateStats"]["mean"])
		network.stateStats["std"] = torch.tensor(data["stateStats"]["std"])
		network.stateStats["count"] = data["stateStats"]["count"]
		
		# Restaurer le compteur d'actions
		network.actionCount = data["actionCount"]
		
		return network