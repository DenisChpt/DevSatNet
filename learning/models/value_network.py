# learning/models/value_network.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import uuid

from utils.serialization import Serializable


class ValueNetwork(nn.Module, Serializable):
	"""
	Réseau de valeur pour l'apprentissage par renforcement.
	Estime la valeur d'un état dans l'environnement.
	"""
	
	def __init__(
		self,
		stateDim: int,
		hiddenDims: List[int] = [128, 64],
		activation: str = "tanh",
		stateNormalization: bool = True,
		valueScale: float = 1.0,
		initStd: float = 0.1
	) -> None:
		"""
		Initialise le réseau de valeur.
		
		Args:
			stateDim: Dimension de l'espace d'état
			hiddenDims: Liste des dimensions des couches cachées
			activation: Fonction d'activation à utiliser
			stateNormalization: Normaliser les états d'entrée
			valueScale: Facteur d'échelle pour les valeurs de sortie
			initStd: Écart-type pour l'initialisation des poids
		"""
		super(ValueNetwork, self).__init__()
		
		self.id: str = str(uuid.uuid4())
		self.stateDim: int = stateDim
		self.hiddenDims: List[int] = hiddenDims
		self.activationName: str = activation
		self.stateNormalization: bool = stateNormalization
		self.valueScale: float = valueScale
		self.initStd: float = initStd
		
		# Statistiques pour la normalisation des états
		self.stateStats = {
			"mean": torch.zeros(stateDim),
			"std": torch.ones(stateDim),
			"count": 0
		}
		
		# Statistiques pour la normalisation des valeurs
		self.valueStats = {
			"mean": torch.tensor(0.0),
			"std": torch.tensor(1.0),
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
		
		# Compteur d'évaluations
		self.evaluationCount: int = 0
	
	def _buildNetwork(self) -> None:
		"""
		Construit l'architecture du réseau de valeur.
		"""
		# Liste des couches
		layers = []
		
		# Première couche cachée à partir de l'entrée
		layers.append(nn.Linear(self.stateDim, self.hiddenDims[0]))
		layers.append(self.activation)
		
		# Couches cachées suivantes
		for i in range(len(self.hiddenDims) - 1):
			layers.append(nn.Linear(self.hiddenDims[i], self.hiddenDims[i+1]))
			layers.append(self.activation)
			
		# Couche de sortie (une seule valeur)
		layers.append(nn.Linear(self.hiddenDims[-1], 1))
		
		# Créer le réseau séquentiel
		self.network = nn.Sequential(*layers)
	
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
	
	def denormalizeValue(self, value: torch.Tensor) -> torch.Tensor:
		"""
		Dénormalise une valeur de sortie.
		
		Args:
			value: Valeur normalisée
			
		Returns:
			Valeur dénormalisée
		"""
		if not self.stateNormalization:
			return value
			
		# Utiliser les statistiques pour dénormaliser
		mean = self.valueStats["mean"].to(value.device)
		std = self.valueStats["std"].to(value.device)
		
		# Éviter la multiplication par zéro
		std = torch.clamp(std, min=1e-6)
		
		# Dénormalisation
		denormalizedValue = value * std + mean
		
		return denormalizedValue
	
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
	
	def updateValueStats(self, values: torch.Tensor) -> None:
		"""
		Met à jour les statistiques pour la normalisation des valeurs.
		
		Args:
			values: Batch de valeurs observées
		"""
		if not self.stateNormalization:
			return
			
		batchSize = values.size(0)
		batchMean = torch.mean(values)
		batchVar = torch.var(values, unbiased=False)
		
		# Mise à jour incrémentale des statistiques
		if self.valueStats["count"] == 0:
			# Premier batch
			self.valueStats["mean"] = batchMean
			self.valueStats["std"] = torch.sqrt(batchVar + 1e-6)
		else:
			# Mettre à jour la moyenne et l'écart-type de manière incrémentale
			oldCount = self.valueStats["count"]
			newCount = oldCount + batchSize
			
			# Mise à jour de la moyenne
			newMean = (self.valueStats["mean"] * oldCount + batchMean * batchSize) / newCount
			
			# Mise à jour de la variance
			oldM = self.valueStats["mean"]
			newM = newMean
			oldS = self.valueStats["std"] ** 2 * oldCount
			newS = batchVar * batchSize
			
			# Formule pour combiner les variances
			combinedVar = (oldS + newS) / newCount + oldCount * batchSize * ((oldM - newM) ** 2) / (newCount ** 2)
			
			# Mettre à jour les statistiques
			self.valueStats["mean"] = newMean
			self.valueStats["std"] = torch.sqrt(combinedVar + 1e-6)
			
		# Mettre à jour le compteur
		self.valueStats["count"] += batchSize
	
	def forward(self, state: torch.Tensor) -> torch.Tensor:
		"""
		Propagation avant dans le réseau de valeur.
		
		Args:
			state: Tensor d'état d'entrée
			
		Returns:
			Valeur estimée de l'état
		"""
		# Mettre à jour le compteur d'évaluations
		self.evaluationCount += 1
		
		# Normaliser l'état si nécessaire
		if self.stateNormalization:
			state = self.normalizeState(state)
			
		# Passage à travers le réseau
		value = self.network(state)
		
		# Mise à l'échelle de la valeur
		value = value * self.valueScale
		
		# Si le tensor a plusieurs dimensions, squeeze pour en faire un vecteur
		if value.dim() > 1 and value.size(1) == 1:
			value = value.squeeze(1)
			
		return value
	
	def evaluateState(self, state: torch.Tensor) -> torch.Tensor:
		"""
		Évalue la valeur d'un état.
		
		Args:
			state: Tensor d'état à évaluer
			
		Returns:
			Valeur estimée de l'état
		"""
		with torch.no_grad():
			return self.forward(state)
	
	def getTDTargets(self, rewards: torch.Tensor, nextStates: torch.Tensor, dones: torch.Tensor, gamma: float) -> torch.Tensor:
		"""
		Calcule les cibles de différence temporelle pour l'apprentissage.
		
		Args:
			rewards: Batch de récompenses
			nextStates: Batch d'états suivants
			dones: Batch d'indicateurs de fin (1 si terminal, 0 sinon)
			gamma: Facteur d'actualisation
			
		Returns:
			Cibles TD pour l'apprentissage
		"""
		# Évaluer les états suivants
		with torch.no_grad():
			nextValues = self.forward(nextStates)
			
		# Calculer les cibles TD: r + gamma * V(s') * (1 - done)
		tdTargets = rewards + gamma * nextValues * (1.0 - dones)
		
		return tdTargets
	
	def calculateAdvantages(
		self,
		states: torch.Tensor,
		nextStates: torch.Tensor,
		rewards: torch.Tensor,
		dones: torch.Tensor,
		gamma: float,
		lambda_gae: float = 0.95
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Calcule les avantages et les retours pour l'apprentissage par avantage généralisé (GAE).
		
		Args:
			states: Batch d'états
			nextStates: Batch d'états suivants
			rewards: Batch de récompenses
			dones: Batch d'indicateurs de fin
			gamma: Facteur d'actualisation
			lambda_gae: Paramètre lambda pour GAE
			
		Returns:
			Tuple contenant les avantages et les retours
		"""
		# Évaluer les états actuels et suivants
		with torch.no_grad():
			values = self.forward(states)
			nextValues = self.forward(nextStates)
		
		# Calculer les deltas TD: r + gamma * V(s') * (1 - done) - V(s)
		deltas = rewards + gamma * nextValues * (1.0 - dones) - values
		
		# Calculer les avantages par GAE
		advantages = torch.zeros_like(deltas)
		gae = 0.0
		
		# Parcourir les transitions en ordre inverse
		for t in reversed(range(len(deltas))):
			if t == len(deltas) - 1:
				# Dernière transition, pas de continuité
				nextNonTerminal = 1.0 - dones[t]
				nextGae = 0.0
			else:
				nextNonTerminal = 1.0 - dones[t]
				nextGae = advantages[t + 1]
				
			# Calculer l'avantage actuel
			gae = deltas[t] + gamma * lambda_gae * nextNonTerminal * nextGae
			advantages[t] = gae
		
		# Calculer les retours: avantage + valeur
		returns = advantages + values
		
		return advantages, returns
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet ValueNetwork en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du réseau de valeur
		"""
		# Sérialiser les paramètres du réseau
		state_dict = {k: v.cpu().numpy() for k, v in self.state_dict().items()}
		
		return {
			"id": self.id,
			"stateDim": self.stateDim,
			"hiddenDims": self.hiddenDims,
			"activationName": self.activationName,
			"stateNormalization": self.stateNormalization,
			"valueScale": self.valueScale,
			"initStd": self.initStd,
			"state_dict": state_dict,
			"stateStats": {
				"mean": self.stateStats["mean"].cpu().numpy(),
				"std": self.stateStats["std"].cpu().numpy(),
				"count": self.stateStats["count"]
			},
			"valueStats": {
				"mean": self.valueStats["mean"].cpu().numpy(),
				"std": self.valueStats["std"].cpu().numpy(),
				"count": self.valueStats["count"]
			},
			"evaluationCount": self.evaluationCount
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'ValueNetwork':
		"""
		Crée une instance de ValueNetwork à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du réseau
			
		Returns:
			Instance de ValueNetwork reconstruite
		"""
		# Créer le réseau avec les hyperparamètres
		network = cls(
			stateDim=data["stateDim"],
			hiddenDims=data["hiddenDims"],
			activation=data["activationName"],
			stateNormalization=data["stateNormalization"],
			valueScale=data["valueScale"],
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
		
		# Restaurer les statistiques de valeur
		network.valueStats["mean"] = torch.tensor(data["valueStats"]["mean"])
		network.valueStats["std"] = torch.tensor(data["valueStats"]["std"])
		network.valueStats["count"] = data["valueStats"]["count"]
		
		# Restaurer le compteur d'évaluations
		network.evaluationCount = data["evaluationCount"]
		
		return network