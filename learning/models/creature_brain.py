# learning/models/creature_brain.py
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import uuid
import copy

from utils.serialization import Serializable


class CreatureBrain(nn.Module, Serializable):
	"""
	Modèle de réseau neuronal représentant le cerveau d'une créature.
	Contrôle le comportement de la créature en transformant les entrées sensorielles en actions.
	"""
	
	def __init__(
		self,
		inputDim: int,
		outputDim: int,
		hiddenDim: int = 64,
		numLayers: int = 2,
		activation: str = "relu",  # "relu", "tanh", "sigmoid"
		useLSTM: bool = False,
		useAttention: bool = False,
		dropout: float = 0.0,
		initStd: float = 0.1,
		genomeId: Optional[str] = None
	) -> None:
		"""
		Initialise le réseau neuronal représentant le cerveau de la créature.
		
		Args:
			inputDim: Dimension de l'espace d'entrée (observations)
			outputDim: Dimension de l'espace de sortie (actions)
			hiddenDim: Dimension des couches cachées
			numLayers: Nombre de couches cachées
			activation: Fonction d'activation à utiliser
			useLSTM: Utiliser des couches LSTM pour la mémoire
			useAttention: Utiliser un mécanisme d'attention
			dropout: Taux de dropout pour la régularisation
			initStd: Écart-type pour l'initialisation des poids
			genomeId: Identifiant du génome associé à ce cerveau
		"""
		super(CreatureBrain, self).__init__()
		
		self.id: str = str(uuid.uuid4())
		self.inputDim: int = inputDim
		self.outputDim: int = outputDim
		self.hiddenDim: int = hiddenDim
		self.numLayers: int = numLayers
		self.activationName: str = activation
		self.useLSTM: bool = useLSTM
		self.useAttention: bool = useAttention
		self.dropout: float = dropout
		self.initStd: float = initStd
		self.genomeId: Optional[str] = genomeId
		
		# Définir la fonction d'activation
		if activation == "relu":
			self.activation = nn.ReLU()
		elif activation == "tanh":
			self.activation = nn.Tanh()
		elif activation == "sigmoid":
			self.activation = nn.Sigmoid()
		else:
			self.activation = nn.ReLU()  # Par défaut
			
		# Construire l'architecture du réseau
		self._buildNetwork()
		
		# État interne pour les réseaux récurrents
		self.hiddenState: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
		
		# Initialiser les poids
		self._initializeWeights()
		
		# Compteur d'inférences
		self.inferenceCount: int = 0
		
		# Horloge interne (temps pour la créature)
		self.internalClock: float = 0.0
	
	def _buildNetwork(self) -> None:
		"""
		Construit l'architecture du réseau neuronal.
		"""
		# Liste des couches
		self.layers = nn.ModuleList()
		
		# Couche d'entrée
		if self.useLSTM:
			# LSTM pour le traitement séquentiel avec mémoire
			self.lstm = nn.LSTM(
				input_size=self.inputDim,
				hidden_size=self.hiddenDim,
				num_layers=self.numLayers,
				batch_first=True,
				dropout=self.dropout if self.numLayers > 1 else 0
			)
			
			# Couche de sortie après LSTM
			self.outputLayer = nn.Linear(self.hiddenDim, self.outputDim)
		else:
			# Réseaux feed-forward standard
			# Première couche cachée
			self.layers.append(nn.Linear(self.inputDim, self.hiddenDim))
			
			# Couches cachées intermédiaires
			for _ in range(self.numLayers - 1):
				self.layers.append(nn.Linear(self.hiddenDim, self.hiddenDim))
				
			# Couche de sortie
			self.layers.append(nn.Linear(self.hiddenDim, self.outputDim))
			
		# Couche de dropout
		self.dropoutLayer = nn.Dropout(self.dropout)
		
		# Mécanisme d'attention (si activé)
		if self.useAttention:
			self.attentionQuery = nn.Linear(self.hiddenDim, self.hiddenDim)
			self.attentionKey = nn.Linear(self.inputDim, self.hiddenDim)
			self.attentionValue = nn.Linear(self.inputDim, self.hiddenDim)
	
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
					
			elif isinstance(module, nn.LSTM):
				# Initialisation des poids LSTM
				for name, param in module.named_parameters():
					if "weight" in name:
						nn.init.normal_(param, mean=0.0, std=self.initStd)
					elif "bias" in name:
						nn.init.zeros_(param)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Propagation avant dans le réseau neuronal.
		
		Args:
			x: Tensor d'entrée (observations)
			
		Returns:
			Tensor de sortie (actions)
		"""
		# Vérifier la dimension d'entrée
		if x.dim() == 1:
			# Ajouter une dimension de batch si nécessaire
			x = x.unsqueeze(0)
			
		# Mettre à jour le compteur d'inférences
		self.inferenceCount += 1
		
		# Mettre à jour l'horloge interne
		self.internalClock += 0.01  # Incrément fixe pour simplifier
		
		if self.useLSTM:
			# Ajouter une dimension de séquence si nécessaire (batch, seq, features)
			if x.dim() == 2:
				x = x.unsqueeze(1)
				
			# Passage dans le LSTM
			if self.hiddenState is None:
				output, self.hiddenState = self.lstm(x)
			else:
				# Vérifier si les dimensions du hidden state correspondent à l'entrée
				batchSize = x.size(0)
				if self.hiddenState[0].size(1) != batchSize:
					# Réinitialiser l'état caché si la taille du batch change
					self.hiddenState = None
					output, self.hiddenState = self.lstm(x)
				else:
					output, self.hiddenState = self.lstm(x, self.hiddenState)
					
			# Prendre la dernière sortie de la séquence
			output = output[:, -1, :]
			
			# Appliquer dropout
			output = self.dropoutLayer(output)
			
			# Couche de sortie
			output = self.outputLayer(output)
		else:
			# Réseau feed-forward
			output = x
			
			# Si l'attention est activée, l'appliquer sur l'entrée
			if self.useAttention:
				# Calculer les composantes d'attention
				query = self.attentionQuery(torch.zeros(x.size(0), self.hiddenDim).to(x.device))
				key = self.attentionKey(x)
				value = self.attentionValue(x)
				
				# Scores d'attention: produit scalaire query-key
				attentionScores = torch.matmul(query.unsqueeze(1), key.unsqueeze(2)).squeeze(2)
				attentionWeights = F.softmax(attentionScores, dim=1)
				
				# Appliquer les poids d'attention
				attentionOutput = torch.matmul(attentionWeights.unsqueeze(1), value.unsqueeze(2)).squeeze(2)
				
				# Combiner avec l'entrée originale
				output = output + attentionOutput
				
			# Passage à travers les couches cachées
			for i, layer in enumerate(self.layers[:-1]):
				output = layer(output)
				output = self.activation(output)
				
				# Appliquer dropout sauf à la dernière couche
				output = self.dropoutLayer(output)
				
			# Couche de sortie (dernière couche)
			output = self.layers[-1](output)
			
		# Limiter les sorties entre -1 et 1 (pour actions normalisées)
		output = torch.tanh(output)
		
		return output
	
	def resetState(self) -> None:
		"""
		Réinitialise l'état interne du réseau (important pour les réseaux récurrents).
		"""
		self.hiddenState = None
		self.internalClock = 0.0
	
	def selectAction(self, observation: torch.Tensor) -> torch.Tensor:
		"""
		Sélectionne une action en fonction de l'observation actuelle.
		
		Args:
			observation: Tensor d'entrée (observations)
			
		Returns:
			Tensor de sortie (actions sélectionnées)
		"""
		# Désactiver le calcul de gradient pour l'inférence
		with torch.no_grad():
			return self.forward(observation)
	
	def copyFromOther(self, otherBrain: 'CreatureBrain', tau: float = 1.0) -> None:
		"""
		Copie les poids d'un autre cerveau (utile pour les algorithmes comme DQN).
		
		Args:
			otherBrain: Autre instance de CreatureBrain
			tau: Facteur de mélange (1.0 = copie complète, 0.0 = pas de copie)
		"""
		if tau <= 0.0:
			return
			
		# Copier les paramètres pondérés par tau
		for targetParam, sourceParam in zip(self.parameters(), otherBrain.parameters()):
			targetParam.data.copy_((1.0 - tau) * targetParam.data + tau * sourceParam.data)
	
	def mutate(self, mutationRate: float = 0.1, mutationStrength: float = 0.1) -> None:
		"""
		Applique des mutations aléatoires aux poids du réseau.
		
		Args:
			mutationRate: Probabilité qu'un poids soit muté
			mutationStrength: Amplitude des mutations
		"""
		for param in self.parameters():
			# Créer un masque de mutation aléatoire
			mutation_mask = (torch.rand_like(param) < mutationRate).float()
			
			# Générer des perturbations aléatoires
			noise = torch.randn_like(param) * mutationStrength
			
			# Appliquer les mutations selon le masque
			param.data += noise * mutation_mask
	
	def getNumParameters(self) -> int:
		"""
		Retourne le nombre total de paramètres dans le réseau.
		
		Returns:
			Nombre de paramètres
		"""
		return sum(p.numel() for p in self.parameters())
	
	def getCurrentComplexity(self) -> float:
		"""
		Calcule une mesure de complexité du réseau basée sur sa structure.
		
		Returns:
			Score de complexité
		"""
		# Complexité basée sur le nombre de couches et leur taille
		complexityScore = self.numLayers * self.hiddenDim / 100.0
		
		# Bonus pour les fonctionnalités avancées
		if self.useLSTM:
			complexityScore *= 1.5
		if self.useAttention:
			complexityScore *= 1.2
			
		return complexityScore
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet CreatureBrain en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du cerveau de la créature
		"""
		# Sérialiser les paramètres du réseau
		state_dict = {k: v.cpu().numpy() for k, v in self.state_dict().items()}
		
		return {
			"id": self.id,
			"inputDim": self.inputDim,
			"outputDim": self.outputDim,
			"hiddenDim": self.hiddenDim,
			"numLayers": self.numLayers,
			"activationName": self.activationName,
			"useLSTM": self.useLSTM,
			"useAttention": self.useAttention,
			"dropout": self.dropout,
			"initStd": self.initStd,
			"genomeId": self.genomeId,
			"state_dict": state_dict,
			"inferenceCount": self.inferenceCount,
			"internalClock": self.internalClock
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'CreatureBrain':
		"""
		Crée une instance de CreatureBrain à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du cerveau
			
		Returns:
			Instance de CreatureBrain reconstruite
		"""
		# Créer le réseau avec les hyperparamètres
		brain = cls(
			inputDim=data["inputDim"],
			outputDim=data["outputDim"],
			hiddenDim=data["hiddenDim"],
			numLayers=data["numLayers"],
			activation=data["activationName"],
			useLSTM=data["useLSTM"],
			useAttention=data["useAttention"],
			dropout=data["dropout"],
			initStd=data["initStd"],
			genomeId=data["genomeId"]
		)
		
		# Restaurer l'ID
		brain.id = data["id"]
		
		# Convertir les paramètres du réseau de numpy à torch
		state_dict = {k: torch.tensor(v) for k, v in data["state_dict"].items()}
		brain.load_state_dict(state_dict)
		
		# Restaurer les compteurs
		brain.inferenceCount = data["inferenceCount"]
		brain.internalClock = data["internalClock"]
		
		return brain