import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class CriticNetwork(nn.Module):
	"""
	Réseau critique pour l'agent PPO qui prédit la valeur d'état.
	Prend en entrée les caractéristiques extraites des satellites et de l'environnement global.
	"""
	
	def __init__(
		self,
		satelliteFeaturesDim: int,
		globalFeaturesDim: int,
		hiddenDims: List[int] = [256, 256]
	):
		"""
		Initialise le réseau critique.
		
		Args:
			satelliteFeaturesDim: Dimension des caractéristiques par satellite
			globalFeaturesDim: Dimension des caractéristiques globales
			hiddenDims: Dimensions des couches cachées
		"""
		super(CriticNetwork, self).__init__()
		
		self.satelliteFeaturesDim = satelliteFeaturesDim
		self.globalFeaturesDim = globalFeaturesDim
		
		# Couche d'agrégation des caractéristiques des satellites par attention
		self.attentionLayer = nn.Sequential(
			nn.Linear(satelliteFeaturesDim, 64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)
		
		# Couche de fusion des caractéristiques
		self.fusionLayer = nn.Sequential(
			nn.Linear(satelliteFeaturesDim + globalFeaturesDim, hiddenDims[0]),
			nn.ReLU()
		)
		
		# Couches cachées
		self.hiddenLayers = nn.ModuleList()
		for i in range(len(hiddenDims) - 1):
			self.hiddenLayers.append(nn.Sequential(
				nn.Linear(hiddenDims[i], hiddenDims[i + 1]),
				nn.ReLU()
			))
		
		# Couche de sortie pour la valeur
		self.valueLayer = nn.Linear(hiddenDims[-1], 1)
		
		# Initialisation des poids
		self._initWeights()
	
	def _initWeights(self) -> None:
		"""
		Initialise les poids du réseau.
		"""
		# Initialisation de la couche d'attention
		nn.init.orthogonal_(self.attentionLayer[0].weight, gain=np.sqrt(2))
		nn.init.constant_(self.attentionLayer[0].bias, 0.0)
		nn.init.orthogonal_(self.attentionLayer[2].weight, gain=np.sqrt(2))
		nn.init.constant_(self.attentionLayer[2].bias, 0.0)
		
		# Initialisation de la couche de fusion
		nn.init.orthogonal_(self.fusionLayer[0].weight, gain=np.sqrt(2))
		nn.init.constant_(self.fusionLayer[0].bias, 0.0)
		
		# Initialisation des couches cachées
		for layer in self.hiddenLayers:
			nn.init.orthogonal_(layer[0].weight, gain=np.sqrt(2))
			nn.init.constant_(layer[0].bias, 0.0)
		
		# Initialisation de la couche de sortie
		nn.init.orthogonal_(self.valueLayer.weight, gain=1.0)
		nn.init.constant_(self.valueLayer.bias, 0.0)
	
	def forward(self, satelliteFeatures: torch.Tensor, globalFeatures: torch.Tensor) -> torch.Tensor:
		"""
		Propagation avant.
		
		Args:
			satelliteFeatures: Caractéristiques des satellites, de forme (batch_size, num_satellites, satellite_features_dim)
			globalFeatures: Caractéristiques globales, de forme (batch_size, global_features_dim)
			
		Returns:
			Valeur d'état de forme (batch_size, 1)
		"""
		batchSize = satelliteFeatures.size(0)
		numSatellites = satelliteFeatures.size(1)
		
		# Appliquer l'attention pour agréger les caractéristiques des satellites
		# Applatir pour appliquer l'attention à chaque satellite individuellement
		satelliteFeaturesFlatted = satelliteFeatures.view(-1, self.satelliteFeaturesDim)
		attentionScores = self.attentionLayer(satelliteFeaturesFlatted).view(batchSize, numSatellites, 1)
		
		# Normaliser les scores d'attention (softmax)
		attentionWeights = F.softmax(attentionScores, dim=1)
		
		# Agréger les caractéristiques des satellites pondérées par l'attention
		satelliteFeaturesAggregated = torch.sum(satelliteFeatures * attentionWeights, dim=1)
		
		# Concaténer les caractéristiques agrégées des satellites avec les caractéristiques globales
		combinedFeatures = torch.cat([satelliteFeaturesAggregated, globalFeatures], dim=1)
		
		# Propagation à travers les couches
		x = self.fusionLayer(combinedFeatures)
		
		for layer in self.hiddenLayers:
			x = layer(x)
		
		# Couche de sortie pour la valeur
		value = self.valueLayer(x)
		
		return value