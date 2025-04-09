import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ActorNetwork(nn.Module):
	"""
	Réseau d'acteur pour l'agent PPO qui prend en entrée les caractéristiques extraites 
	des satellites et de l'environnement global et prédit la distribution des actions.
	"""
	
	def __init__(
		self,
		satelliteFeaturesDim: int,
		globalFeaturesDim: int,
		actionDim: int,
		hiddenDims: List[int] = [256, 256],
		numSatellites: int = 30
	):
		"""
		Initialise le réseau d'acteur.
		
		Args:
			satelliteFeaturesDim: Dimension des caractéristiques par satellite
			globalFeaturesDim: Dimension des caractéristiques globales
			actionDim: Dimension de l'espace d'action par satellite
			hiddenDims: Dimensions des couches cachées
			numSatellites: Nombre de satellites dans la constellation
		"""
		super(ActorNetwork, self).__init__()
		
		self.satelliteFeaturesDim = satelliteFeaturesDim
		self.globalFeaturesDim = globalFeaturesDim
		self.actionDim = actionDim
		self.numSatellites = numSatellites
		
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
		
		# Couche de sortie pour la moyenne des actions
		self.meanLayer = nn.Linear(hiddenDims[-1], actionDim)
		
		# Couche de sortie pour le log écart-type des actions
		self.logStdLayer = nn.Linear(hiddenDims[-1], actionDim)
		
		# Initialisation des poids
		self._initWeights()
	
	def _initWeights(self) -> None:
		"""
		Initialise les poids du réseau.
		"""
		# Initialisation de la couche de fusion
		nn.init.orthogonal_(self.fusionLayer[0].weight, gain=np.sqrt(2))
		nn.init.constant_(self.fusionLayer[0].bias, 0.0)
		
		# Initialisation des couches cachées
		for layer in self.hiddenLayers:
			nn.init.orthogonal_(layer[0].weight, gain=np.sqrt(2))
			nn.init.constant_(layer[0].bias, 0.0)
		
		# Initialisation de la couche de sortie pour la moyenne
		nn.init.orthogonal_(self.meanLayer.weight, gain=0.01)
		nn.init.constant_(self.meanLayer.bias, 0.0)
		
		# Initialisation de la couche de sortie pour le log écart-type
		nn.init.orthogonal_(self.logStdLayer.weight, gain=0.01)
		nn.init.constant_(self.logStdLayer.bias, 0.0)
	
	def forward(self, satelliteFeatures: torch.Tensor, globalFeatures: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Propagation avant.
		
		Args:
			satelliteFeatures: Caractéristiques des satellites, de forme (batch_size, num_satellites, satellite_features_dim)
			globalFeatures: Caractéristiques globales, de forme (batch_size, global_features_dim)
			
		Returns:
			Tuple (action_mean, action_log_std) où action_mean et action_log_std sont de forme
			(batch_size, num_satellites, action_dim)
		"""
		batchSize = satelliteFeatures.size(0)
		
		# Répéter les caractéristiques globales pour chaque satellite
		globalFeaturesExpanded = globalFeatures.unsqueeze(1).expand(-1, self.numSatellites, -1)
		
		# Concaténer les caractéristiques des satellites avec les caractéristiques globales
		combinedFeatures = torch.cat([satelliteFeatures, globalFeaturesExpanded], dim=2)
		
		# Applatir pour traiter chaque satellite individuellement
		combinedFeaturesFlat = combinedFeatures.view(-1, self.satelliteFeaturesDim + self.globalFeaturesDim)
		
		# Propagation à travers les couches
		x = self.fusionLayer(combinedFeaturesFlat)
		
		for layer in self.hiddenLayers:
			x = layer(x)
		
		# Calculer la moyenne et le log écart-type des actions
		actionMean = self.meanLayer(x)
		actionLogStd = self.logStdLayer(x)
		
		# Limiter le log écart-type pour la stabilité
		actionLogStd = torch.clamp(actionLogStd, -20, 2)
		
		# Reformer pour obtenir (batch_size, num_satellites, action_dim)
		actionMean = actionMean.view(batchSize, self.numSatellites, self.actionDim)
		actionLogStd = actionLogStd.view(batchSize, self.numSatellites, self.actionDim)
		
		return actionMean, actionLogStd


# Pour éviter les erreurs d'importation
import numpy as np