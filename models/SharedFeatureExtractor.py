import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class SharedFeatureExtractor(nn.Module):
	"""
	Extracteur de caractéristiques partagé pour l'agent PPO.
	Traite les observations des satellites et de l'environnement global pour extraire des caractéristiques.
	"""
	
	def __init__(
		self,
		satelliteObsDim: int,
		globalObsDim: int,
		satelliteFeaturesDim: int = 128,
		globalFeaturesDim: int = 64,
		numSatellites: int = 30
	):
		"""
		Initialise l'extracteur de caractéristiques.
		
		Args:
			satelliteObsDim: Dimension de l'observation par satellite
			globalObsDim: Dimension de l'observation globale
			satelliteFeaturesDim: Dimension des caractéristiques par satellite à extraire
			globalFeaturesDim: Dimension des caractéristiques globales à extraire
			numSatellites: Nombre de satellites dans la constellation
		"""
		super(SharedFeatureExtractor, self).__init__()
		
		self.satelliteObsDim = satelliteObsDim
		self.globalObsDim = globalObsDim
		self.satelliteFeaturesDim = satelliteFeaturesDim
		self.globalFeaturesDim = globalFeaturesDim
		self.numSatellites = numSatellites
		
		# Réseau d'extraction de caractéristiques pour chaque satellite
		self.satelliteEncoder = nn.Sequential(
			nn.Linear(satelliteObsDim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, satelliteFeaturesDim)
		)
		
		# Réseau d'extraction de caractéristiques pour l'observation globale
		self.globalEncoder = nn.Sequential(
			nn.Linear(globalObsDim, 128),
			nn.ReLU(),
			nn.Linear(128, 128),
			nn.ReLU(),
			nn.Linear(128, globalFeaturesDim)
		)
		
		# Initialisation des poids
		self._initWeights()
	
	def _initWeights(self) -> None:
		"""
		Initialise les poids du réseau.
		"""
		# Initialisation du réseau d'extraction pour les satellites
		for i in range(0, len(self.satelliteEncoder), 2):
			if isinstance(self.satelliteEncoder[i], nn.Linear):
				nn.init.orthogonal_(self.satelliteEncoder[i].weight, gain=np.sqrt(2))
				nn.init.constant_(self.satelliteEncoder[i].bias, 0.0)
		
		# Initialisation du réseau d'extraction pour l'observation globale
		for i in range(0, len(self.globalEncoder), 2):
			if isinstance(self.globalEncoder[i], nn.Linear):
				nn.init.orthogonal_(self.globalEncoder[i].weight, gain=np.sqrt(2))
				nn.init.constant_(self.globalEncoder[i].bias, 0.0)
	
	def forward(self, observations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Propagation avant.
		
		Args:
			observations: Dictionnaire d'observations avec des clés 'satellite_i' pour chaque satellite
						et 'global' pour l'observation globale
						
		Returns:
			Tuple (satellite_features, global_features) où satellite_features est de forme
			(batch_size, num_satellites, satellite_features_dim) et global_features est de forme
			(batch_size, global_features_dim)
		"""
		batchSize = list(observations.values())[0].size(0)
		
		# Extraire les caractéristiques pour chaque satellite
		satelliteFeatures = []
		
		for i in range(self.numSatellites):
			if f"satellite_{i}" in observations:
				satelliteObs = observations[f"satellite_{i}"]
				features = self.satelliteEncoder(satelliteObs)
				satelliteFeatures.append(features)
			else:
				# Si l'observation pour ce satellite est manquante, utiliser un tenseur de zéros
				features = torch.zeros(batchSize, self.satelliteFeaturesDim, device=observations['global'].device)
				satelliteFeatures.append(features)
		
		# Empiler les caractéristiques de tous les satellites
		satelliteFeatures = torch.stack(satelliteFeatures, dim=1)
		
		# Extraire les caractéristiques globales
		globalFeatures = self.globalEncoder(observations['global'])
		
		return satelliteFeatures, globalFeatures