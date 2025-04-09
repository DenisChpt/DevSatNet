import json
import os
from typing import List, Dict, Any, Optional

class HyperParameters:
	"""
	Classe pour gérer les hyperparamètres d'entraînement.
	"""
	
	def __init__(
		self,
		# Paramètres du modèle
		satelliteFeaturesDim: int = 128,
		globalFeaturesDim: int = 64,
		hiddenDims: List[int] = [256, 256],
		
		# Paramètres d'optimisation
		learningRate: float = 3e-4,
		batchSize: int = 64,
		numEpochs: int = 10,
		clipRange: float = 0.2,
		valueLossCoef: float = 0.5,
		entropyCoef: float = 0.01,
		maxGradNorm: float = 0.5,
		targetKL: float = 0.01,
		
		# Paramètres de l'algorithme RL
		gamma: float = 0.99,
		lambd: float = 0.95,
		bufferSize: int = 2048,
		trainingFrequency: int = 2048
	):
		"""
		Initialise les hyperparamètres avec des valeurs par défaut.
		
		Args:
			satelliteFeaturesDim: Dimension des caractéristiques par satellite
			globalFeaturesDim: Dimension des caractéristiques globales
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
			trainingFrequency: Fréquence d'entraînement (en étapes de temps)
		"""
		# Paramètres du modèle
		self.satelliteFeaturesDim = satelliteFeaturesDim
		self.globalFeaturesDim = globalFeaturesDim
		self.hiddenDims = hiddenDims
		
		# Paramètres d'optimisation
		self.learningRate = learningRate
		self.batchSize = batchSize
		self.numEpochs = numEpochs
		self.clipRange = clipRange
		self.valueLossCoef = valueLossCoef
		self.entropyCoef = entropyCoef
		self.maxGradNorm = maxGradNorm
		self.targetKL = targetKL
		
		# Paramètres de l'algorithme RL
		self.gamma = gamma
		self.lambd = lambd
		self.bufferSize = bufferSize
		self.trainingFrequency = trainingFrequency
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit les hyperparamètres en dictionnaire.
		
		Returns:
			Dictionnaire des hyperparamètres
		"""
		return {
			"model": {
				"satellite_features_dim": self.satelliteFeaturesDim,
				"global_features_dim": self.globalFeaturesDim,
				"hidden_dims": self.hiddenDims
			},
			"optimization": {
				"learning_rate": self.learningRate,
				"batch_size": self.batchSize,
				"num_epochs": self.numEpochs,
				"clip_range": self.clipRange,
				"value_loss_coef": self.valueLossCoef,
				"entropy_coef": self.entropyCoef,
				"max_grad_norm": self.maxGradNorm,
				"target_kl": self.targetKL
			},
			"rl": {
				"gamma": self.gamma,
				"lambda": self.lambd,
				"buffer_size": self.bufferSize,
				"training_frequency": self.trainingFrequency
			}
		}
	
	def save(self, path: str) -> None:
		"""
		Enregistre les hyperparamètres dans un fichier JSON.
		
		Args:
			path: Chemin du fichier
		"""
		with open(path, 'w') as f:
			json.dump(self.toDict(), f, indent=4)
	
	@classmethod
	def fromDict(cls, config: Dict[str, Any]) -> 'HyperParameters':
		"""
		Crée une instance d'HyperParameters à partir d'un dictionnaire.
		
		Args:
			config: Dictionnaire de configuration
			
		Returns:
			Instance d'HyperParameters
		"""
		model_config = config.get("model", {})
		optimization_config = config.get("optimization", {})
		rl_config = config.get("rl", {})
		
		return cls(
			# Paramètres du modèle
			satelliteFeaturesDim=model_config.get("satellite_features_dim", 128),
			globalFeaturesDim=model_config.get("global_features_dim", 64),
			hiddenDims=model_config.get("hidden_dims", [256, 256]),
			
			# Paramètres d'optimisation
			learningRate=optimization_config.get("learning_rate", 3e-4),
			batchSize=optimization_config.get("batch_size", 64),
			numEpochs=optimization_config.get("num_epochs", 10),
			clipRange=optimization_config.get("clip_range", 0.2),
			valueLossCoef=optimization_config.get("value_loss_coef", 0.5),
			entropyCoef=optimization_config.get("entropy_coef", 0.01),
			maxGradNorm=optimization_config.get("max_grad_norm", 0.5),
			targetKL=optimization_config.get("target_kl", 0.01),
			
			# Paramètres de l'algorithme RL
			gamma=rl_config.get("gamma", 0.99),
			lambd=rl_config.get("lambda", 0.95),
			bufferSize=rl_config.get("buffer_size", 2048),
			trainingFrequency=rl_config.get("training_frequency", 2048)
		)
	
	@classmethod
	def fromFile(cls, path: str) -> 'HyperParameters':
		"""
		Crée une instance d'HyperParameters à partir d'un fichier JSON.
		
		Args:
			path: Chemin du fichier
			
		Returns:
			Instance d'HyperParameters
		"""
		if not os.path.exists(path):
			print(f"Le fichier {path} n'existe pas. Utilisation des valeurs par défaut.")
			return cls()
		
		try:
			with open(path, 'r') as f:
				config = json.load(f)
			
			# Extraire la section "hyperparameters" si elle existe
			if "hyperparameters" in config:
				config = config["hyperparameters"]
			
			return cls.fromDict(config)
		except Exception as e:
			print(f"Erreur lors du chargement des hyperparamètres: {e}")
			print("Utilisation des valeurs par défaut.")
			return cls()
	
	def __str__(self) -> str:
		"""
		Représentation sous forme de chaîne de caractères.
		
		Returns:
			Chaîne formatée
		"""
		return json.dumps(self.toDict(), indent=2)