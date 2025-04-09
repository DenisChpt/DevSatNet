import torch
import numpy as np
from typing import Dict, List, Tuple, Generator, Optional, Any
import gym
from gym import spaces

class MemoryBuffer:
	"""
	Tampon de mémoire pour stocker les transitions d'expérience pour l'apprentissage par renforcement.
	"""
	
	def __init__(
		self,
		bufferSize: int,
		observationSpace: spaces.Dict,
		actionSpace: spaces.Box,
		device: torch.device,
		gamma: float = 0.99,
		lambd: float = 0.95
	):
		"""
		Initialise le tampon de mémoire.
		
		Args:
			bufferSize: Taille du tampon (nombre de transitions)
			observationSpace: Espace d'observation
			actionSpace: Espace d'action
			device: Dispositif pour les tenseurs PyTorch
			gamma: Facteur d'actualisation pour les récompenses
			lambd: Paramètre lambda pour l'avantage généralisé
		"""
		self.bufferSize = bufferSize
		self.observationSpace = observationSpace
		self.actionSpace = actionSpace
		self.device = device
		self.gamma = gamma
		self.lambd = lambd
		
		# Trouver le nombre de satellites et les dimensions des observations
		self.numSatellites = 0
		self.observationsDim = {}
		
		for key, space in observationSpace.spaces.items():
			if key.startswith("satellite_"):
				self.numSatellites += 1
				dim = int(np.prod(space.shape))
				self.observationsDim[key] = dim
			elif key == "global":
				dim = int(np.prod(space.shape))
				self.observationsDim[key] = dim
		
		# Dimension des actions
		self.actionDim = int(np.prod(actionSpace.shape[1:]))
		
		# Initialiser les tampons
		self.clear()
	
	def clear(self) -> None:
		"""
		Vide le tampon.
		"""
		# Tampon des observations
		self.observations: Dict[str, List[np.ndarray]] = {key: [] for key in self.observationsDim.keys()}
		
		# Tampon des actions, valeurs, récompenses, etc.
		self.actions: List[np.ndarray] = []
		self.rewards: List[float] = []
		self.values: Optional[torch.Tensor] = None
		self.returns: Optional[torch.Tensor] = None
		self.advantages: Optional[torch.Tensor] = None
		self.logProbs: List[float] = []
		self.dones: List[bool] = []
		
		# Position actuelle dans le tampon
		self.pos = 0
		self.full = False
	
	def add(
		self,
		observation: Dict[str, np.ndarray],
		action: np.ndarray,
		reward: float,
		nextObservation: Dict[str, np.ndarray],
		done: bool,
		value: float,
		actionLogProb: float
	) -> None:
		"""
		Ajoute une transition au tampon.
		
		Args:
			observation: Observation actuelle
			action: Action prise
			reward: Récompense reçue
			nextObservation: Observation suivante
			done: Indicateur de fin d'épisode
			value: Valeur d'état prédite
			actionLogProb: Log probabilité de l'action
		"""
		# Ajouter l'observation
		for key, obs in observation.items():
			if key in self.observations:
				if self.full and self.pos < len(self.observations[key]):
					self.observations[key][self.pos] = obs
				else:
					self.observations[key].append(obs)
		
		# Ajouter l'action, la récompense, etc.
		if self.full and self.pos < len(self.actions):
			self.actions[self.pos] = action
			self.rewards[self.pos] = reward
			self.logProbs[self.pos] = actionLogProb
			self.dones[self.pos] = done
		else:
			self.actions.append(action)
			self.rewards.append(reward)
			self.logProbs.append(actionLogProb)
			self.dones.append(done)
		
		# Mettre à jour la position
		self.pos = (self.pos + 1) % self.bufferSize
		
		# Si on a fait un tour complet, le tampon est plein
		if self.pos == 0:
			self.full = True
	
	def computeReturnsAndAdvantages(self, nextValue: Optional[float] = 0.0, useTorch: bool = True) -> None:
		"""
		Calcule les retours et les avantages généralisés pour toutes les transitions dans le tampon.
		
		Args:
			nextValue: Valeur de l'état suivant après la dernière transition
			useTorch: Si vrai, utilise PyTorch pour les calculs
		"""
		# Déterminer le nombre effectif de transitions dans le tampon
		bufferSize = self.bufferSize if self.full else self.pos
		
		# Créer des tenseurs pour les valeurs, récompenses, log probabilités et drapeaux de fin
		if useTorch:
			rewards = torch.tensor(self.rewards[:bufferSize], dtype=torch.float32, device=self.device)
			values = torch.tensor([0.0] * bufferSize, dtype=torch.float32, device=self.device)
			dones = torch.tensor(self.dones[:bufferSize], dtype=torch.float32, device=self.device)
			logProbs = torch.tensor(self.logProbs[:bufferSize], dtype=torch.float32, device=self.device)
			returns = torch.zeros_like(rewards)
			advantages = torch.zeros_like(rewards)
			
			lastGae = 0.0
			
			# Remplir le tenseur des valeurs
			for i in range(bufferSize):
				values[i] = self.values[i] if self.values is not None else 0.0
			
			# Calcul récursif des retours et avantages (en partant de la fin)
			for t in reversed(range(bufferSize)):
				if t == bufferSize - 1:
					nextNonTerminal = 1.0 - float(self.dones[t])
					nextValue = nextValue
				else:
					nextNonTerminal = 1.0 - float(self.dones[t])
					nextValue = values[t + 1]
				
				# Calcul de l'erreur temporelle delta
				delta = rewards[t] + self.gamma * nextValue * nextNonTerminal - values[t]
				
				# Calcul de l'avantage généralisé
				lastGae = delta + self.gamma * self.lambd * nextNonTerminal * lastGae
				advantages[t] = lastGae
			
			# Calcul des retours (valeur + avantage)
			returns = advantages + values
			
			# Stocker les résultats
			self.returns = returns
			self.advantages = advantages
			self.values = values
			self.logProbs = logProbs
		else:
			# Implémentation NumPy pour comparaison (non utilisée par défaut)
			rewards = np.array(self.rewards[:bufferSize], dtype=np.float32)
			values = np.zeros(bufferSize, dtype=np.float32)
			dones = np.array(self.dones[:bufferSize], dtype=np.float32)
			returns = np.zeros_like(rewards)
			advantages = np.zeros_like(rewards)
			
			lastGae = 0.0
			
			# Remplir le tableau des valeurs
			for i in range(bufferSize):
				values[i] = self.values[i] if self.values is not None else 0.0
			
			# Calcul récursif des retours et avantages (en partant de la fin)
			for t in reversed(range(bufferSize)):
				if t == bufferSize - 1:
					nextNonTerminal = 1.0 - float(self.dones[t])
					nextValue = nextValue
				else:
					nextNonTerminal = 1.0 - float(self.dones[t])
					nextValue = values[t + 1]
				
				# Calcul de l'erreur temporelle delta
				delta = rewards[t] + self.gamma * nextValue * nextNonTerminal - values[t]
				
				# Calcul de l'avantage généralisé
				lastGae = delta + self.gamma * self.lambd * nextNonTerminal * lastGae
				advantages[t] = lastGae
			
			# Calcul des retours (valeur + avantage)
			returns = advantages + values
			
			# Convertir en tenseurs PyTorch et stocker les résultats
			self.returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
			self.advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
			self.values = torch.tensor(values, dtype=torch.float32, device=self.device)
			self.logProbs = torch.tensor(self.logProbs[:bufferSize], dtype=torch.float32, device=self.device)
	
	def getBatchGenerator(self, batchSize: int) -> Generator[Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
		"""
		Génère des lots de transitions pour l'entraînement.
		
		Args:
			batchSize: Taille des lots
			
		Returns:
			Générateur de lots (observations, actions, valeurs, log_probs, advantages, returns)
		"""
		# Déterminer le nombre effectif de transitions dans le tampon
		bufferSize = self.bufferSize if self.full else self.pos
		
		# Générer des indices aléatoires
		indices = np.random.permutation(bufferSize)
		
		# Générer des lots
		startIdx = 0
		
		while startIdx < bufferSize:
			# Indices pour ce lot
			batchIndices = indices[startIdx:min(startIdx + batchSize, bufferSize)]
			
			# Créer des tenseurs pour les observations
			batchObservations = {}
			
			for key, obsList in self.observations.items():
				batchObservations[key] = torch.tensor(
					np.array([obsList[i] for i in batchIndices]),
					dtype=torch.float32, device=self.device
				)
			
			# Créer des tenseurs pour les actions, valeurs, etc.
			batchActions = torch.tensor(
				np.array([self.actions[i] for i in batchIndices]),
				dtype=torch.float32, device=self.device
			)
			
			batchValues = self.values[batchIndices]
			batchLogProbs = self.logProbs[batchIndices]
			batchAdvantages = self.advantages[batchIndices]
			batchReturns = self.returns[batchIndices]
			
			# Normaliser les avantages dans ce lot
			batchAdvantages = (batchAdvantages - batchAdvantages.mean()) / (batchAdvantages.std() + 1e-8)
			
			yield batchObservations, batchActions, batchValues, batchLogProbs, batchAdvantages, batchReturns
			
			# Passer au lot suivant
			startIdx += batchSize
	
	def isReady(self) -> bool:
		"""
		Vérifie si le tampon est prêt pour l'entraînement (au moins à moitié plein).
		
		Returns:
			True si le tampon est prêt, False sinon
		"""
		return self.full or self.pos >= self.bufferSize // 2