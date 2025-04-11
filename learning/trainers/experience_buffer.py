# learning/trainers/experience_buffer.py
from typing import Dict, List, Tuple, Any, NamedTuple
import numpy as np
from collections import deque
import random


class Experience(NamedTuple):
	"""Structure pour stocker une seule transition d'expérience."""
	state: np.ndarray
	action: np.ndarray
	reward: float
	nextState: np.ndarray
	done: bool
	info: Dict[str, Any]


class ExperienceBatch(NamedTuple):
	"""Structure pour stocker un lot d'expériences."""
	states: np.ndarray
	actions: np.ndarray
	rewards: np.ndarray
	nextStates: np.ndarray
	dones: np.ndarray
	infos: List[Dict[str, Any]]


class ExperienceBuffer:
	"""
	Buffer pour stocker et échantillonner des expériences d'interactions avec l'environnement.
	"""
	
	def __init__(self, capacity: int = 10000):
		"""
		Initialise le buffer d'expérience.
		
		Args:
			capacity: Capacité maximale du buffer
		"""
		self.buffer = deque(maxlen=capacity)
		self.capacity = capacity
	
	def add(
		self,
		state: np.ndarray,
		action: np.ndarray,
		reward: float,
		nextState: np.ndarray,
		done: bool,
		info: Dict[str, Any] = {}
	) -> None:
		"""
		Ajoute une expérience au buffer.
		
		Args:
			state: État observé
			action: Action effectuée
			reward: Récompense reçue
			nextState: État suivant
			done: Indicateur de fin d'épisode
			info: Informations supplémentaires
		"""
		experience = Experience(state, action, reward, nextState, float(done), info)
		self.buffer.append(experience)
	
	def sample(self, batchSize: int) -> ExperienceBatch:
		"""
		Échantillonne un lot d'expériences du buffer.
		
		Args:
			batchSize: Taille du lot à échantillonner
			
		Returns:
			Lot d'expériences
		"""
		# Limiter la taille du lot à la taille du buffer
		batchSize = min(batchSize, len(self.buffer))
		
		# Échantillonner des expériences aléatoires
		experiences = random.sample(self.buffer, batchSize)
		
		# Convertir les expériences en batch
		states = np.array([exp.state for exp in experiences])
		actions = np.array([exp.action for exp in experiences])
		rewards = np.array([exp.reward for exp in experiences])
		nextStates = np.array([exp.nextState for exp in experiences])
		dones = np.array([exp.done for exp in experiences])
		infos = [exp.info for exp in experiences]
		
		return ExperienceBatch(states, actions, rewards, nextStates, dones, infos)
	
	def sampleSequential(self, batchSize: int) -> ExperienceBatch:
		"""
		Échantillonne un lot d'expériences séquentielles du buffer.
		Utile pour les algorithmes qui nécessitent des transitions consécutives.
		
		Args:
			batchSize: Taille du lot à échantillonner
			
		Returns:
			Lot d'expériences séquentielles
		"""
		# Limiter la taille du lot à la taille du buffer
		batchSize = min(batchSize, len(self.buffer))
		
		# Choisir un point de départ aléatoire
		startIdx = random.randint(0, len(self.buffer) - batchSize)
		
		# Extraire les expériences séquentielles
		experiences = list(self.buffer)[startIdx:startIdx + batchSize]
		
		# Convertir les expériences en batch
		states = np.array([exp.state for exp in experiences])
		actions = np.array([exp.action for exp in experiences])
		rewards = np.array([exp.reward for exp in experiences])
		nextStates = np.array([exp.nextState for exp in experiences])
		dones = np.array([exp.done for exp in experiences])
		infos = [exp.info for exp in experiences]
		
		return ExperienceBatch(states, actions, rewards, nextStates, dones, infos)
	
	def clear(self) -> None:
		"""Vide le buffer."""
		self.buffer.clear()
	
	def __len__(self) -> int:
		"""Retourne le nombre d'expériences dans le buffer."""
		return len(self.buffer)
	
	def isFull(self) -> bool:
		"""Vérifie si le buffer est plein."""
		return len(self.buffer) == self.capacity


class ReplayBuffer(ExperienceBuffer):
	"""
	Extension du buffer d'expérience avec des fonctionnalités supplémentaires
	pour l'apprentissage par experience replay.
	"""
	
	def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
		"""
		Initialise le buffer de replay.
		
		Args:
			capacity: Capacité maximale du buffer
			alpha: Exposant pour la priorisation (0 = échantillonnage uniforme)
			beta: Exposant pour la correction du biais (0 = pas de correction)
		"""
		super().__init__(capacity)
		self.alpha = alpha
		self.beta = beta
		self.priorities = deque(maxlen=capacity)
		self.epsilon = 1e-6  # Priorité minimale pour éviter les probabilités nulles
	
	def add(
		self,
		state: np.ndarray,
		action: np.ndarray,
		reward: float,
		nextState: np.ndarray,
		done: bool,
		info: Dict[str, Any] = {},
		priority: float = None
	) -> None:
		"""
		Ajoute une expérience au buffer avec une priorité optionnelle.
		
		Args:
			state: État observé
			action: Action effectuée
			reward: Récompense reçue
			nextState: État suivant
			done: Indicateur de fin d'épisode
			info: Informations supplémentaires
			priority: Priorité de l'expérience (si None, priorité maximale)
		"""
		experience = Experience(state, action, reward, nextState, float(done), info)
		self.buffer.append(experience)
		
		# Utiliser la priorité maximale actuelle ou 1.0 si le buffer est vide
		if priority is None:
			priority = max(self.priorities, default=1.0)
			
		self.priorities.append(priority)
	
	def updatePriorities(self, indices: List[int], priorities: List[float]) -> None:
		"""
		Met à jour les priorités pour certaines expériences.
		
		Args:
			indices: Indices des expériences à mettre à jour
			priorities: Nouvelles priorités
		"""
		for idx, priority in zip(indices, priorities):
			if 0 <= idx < len(self.priorities):
				self.priorities[idx] = priority + self.epsilon
	
	def samplePrioritized(self, batchSize: int, beta: float = None) -> Tuple[ExperienceBatch, np.ndarray, np.ndarray]:
		"""
		Échantillonne un lot d'expériences selon leurs priorités.
		
		Args:
			batchSize: Taille du lot à échantillonner
			beta: Exposant pour la correction du biais (si None, utilise self.beta)
			
		Returns:
			Tuple contenant le lot d'expériences, les indices échantillonnés et les poids d'importance-sampling
		"""
		if len(self.buffer) == 0:
			return None, np.array([]), np.array([])
			
		# Limiter la taille du lot à la taille du buffer
		batchSize = min(batchSize, len(self.buffer))
		
		if beta is None:
			beta = self.beta
			
		# Calculer les probabilités d'échantillonnage
		priorities = np.array(self.priorities)
		probabilities = priorities ** self.alpha
		probabilities /= probabilities.sum()
		
		# Échantillonner les indices
		indices = np.random.choice(len(self.buffer), batchSize, replace=False, p=probabilities)
		
		# Calculer les poids d'importance-sampling
		weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
		weights /= weights.max()  # Normaliser les poids
		
		# Extraire les expériences
		experiences = [self.buffer[idx] for idx in indices]
		
		# Convertir les expériences en batch
		states = np.array([exp.state for exp in experiences])
		actions = np.array([exp.action for exp in experiences])
		rewards = np.array([exp.reward for exp in experiences])
		nextStates = np.array([exp.nextState for exp in experiences])
		dones = np.array([exp.done for exp in experiences])
		infos = [exp.info for exp in experiences]
		
		batch = ExperienceBatch(states, actions, rewards, nextStates, dones, infos)
		
		return batch, indices, weights
	
	def clear(self) -> None:
		"""Vide le buffer et les priorités."""
		super().clear()
		self.priorities.clear()