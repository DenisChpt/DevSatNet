import numpy as np
import torch
from typing import List, Dict, Union, Any

def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	"""
	Calcule la variance expliquée.
	
	var_explained = 1 - var(y_true - y_pred) / var(y_true)
	
	Args:
		y_true: Valeurs cibles
		y_pred: Valeurs prédites
		
	Returns:
		Variance expliquée (entre -inf et 1.0)
	"""
	var_y = np.var(y_true)
	if var_y == 0:
		return 0.0
	return 1 - np.var(y_true - y_pred) / var_y

def compute_gae(
	rewards: np.ndarray,
	values: np.ndarray,
	dones: np.ndarray,
	gamma: float,
	lambd: float,
	next_value: float = 0.0
) -> np.ndarray:
	"""
	Calcule l'estimation d'avantage généralisé (GAE).
	
	Args:
		rewards: Tableau des récompenses [T]
		values: Tableau des valeurs d'état [T]
		dones: Tableau des indicateurs de fin d'épisode [T]
		gamma: Facteur d'actualisation
		lambd: Paramètre lambda pour GAE
		next_value: Valeur de l'état suivant après la dernière transition
		
	Returns:
		Avantages généralisés [T]
	"""
	T = len(rewards)
	advantages = np.zeros_like(rewards)
	last_gae = 0.0
	
	# Ajouter la valeur suivante aux tableaux pour faciliter le calcul
	values_extended = np.append(values, next_value)
	
	# Calcul récursif des avantages (en partant de la fin)
	for t in reversed(range(T)):
		# Si c'est un état terminal, la valeur suivante est 0
		next_non_terminal = 1.0 - dones[t]
		
		# Calcul de l'erreur temporelle delta
		delta = rewards[t] + gamma * values_extended[t + 1] * next_non_terminal - values[t]
		
		# Calcul de l'avantage généralisé
		advantages[t] = last_gae = delta + gamma * lambd * next_non_terminal * last_gae
	
	return advantages

def normalize_batch(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
	"""
	Normalise un batch de données.
	
	Args:
		x: Données à normaliser
		epsilon: Petite valeur pour éviter la division par zéro
		
	Returns:
		Données normalisées
	"""
	x_mean = np.mean(x)
	x_std = np.std(x)
	return (x - x_mean) / (x_std + epsilon)

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
	"""
	Met à jour doucement les paramètres d'un réseau cible à partir d'un réseau source.
	target = tau * source + (1 - tau) * target
	
	Args:
		target: Réseau cible
		source: Réseau source
		tau: Facteur de mise à jour (0 < tau <= 1)
	"""
	for target_param, source_param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def compute_entropy(mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
	"""
	Calcule l'entropie d'une distribution normale multivariée.
	
	Args:
		mean: Moyenne de la distribution
		log_std: Log de l'écart-type de la distribution
		
	Returns:
		Entropie de la distribution
	"""
	# L'entropie d'une distribution normale multivariée est:
	# H = 0.5 * k * (1 + log(2*pi)) + sum(log(std))
	# où k est la dimension de la distribution
	std = torch.exp(log_std)
	entropy = 0.5 * (1.0 + np.log(2 * np.pi)) + log_std
	return entropy.sum(dim=-1).mean()

def build_mlp(
	input_dim: int,
	output_dim: int,
	hidden_dims: List[int],
	activation: torch.nn.Module = torch.nn.ReLU,
	output_activation: torch.nn.Module = None
) -> torch.nn.Sequential:
	"""
	Construit un perceptron multicouche (MLP).
	
	Args:
		input_dim: Dimension de l'entrée
		output_dim: Dimension de la sortie
		hidden_dims: Liste des dimensions des couches cachées
		activation: Fonction d'activation pour les couches cachées
		output_activation: Fonction d'activation pour la couche de sortie
		
	Returns:
		MLP comme un Sequential
	"""
	layers = []
	
	# Première couche cachée
	layers.append(torch.nn.Linear(input_dim, hidden_dims[0]))
	layers.append(activation())
	
	# Couches cachées restantes
	for i in range(len(hidden_dims) - 1):
		layers.append(torch.nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
		layers.append(activation())
	
	# Couche de sortie
	layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
	
	# Activation de sortie si spécifiée
	if output_activation is not None:
		layers.append(output_activation())
	
	return torch.nn.Sequential(*layers)