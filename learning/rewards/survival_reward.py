# learning/rewards/survival_reward.py
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from core.entities.creature import Creature


class SurvivalReward:
	"""
	Classe qui calcule les récompenses liées à la survie des créatures.
	Ces récompenses favorisent les comportements qui prolongent la vie et
	maintiennent une bonne santé.
	"""
	
	def __init__(
		self,
		energyWeight: float = 1.0,
		healthWeight: float = 1.0,
		ageWeight: float = 0.01,
		stayAliveReward: float = 0.1,
		deathPenalty: float = -10.0,
		energyThresholds: List[float] = [0.2, 0.5, 0.8],
		healthThresholds: List[float] = [0.3, 0.6, 0.9],
		criticalEnergyThreshold: float = 0.1,
		criticalHealthThreshold: float = 0.1,
		previousStateMemory: bool = True
	) -> None:
		"""
		Initialise le calculateur de récompenses de survie.
		
		Args:
			energyWeight: Poids des récompenses liées à l'énergie
			healthWeight: Poids des récompenses liées à la santé
			ageWeight: Poids des récompenses liées à l'âge
			stayAliveReward: Récompense de base pour rester en vie à chaque pas de temps
			deathPenalty: Pénalité appliquée lors de la mort
			energyThresholds: Seuils d'énergie pour les récompenses graduelles
			healthThresholds: Seuils de santé pour les récompenses graduelles
			criticalEnergyThreshold: Seuil d'énergie critique
			criticalHealthThreshold: Seuil de santé critique
			previousStateMemory: Utiliser la mémoire de l'état précédent pour les récompenses différentielles
		"""
		self.energyWeight = energyWeight
		self.healthWeight = healthWeight
		self.ageWeight = ageWeight
		self.stayAliveReward = stayAliveReward
		self.deathPenalty = deathPenalty
		self.energyThresholds = sorted(energyThresholds)
		self.healthThresholds = sorted(healthThresholds)
		self.criticalEnergyThreshold = criticalEnergyThreshold
		self.criticalHealthThreshold = criticalHealthThreshold
		self.previousStateMemory = previousStateMemory
		
		# État précédent pour les récompenses différentielles
		self.previousStates: Dict[str, Dict[str, Any]] = {}
	
	def calculateReward(self, creature: Creature, environmentInfo: Dict[str, Any] = None) -> float:
		"""
		Calcule la récompense de survie pour une créature.
		
		Args:
			creature: Créature pour laquelle calculer la récompense
			environmentInfo: Informations supplémentaires sur l'environnement
			
		Returns:
			Récompense totale
		"""
		# Si la créature est morte, appliquer la pénalité de mort
		if not creature.isAlive:
			# Effacer l'état précédent
			if creature.id in self.previousStates:
				del self.previousStates[creature.id]
				
			return self.deathPenalty
		
		# Récompense de base pour rester en vie
		reward = self.stayAliveReward
		
		# Récupérer l'état actuel
		currentState = self._getCreatureState(creature)
		
		# Si nous avons un état précédent, calculer les récompenses différentielles
		if self.previousStateMemory and creature.id in self.previousStates:
			previousState = self.previousStates[creature.id]
			differentialReward = self._calculateDifferentialReward(previousState, currentState)
			reward += differentialReward
		
		# Calculer les récompenses basées sur les niveaux absolus
		absoluteReward = self._calculateAbsoluteReward(currentState)
		reward += absoluteReward
		
		# Récompenses liées à l'âge
		ageReward = currentState["age"] * self.ageWeight
		reward += ageReward
		
		# Stocker l'état actuel comme état précédent pour le prochain calcul
		if self.previousStateMemory:
			self.previousStates[creature.id] = currentState
		
		return reward
	
	def _getCreatureState(self, creature: Creature) -> Dict[str, Any]:
		"""
		Extrait les informations d'état pertinentes de la créature.
		
		Args:
			creature: Créature dont extraire l'état
			
		Returns:
			Dictionnaire de l'état de la créature
		"""
		return {
			"energy": creature.energy,
			"maxEnergy": creature.maxEnergy,
			"energyRatio": creature.energy / creature.maxEnergy,
			"health": getattr(creature, "health", 1.0),  # Pas toutes les créatures ont une santé
			"maxHealth": getattr(creature, "maxHealth", 1.0),
			"healthRatio": getattr(creature, "health", 1.0) / getattr(creature, "maxHealth", 1.0),
			"age": creature.age,
			"isAlive": creature.isAlive,
			"reproductionReadiness": creature.reproductionReadiness
		}
	
	def _calculateDifferentialReward(self, previousState: Dict[str, Any], currentState: Dict[str, Any]) -> float:
		"""
		Calcule la récompense basée sur les changements d'état.
		
		Args:
			previousState: État précédent de la créature
			currentState: État actuel de la créature
			
		Returns:
			Récompense différentielle
		"""
		# Calcul des différences
		energyDiff = currentState["energyRatio"] - previousState["energyRatio"]
		healthDiff = currentState["healthRatio"] - previousState["healthRatio"]
		
		# Récompense pour l'amélioration de l'énergie
		energyReward = energyDiff * self.energyWeight
		
		# Récompense pour l'amélioration de la santé
		healthReward = healthDiff * self.healthWeight
		
		# Combiner les récompenses
		return energyReward + healthReward
	
	def _calculateAbsoluteReward(self, state: Dict[str, Any]) -> float:
		"""
		Calcule la récompense basée sur les niveaux absolus d'énergie et de santé.
		
		Args:
			state: État actuel de la créature
			
		Returns:
			Récompense absolue
		"""
		# Récompenses graduelles basées sur les seuils d'énergie
		energyReward = 0.0
		energyRatio = state["energyRatio"]
		
		# Pénalités pour niveau d'énergie critique
		if energyRatio < self.criticalEnergyThreshold:
			energyReward -= 1.0 * self.energyWeight
		else:
			# Récompenses pour chaque seuil atteint
			for threshold in self.energyThresholds:
				if energyRatio >= threshold:
					energyReward += 0.2 * self.energyWeight
		
		# Récompenses graduelles basées sur les seuils de santé
		healthReward = 0.0
		healthRatio = state["healthRatio"]
		
		# Pénalités pour niveau de santé critique
		if healthRatio < self.criticalHealthThreshold:
			healthReward -= 1.0 * self.healthWeight
		else:
			# Récompenses pour chaque seuil atteint
			for threshold in self.healthThresholds:
				if healthRatio >= threshold:
					healthReward += 0.2 * self.healthWeight
		
		# Récompense pour être prêt à se reproduire
		reproductionReward = 0.0
		if state["reproductionReadiness"] > 0.8:
			reproductionReward = 0.5
		
		return energyReward + healthReward + reproductionReward