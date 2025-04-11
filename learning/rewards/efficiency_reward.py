# learning/rewards/efficiency_reward.py
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from core.entities.creature import Creature


class EfficiencyReward:
	"""
	Classe qui calcule les récompenses liées à l'efficacité énergétique des créatures.
	Ces récompenses favorisent une utilisation optimale de l'énergie lors des déplacements
	et autres actions.
	"""
	
	def __init__(
		self,
		movementEfficiencyWeight: float = 1.0,
		metabolicEfficiencyWeight: float = 0.5,
		actionEfficiencyWeight: float = 0.7,
		distanceWeight: float = 0.8,
		energyConsumptionPenaltyFactor: float = 0.2,
		excessiveMovementPenalty: float = 0.3,
		optimalSpeedFactor: float = 0.7,  # Facteur de vitesse optimal par rapport à la vitesse max
		useMovingAverage: bool = True,
		averageWindowSize: int = 10
	) -> None:
		"""
		Initialise le calculateur de récompenses d'efficacité.
		
		Args:
			movementEfficiencyWeight: Poids des récompenses liées à l'efficacité de mouvement
			metabolicEfficiencyWeight: Poids des récompenses liées à l'efficacité métabolique
			actionEfficiencyWeight: Poids des récompenses liées à l'efficacité des actions
			distanceWeight: Poids des récompenses liées à la distance parcourue
			energyConsumptionPenaltyFactor: Facteur de pénalité pour consommation d'énergie
			excessiveMovementPenalty: Pénalité pour mouvements excessifs ou erratiques
			optimalSpeedFactor: Facteur de vitesse optimal pour l'efficacité
			useMovingAverage: Utiliser une moyenne mobile pour lisser les récompenses
			averageWindowSize: Taille de la fenêtre pour la moyenne mobile
		"""
		self.movementEfficiencyWeight = movementEfficiencyWeight
		self.metabolicEfficiencyWeight = metabolicEfficiencyWeight
		self.actionEfficiencyWeight = actionEfficiencyWeight
		self.distanceWeight = distanceWeight
		self.energyConsumptionPenaltyFactor = energyConsumptionPenaltyFactor
		self.excessiveMovementPenalty = excessiveMovementPenalty
		self.optimalSpeedFactor = optimalSpeedFactor
		self.useMovingAverage = useMovingAverage
		self.averageWindowSize = averageWindowSize
		
		# États précédents pour les calculs d'efficacité
		self.previousStates: Dict[str, Dict[str, Any]] = {}
		
		# Historique des récompenses pour la moyenne mobile
		self.rewardHistory: Dict[str, List[float]] = {}
	
	def calculateReward(self, creature: Creature, environmentInfo: Dict[str, Any] = None) -> float:
		"""
		Calcule la récompense d'efficacité pour une créature.
		
		Args:
			creature: Créature pour laquelle calculer la récompense
			environmentInfo: Informations supplémentaires sur l'environnement
			
		Returns:
			Récompense totale
		"""
		# Si la créature est morte, aucune récompense d'efficacité
		if not creature.isAlive:
			# Effacer les états précédents
			if creature.id in self.previousStates:
				del self.previousStates[creature.id]
			if creature.id in self.rewardHistory:
				del self.rewardHistory[creature.id]
				
			return 0.0
		
		# Récupérer l'état actuel
		currentState = self._getCreatureState(creature, environmentInfo)
		
		# Si nous n'avons pas d'état précédent, l'enregistrer et retourner 0
		if creature.id not in self.previousStates:
			self.previousStates[creature.id] = currentState
			return 0.0
			
		# Récupérer l'état précédent
		previousState = self.previousStates[creature.id]
		
		# Calculer les différentes composantes de la récompense
		movementReward = self._calculateMovementEfficiency(previousState, currentState)
		metabolicReward = self._calculateMetabolicEfficiency(previousState, currentState)
		actionReward = self._calculateActionEfficiency(previousState, currentState)
		
		# Combiner les récompenses avec leurs poids
		totalReward = (
			movementReward * self.movementEfficiencyWeight +
			metabolicReward * self.metabolicEfficiencyWeight +
			actionReward * self.actionEfficiencyWeight
		)
		
		# Mettre à jour l'historique des récompenses
		if self.useMovingAverage:
			if creature.id not in self.rewardHistory:
				self.rewardHistory[creature.id] = []
				
			history = self.rewardHistory[creature.id]
			history.append(totalReward)
			
			# Limiter la taille de l'historique
			if len(history) > self.averageWindowSize:
				history = history[-self.averageWindowSize:]
				self.rewardHistory[creature.id] = history
				
			# Calculer la moyenne mobile
			totalReward = sum(history) / len(history)
		
		# Stocker l'état actuel comme état précédent pour le prochain calcul
		self.previousStates[creature.id] = currentState
		
		return totalReward
	
	def _getCreatureState(self, creature: Creature, environmentInfo: Dict[str, Any] = None) -> Dict[str, Any]:
		"""
		Extrait les informations d'état pertinentes de la créature.
		
		Args:
			creature: Créature dont extraire l'état
			environmentInfo: Informations sur l'environnement
			
		Returns:
			Dictionnaire de l'état de la créature
		"""
		# Vitesse et direction
		velocity = np.linalg.norm(creature.velocity)
		
		# Créer l'état
		state = {
			"position": np.array(creature.position),
			"velocity": velocity,
			"energy": creature.energy,
			"totalEnergyConsumed": creature.totalEnergyConsumed,
			"totalDistanceTraveled": creature.totalDistanceTraveled,
			"time": environmentInfo.get("time", 0.0) if environmentInfo else 0.0,
			"lastAction": creature.lastAction.copy() if creature.lastAction is not None else None
		}
		
		return state
	
	def _calculateMovementEfficiency(self, previousState: Dict[str, Any], currentState: Dict[str, Any]) -> float:
		"""
		Calcule l'efficacité de mouvement basée sur la distance parcourue et l'énergie consommée.
		
		Args:
			previousState: État précédent de la créature
			currentState: État actuel de la créature
			
		Returns:
			Récompense d'efficacité de mouvement
		"""
		# Calculer la distance parcourue
		distanceTraveled = currentState["totalDistanceTraveled"] - previousState["totalDistanceTraveled"]
		
		# Calculer l'énergie consommée
		energyConsumed = currentState["totalEnergyConsumed"] - previousState["totalEnergyConsumed"]
		
		# Si aucune énergie n'a été consommée ou aucune distance parcourue
		if energyConsumed < 1e-6 or distanceTraveled < 1e-6:
			return 0.0
			
		# Calculer l'efficacité: distance parcourue par unité d'énergie
		efficiency = distanceTraveled / energyConsumed
		
		# Normaliser l'efficacité (supposons qu'une efficacité de 1.0 est bonne)
		normalizedEfficiency = min(1.0, efficiency)
		
		# Récompense pour la distance parcourue
		distanceReward = distanceTraveled * self.distanceWeight
		
		# Pénalité pour la consommation d'énergie excessive
		energyPenalty = -energyConsumed * self.energyConsumptionPenaltyFactor
		
		# Pénalité pour les mouvements excessifs ou erratiques
		# Vérifier si la vitesse est proche de l'optimal
		currentVelocity = currentState["velocity"]
		
		# Supposons que nous connaissons la vitesse maximale de la créature (à ajuster selon le modèle)
		maxVelocity = 10.0  # Valeur arbitraire, à ajuster
		optimalVelocity = maxVelocity * self.optimalSpeedFactor
		
		# Pénalité si la vitesse s'éloigne trop de l'optimal
		velocityDeviation = abs(currentVelocity - optimalVelocity) / maxVelocity
		movementPenalty = -velocityDeviation * self.excessiveMovementPenalty
		
		# Combiner toutes les composantes
		return normalizedEfficiency + distanceReward + energyPenalty + movementPenalty
	
	def _calculateMetabolicEfficiency(self, previousState: Dict[str, Any], currentState: Dict[str, Any]) -> float:
		"""
		Calcule l'efficacité métabolique basée sur la consommation d'énergie au repos.
		
		Args:
			previousState: État précédent de la créature
			currentState: État actuel de la créature
			
		Returns:
			Récompense d'efficacité métabolique
		"""
		# Calculer l'énergie consommée
		energyConsumed = currentState["totalEnergyConsumed"] - previousState["totalEnergyConsumed"]
		
		# Calculer le temps écoulé
		timeElapsed = currentState["time"] - previousState["time"]
		
		if timeElapsed < 1e-6:
			return 0.0
			
		# Taux de consommation d'énergie par unité de temps
		energyRate = energyConsumed / timeElapsed
		
		# Pénalité pour un taux métabolique élevé
		# Plus le taux est bas, meilleure est l'efficacité
		# Supposons qu'un taux de 1.0 est normal
		normalizedRate = min(2.0, energyRate) / 2.0  # Normaliser entre 0 et 1
		
		# Inverser pour obtenir une récompense (plus bas = meilleur)
		metabolicEfficiency = 1.0 - normalizedRate
		
		return metabolicEfficiency
	
	def _calculateActionEfficiency(self, previousState: Dict[str, Any], currentState: Dict[str, Any]) -> float:
		"""
		Calcule l'efficacité des actions basée sur l'énergie consommée par action.
		
		Args:
			previousState: État précédent de la créature
			currentState: État actuel de la créature
			
		Returns:
			Récompense d'efficacité d'action
		"""
		# Si pas d'action, pas de récompense
		if currentState["lastAction"] is None or previousState["lastAction"] is None:
			return 0.0
			
		# Calculer la différence d'action
		actionDiff = np.linalg.norm(currentState["lastAction"] - previousState["lastAction"])
		
		# Calculer l'énergie consommée
		energyConsumed = currentState["totalEnergyConsumed"] - previousState["totalEnergyConsumed"]
		
		if energyConsumed < 1e-6:
			return 0.0
			
		# Si peu de changement d'action mais beaucoup d'énergie consommée, mauvaise efficacité
		# Si beaucoup de changement d'action avec peu d'énergie, bonne efficacité
		actionEfficiency = actionDiff / energyConsumed
		
		# Normaliser l'efficacité
		normalizedEfficiency = min(1.0, actionEfficiency)
		
		return normalizedEfficiency