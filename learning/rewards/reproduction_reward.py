# learning/rewards/reproduction_reward.py
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from core.entities.creature import Creature


class ReproductionReward:
	"""
	Classe qui calcule les récompenses liées à la reproduction des créatures.
	Ces récompenses favorisent les comportements qui augmentent les chances de
	reproduction et le succès reproductif.
	"""
	
	def __init__(
		self,
		reproductionWeight: float = 5.0,
		matingWeight: float = 2.0,
		offspringWeight: float = 10.0,
		preparationWeight: float = 1.0,
		matingProximityThreshold: float = 10.0,  # Distance maximale pour considérer une tentative d'accouplement
		energyThresholdForReproduction: float = 0.7,  # Niveau d'énergie minimal pour la reproduction
		matingSuccessReward: float = 3.0,
		trackHierarchy: bool = True,
		dominanceBonus: float = 0.5
	) -> None:
		"""
		Initialise le calculateur de récompenses de reproduction.
		
		Args:
			reproductionWeight: Poids global des récompenses de reproduction
			matingWeight: Poids des récompenses liées à l'accouplement
			offspringWeight: Poids des récompenses liées aux descendants
			preparationWeight: Poids des récompenses liées à la préparation à la reproduction
			matingProximityThreshold: Distance maximale pour considérer une tentative d'accouplement
			energyThresholdForReproduction: Niveau d'énergie minimal pour la reproduction
			matingSuccessReward: Récompense pour un accouplement réussi
			trackHierarchy: Tenir compte de la hiérarchie sociale
			dominanceBonus: Bonus pour les individus dominants
		"""
		self.reproductionWeight = reproductionWeight
		self.matingWeight = matingWeight
		self.offspringWeight = offspringWeight
		self.preparationWeight = preparationWeight
		self.matingProximityThreshold = matingProximityThreshold
		self.energyThresholdForReproduction = energyThresholdForReproduction
		self.matingSuccessReward = matingSuccessReward
		self.trackHierarchy = trackHierarchy
		self.dominanceBonus = dominanceBonus
		
		# Suivi des accouplements et des naissances
		self.matingHistory: Dict[str, List[Dict[str, Any]]] = {}
		self.birthHistory: Dict[str, List[Dict[str, Any]]] = {}
		
		# Suivi de la hiérarchie sociale (si activé)
		self.dominanceRanking: Dict[str, float] = {}
	
	def calculateReward(
		self,
		creature: Creature,
		environmentInfo: Dict[str, Any] = None,
		nearbyCreatures: List[Creature] = None
	) -> float:
		"""
		Calcule la récompense de reproduction pour une créature.
		
		Args:
			creature: Créature pour laquelle calculer la récompense
			environmentInfo: Informations supplémentaires sur l'environnement
			nearbyCreatures: Liste des créatures à proximité
			
		Returns:
			Récompense totale
		"""
		# Si la créature est morte, aucune récompense de reproduction
		if not creature.isAlive:
			return 0.0
		
		totalReward = 0.0
		
		# Récompense pour la préparation à la reproduction
		preparationReward = self._calculatePreparationReward(creature)
		totalReward += preparationReward * self.preparationWeight
		
		# Récompense pour les comportements d'accouplement
		if nearbyCreatures:
			matingReward = self._calculateMatingReward(creature, nearbyCreatures, environmentInfo)
			totalReward += matingReward * self.matingWeight
		
		# Récompense pour avoir des descendants
		offspringReward = self._calculateOffspringReward(creature)
		totalReward += offspringReward * self.offspringWeight
		
		# Appliquer le poids global
		return totalReward * self.reproductionWeight
	
	def _calculatePreparationReward(self, creature: Creature) -> float:
		"""
		Calcule la récompense pour la préparation à la reproduction.
		
		Args:
			creature: Créature pour laquelle calculer la récompense
			
		Returns:
			Récompense de préparation
		"""
		# Vérifier si la créature est prête pour la reproduction
		readiness = creature.reproductionReadiness
		
		# Vérifier si l'énergie est suffisante
		energyRatio = creature.energy / creature.maxEnergy
		energySufficient = energyRatio >= self.energyThresholdForReproduction
		
		# Récompense de base pour la préparation
		baseReward = readiness * 0.5
		
		# Bonus si l'énergie est suffisante
		energyBonus = 0.5 if energySufficient else 0.0
		
		# Pénalité si la créature est trop jeune
		agePenalty = 0.0
		if hasattr(creature, "getMinReproductionAge"):
			minAge = creature.getMinReproductionAge()
			if creature.age < minAge:
				agePenalty = -0.2 * (1.0 - creature.age / minAge)
		
		return baseReward + energyBonus + agePenalty
	
	def _calculateMatingReward(
		self,
		creature: Creature,
		nearbyCreatures: List[Creature],
		environmentInfo: Dict[str, Any] = None
	) -> float:
		"""
		Calcule la récompense pour les comportements d'accouplement.
		
		Args:
			creature: Créature pour laquelle calculer la récompense
			nearbyCreatures: Liste des créatures à proximité
			environmentInfo: Informations supplémentaires sur l'environnement
			
		Returns:
			Récompense d'accouplement
		"""
		# Si la créature n'est pas prête pour la reproduction, pas de récompense
		if creature.reproductionReadiness < 0.8 or creature.energy / creature.maxEnergy < self.energyThresholdForReproduction:
			return 0.0
		
		# Identifier les partenaires potentiels
		potentialMates = []
		for other in nearbyCreatures:
			# Vérifier si c'est une créature différente, vivante et de la même espèce
			if (other.id != creature.id and 
				other.isAlive and 
				other.speciesId == creature.speciesId and
				other.reproductionReadiness >= 0.8 and
				other.energy / other.maxEnergy >= self.energyThresholdForReproduction):
				
				# Calculer la distance
				distance = np.linalg.norm(creature.position - other.position)
				
				# Si suffisamment proche pour l'accouplement
				if distance <= self.matingProximityThreshold:
					potentialMates.append((other, distance))
		
		# Si aucun partenaire potentiel, pas de récompense
		if not potentialMates:
			return 0.0
		
		# Trier par proximité
		potentialMates.sort(key=lambda x: x[1])
		
		# Récompense basée sur le nombre de partenaires potentiels
		numMatesReward = min(1.0, len(potentialMates) / 3)  # Plafonné à 1.0 pour 3+ partenaires
		
		# Récompense pour la proximité avec le partenaire le plus proche
		proximityReward = 1.0 - (potentialMates[0][1] / self.matingProximityThreshold)
		
		# Vérifier si un accouplement a eu lieu
		matingSuccess = False
		if environmentInfo and "mating_events" in environmentInfo:
			for event in environmentInfo["mating_events"]:
				if event["creature1_id"] == creature.id or event["creature2_id"] == creature.id:
					matingSuccess = True
					
					# Enregistrer l'accouplement dans l'historique
					if creature.id not in self.matingHistory:
						self.matingHistory[creature.id] = []
						
					self.matingHistory[creature.id].append({
						"time": environmentInfo.get("time", 0),
						"partner_id": event["creature1_id"] if event["creature2_id"] == creature.id else event["creature2_id"],
						"offspring_ids": event.get("offspring_ids", [])
					})
					
					break
		
		# Récompense pour un accouplement réussi
		successReward = self.matingSuccessReward if matingSuccess else 0.0
		
		# Bonus de dominance
		dominanceBonus = 0.0
		if self.trackHierarchy and creature.id in self.dominanceRanking:
			dominanceRank = self.dominanceRanking[creature.id]
			dominanceBonus = dominanceRank * self.dominanceBonus
		
		return numMatesReward + proximityReward + successReward + dominanceBonus
	
	def _calculateOffspringReward(self, creature: Creature) -> float:
		"""
		Calcule la récompense pour avoir des descendants.
		
		Args:
			creature: Créature pour laquelle calculer la récompense
			
		Returns:
			Récompense pour les descendants
		"""
		# Nombre de descendants
		offspringCount = creature.offspring
		
		# Récompense de base pour avoir des descendants
		if offspringCount == 0:
			return 0.0
			
		# Récompense logarithmique pour éviter l'explosion pour de nombreux descendants
		offspringReward = np.log1p(offspringCount)  # log(1 + x) pour éviter log(0)
		
		# Limiter la récompense maximale
		return min(5.0, offspringReward)
	
	def recordBirth(self, parentId: str, offspringIds: List[str], time: float = 0.0) -> None:
		"""
		Enregistre une naissance dans l'historique.
		
		Args:
			parentId: ID du parent
			offspringIds: IDs des descendants
			time: Heure de la naissance
		"""
		if parentId not in self.birthHistory:
			self.birthHistory[parentId] = []
			
		self.birthHistory[parentId].append({
			"time": time,
			"offspring_ids": offspringIds
		})
		
		# Mettre à jour la hiérarchie sociale
		if self.trackHierarchy:
			currentRank = self.dominanceRanking.get(parentId, 0.0)
			# Augmenter le rang en fonction du nombre de descendants
			self.dominanceRanking[parentId] = min(1.0, currentRank + 0.1 * len(offspringIds))
	
	def updateDominanceRanking(self, creatureId: str, dominanceValue: float) -> None:
		"""
		Met à jour le classement de dominance d'une créature.
		
		Args:
			creatureId: ID de la créature
			dominanceValue: Nouvelle valeur de dominance (0-1)
		"""
		if self.trackHierarchy:
			self.dominanceRanking[creatureId] = max(0.0, min(1.0, dominanceValue))
	
	def resetHistories(self) -> None:
		"""Réinitialise les historiques d'accouplement et de naissance."""
		self.matingHistory.clear()
		self.birthHistory.clear()
		
	def getReproductiveSuccess(self, creatureId: str) -> Dict[str, Any]:
		"""
		Calcule les statistiques de succès reproductif pour une créature.
		
		Args:
			creatureId: ID de la créature
			
		Returns:
			Dictionnaire des statistiques de reproduction
		"""
		matingCount = len(self.matingHistory.get(creatureId, []))
		birthEvents = self.birthHistory.get(creatureId, [])
		offspringCount = sum(len(event["offspring_ids"]) for event in birthEvents)
		
		return {
			"matings": matingCount,
			"births": len(birthEvents),
			"offspring_count": offspringCount,
			"dominance_rank": self.dominanceRanking.get(creatureId, 0.0)
		}