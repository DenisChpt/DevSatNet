from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import uuid

from utils.serialization import Serializable


class Species(Serializable):
	"""
	Classe représentant une espèce de créature marine avec ses caractéristiques génétiques.
	"""
	
	def __init__(
		self,
		name: str,
		bodyPlan: str,  # Types: "fish", "jellyfish", "cephalopod", "crustacean", "marine_mammal"
		habitat: str = "pelagic",  # Types: "pelagic", "benthic", "reef", "deep_sea", "coastal"
		dietType: str = "omnivore",  # Types: "herbivore", "carnivore", "omnivore", "filter_feeder"
		initialPopulation: int = 10,
		color: Tuple[float, float, float] = (0.5, 0.5, 0.8)
	) -> None:
		"""
		Initialise une nouvelle espèce.
		
		Args:
			name: Nom de l'espèce
			bodyPlan: Plan corporel de base
			habitat: Habitat préféré
			dietType: Type d'alimentation
			initialPopulation: Population initiale
			color: Couleur représentative de l'espèce
		"""
		self.id: str = str(uuid.uuid4())
		self.name: str = name
		self.bodyPlan: str = bodyPlan
		self.habitat: str = habitat
		self.dietType: str = dietType
		self.initialPopulation: int = initialPopulation
		self.color: np.ndarray = np.array(color, dtype=np.float32)
		
		# Caractéristiques évolutives
		self.traits: Dict[str, float] = self._initializeTraits()
		self.compatibility: Dict[str, float] = {}  # Compatibilité avec d'autres espèces
		
		# Statistiques de l'espèce
		self.currentPopulation: int = initialPopulation
		self.peakPopulation: int = initialPopulation
		self.totalBirths: int = 0
		self.totalDeaths: int = 0
		self.generationCount: int = 1
		self.averageFitness: float = 0.0
		self.fitnessHistory: List[float] = []
		self.adaptationScore: float = 0.0
		
		# Génomes représentatifs
		self.representativeGenomeIds: List[str] = []
		self.mostSuccessfulGenomeId: Optional[str] = None
	
	def _initializeTraits(self) -> Dict[str, float]:
		"""
		Initialise les traits biologiques de l'espèce selon son plan corporel.
		
		Returns:
			Dictionnaire des traits avec leurs valeurs initiales
		"""
		# Traits communs à toutes les espèces
		traits = {
			"size": 0.5,  # Taille (0.0 = très petit, 1.0 = très grand)
			"speed": 0.5,  # Vitesse (0.0 = très lent, 1.0 = très rapide)
			"agility": 0.5,  # Agilité (0.0 = très rigide, 1.0 = très agile)
			"perception": 0.5,  # Perception (0.0 = faible, 1.0 = excellente)
			"metabolism": 0.5,  # Métabolisme (0.0 = lent, 1.0 = rapide)
			"reproductionRate": 0.5,  # Taux de reproduction (0.0 = faible, 1.0 = élevé)
			"longevity": 0.5,  # Longévité (0.0 = courte, 1.0 = longue)
			"temperatureTolerance": 0.5,  # Tolérance à la température
			"pressureTolerance": 0.5,  # Tolérance à la pression
			"oxygenEfficiency": 0.5,  # Efficacité d'utilisation de l'oxygène
			"energyEfficiency": 0.5,  # Efficacité énergétique
			"radiationResistance": 0.5,  # Résistance aux radiations
			"toxinResistance": 0.5,  # Résistance aux toxines
			"socialBehavior": 0.5,  # Comportement social (0.0 = solitaire, 1.0 = très social)
			"intelligence": 0.5  # Intelligence (0.0 = basique, 1.0 = complexe)
		}
		
		# Ajuster les traits en fonction du plan corporel
		if self.bodyPlan == "fish":
			traits.update({
				"speed": 0.7,
				"agility": 0.6,
				"perception": 0.6,
				"socialBehavior": 0.7
			})
		elif self.bodyPlan == "jellyfish":
			traits.update({
				"speed": 0.2,
				"agility": 0.3,
				"metabolism": 0.3,
				"toxinResistance": 0.8,
				"pressureTolerance": 0.7
			})
		elif self.bodyPlan == "cephalopod":
			traits.update({
				"agility": 0.8,
				"perception": 0.8,
				"intelligence": 0.8,
				"toxinResistance": 0.7,
				"socialBehavior": 0.4
			})
		elif self.bodyPlan == "crustacean":
			traits.update({
				"speed": 0.4,
				"agility": 0.5,
				"pressureTolerance": 0.6,
				"toxinResistance": 0.6,
				"metabolism": 0.4
			})
		elif self.bodyPlan == "marine_mammal":
			traits.update({
				"size": 0.8,
				"speed": 0.7,
				"intelligence": 0.8,
				"perception": 0.7,
				"socialBehavior": 0.8,
				"longevity": 0.7
			})
			
		# Ajuster en fonction de l'habitat
		if self.habitat == "deep_sea":
			traits.update({
				"pressureTolerance": 0.9,
				"metabolismRate": 0.3,
				"size": traits["size"] * 0.8,
				"perception": traits["perception"] * 1.2
			})
		elif self.habitat == "reef":
			traits.update({
				"agility": traits["agility"] * 1.2,
				"perception": traits["perception"] * 1.1,
				"socialBehavior": traits["socialBehavior"] * 1.1
			})
			
		# Ajuster en fonction du régime alimentaire
		if self.dietType == "carnivore":
			traits.update({
				"speed": traits["speed"] * 1.2,
				"perception": traits["perception"] * 1.1,
				"metabolismRate": traits["metabolism"] * 1.1
			})
		elif self.dietType == "herbivore":
			traits.update({
				"toxinResistance": traits["toxinResistance"] * 1.2,
				"metabolismRate": traits["metabolism"] * 0.9,
				"reproductionRate": traits["reproductionRate"] * 1.1
			})
		elif self.dietType == "filter_feeder":
			traits.update({
				"speed": traits["speed"] * 0.8,
				"energyEfficiency": traits["energyEfficiency"] * 1.2,
				"size": traits["size"] * 1.2
			})
			
		# S'assurer que toutes les valeurs sont dans la plage [0, 1]
		for key in traits:
			traits[key] = max(0.0, min(1.0, traits[key]))
			
		return traits
	
	def updateStats(self, population: List[Any], averageFitness: float) -> None:
		"""
		Met à jour les statistiques de l'espèce en fonction de la population actuelle.
		
		Args:
			population: Liste des créatures de cette espèce
			averageFitness: Fitness moyen de la population
		"""
		# Mettre à jour la population actuelle
		self.currentPopulation = len(population)
		
		# Mettre à jour le pic de population si nécessaire
		if self.currentPopulation > self.peakPopulation:
			self.peakPopulation = self.currentPopulation
			
		# Mettre à jour le fitness moyen et l'historique
		self.averageFitness = averageFitness
		self.fitnessHistory.append(averageFitness)
		
		# Calculer le score d'adaptation (tendance sur les 5 dernières générations)
		if len(self.fitnessHistory) >= 5:
			recentFitness = self.fitnessHistory[-5:]
			fitnessGrowth = (recentFitness[-1] - recentFitness[0]) / recentFitness[0] if recentFitness[0] > 0 else 0
			popGrowth = max(-1.0, min(1.0, (self.currentPopulation - self.initialPopulation) / max(1, self.initialPopulation)))
			
			self.adaptationScore = 0.7 * fitnessGrowth + 0.3 * popGrowth
	
	def recordBirth(self) -> None:
		"""Enregistre une naissance."""
		self.totalBirths += 1
		self.currentPopulation += 1
		
		# Mettre à jour le pic de population si nécessaire
		if self.currentPopulation > self.peakPopulation:
			self.peakPopulation = self.currentPopulation
	
	def recordDeath(self) -> None:
		"""Enregistre un décès."""
		self.totalDeaths += 1
		self.currentPopulation = max(0, self.currentPopulation - 1)
	
	def incrementGeneration(self) -> None:
		"""Incrémente le compteur de génération."""
		self.generationCount += 1
	
	def updateTraits(self, newTraits: Dict[str, float]) -> None:
		"""
		Met à jour les traits de l'espèce en fonction de l'évolution.
		
		Args:
			newTraits: Nouveaux traits moyens calculés à partir de la population
		"""
		# Vérifier que tous les traits sont présents
		for key in self.traits:
			if key in newTraits:
				# Mise à jour progressive des traits (évite les changements brusques)
				self.traits[key] = 0.8 * self.traits[key] + 0.2 * newTraits[key]
				
				# S'assurer que la valeur reste dans la plage [0, 1]
				self.traits[key] = max(0.0, min(1.0, self.traits[key]))
	
	def setRepresentativeGenomes(self, genomeIds: List[str], mostSuccessfulId: Optional[str] = None) -> None:
		"""
		Définit les génomes représentatifs de l'espèce.
		
		Args:
			genomeIds: Liste des IDs de génomes représentatifs
			mostSuccessfulId: ID du génome le plus performant
		"""
		self.representativeGenomeIds = genomeIds
		if mostSuccessfulId is not None:
			self.mostSuccessfulGenomeId = mostSuccessfulId
	
	def calculateCompatibilityWith(self, otherSpecies: 'Species') -> float:
		"""
		Calcule la compatibilité génétique avec une autre espèce.
		
		Args:
			otherSpecies: L'autre espèce à comparer
			
		Returns:
			Score de compatibilité entre 0.0 (incompatible) et 1.0 (très compatible)
		"""
		# Si la compatibilité a déjà été calculée, la retourner
		if otherSpecies.id in self.compatibility:
			return self.compatibility[otherSpecies.id]
			
		# La compatibilité dépend de la similitude des traits et du plan corporel
		traitSimilarity = 0.0
		
		# Comparer les traits communs
		commonTraits = set(self.traits.keys()).intersection(set(otherSpecies.traits.keys()))
		for trait in commonTraits:
			traitDiff = abs(self.traits[trait] - otherSpecies.traits[trait])
			traitSimilarity += 1.0 - traitDiff
			
		# Normaliser la similarité des traits
		traitSimilarity /= max(1, len(commonTraits))
		
		# Facteur de compatibilité du plan corporel
		bodyPlanCompatibility = 1.0 if self.bodyPlan == otherSpecies.bodyPlan else 0.3
		
		# Facteur de compatibilité de l'habitat
		habitatCompatibility = 1.0 if self.habitat == otherSpecies.habitat else 0.5
		
		# Calculer la compatibilité globale
		compatibility = 0.6 * traitSimilarity + 0.3 * bodyPlanCompatibility + 0.1 * habitatCompatibility
		
		# Mémoriser la compatibilité pour une utilisation future
		self.compatibility[otherSpecies.id] = compatibility
		
		return compatibility
	
	def generateDescription(self) -> str:
		"""
		Génère une description textuelle de l'espèce et de ses caractéristiques.
		
		Returns:
			Description de l'espèce
		"""
		# Description de base
		description = f"Espèce: {self.name}\n"
		description += f"Plan corporel: {self.bodyPlan}\n"
		description += f"Habitat préféré: {self.habitat}\n"
		description += f"Régime alimentaire: {self.dietType}\n\n"
		
		# Traits notables (les 5 plus élevés)
		description += "Traits notables:\n"
		sortedTraits = sorted(self.traits.items(), key=lambda x: x[1], reverse=True)
		for trait, value in sortedTraits[:5]:
			description += f"- {trait}: {value:.2f}\n"
			
		# Statistiques
		description += f"\nPopulation actuelle: {self.currentPopulation}\n"
		description += f"Population maximale atteinte: {self.peakPopulation}\n"
		description += f"Génération actuelle: {self.generationCount}\n"
		description += f"Fitness moyen: {self.averageFitness:.2f}\n"
		description += f"Score d'adaptation: {self.adaptationScore:.2f}\n"
		
		return description
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Species en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de l'espèce
		"""
		return {
			"id": self.id,
			"name": self.name,
			"bodyPlan": self.bodyPlan,
			"habitat": self.habitat,
			"dietType": self.dietType,
			"initialPopulation": self.initialPopulation,
			"color": self.color.tolist(),
			"traits": self.traits,
			"compatibility": self.compatibility,
			"currentPopulation": self.currentPopulation,
			"peakPopulation": self.peakPopulation,
			"totalBirths": self.totalBirths,
			"totalDeaths": self.totalDeaths,
			"generationCount": self.generationCount,
			"averageFitness": self.averageFitness,
			"fitnessHistory": self.fitnessHistory,
			"adaptationScore": self.adaptationScore,
			"representativeGenomeIds": self.representativeGenomeIds,
			"mostSuccessfulGenomeId": self.mostSuccessfulGenomeId
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Species':
		"""
		Crée une instance de Species à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de l'espèce
			
		Returns:
			Instance de Species reconstruite
		"""
		species = cls(
			name=data["name"],
			bodyPlan=data["bodyPlan"],
			habitat=data["habitat"],
			dietType=data["dietType"],
			initialPopulation=data["initialPopulation"],
			color=tuple(data["color"])
		)
		
		species.id = data["id"]
		species.traits = data["traits"]
		species.compatibility = data["compatibility"]
		species.currentPopulation = data["currentPopulation"]
		species.peakPopulation = data["peakPopulation"]
		species.totalBirths = data["totalBirths"]
		species.totalDeaths = data["totalDeaths"]
		species.generationCount = data["generationCount"]
		species.averageFitness = data["averageFitness"]
		species.fitnessHistory = data["fitnessHistory"]
		species.adaptationScore = data["adaptationScore"]
		species.representativeGenomeIds = data["representativeGenomeIds"]
		species.mostSuccessfulGenomeId = data["mostSuccessfulGenomeId"]
		
		return species