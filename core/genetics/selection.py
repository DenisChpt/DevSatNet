# core/genetics/selection.py
from typing import Dict, List, Tuple, Any, Optional, Set, Callable
import numpy as np
import uuid
import copy

from core.genetics.genome import Genome
from utils.serialization import Serializable


class SelectionSystem(Serializable):
	"""
	Système gérant la sélection naturelle des créatures.
	Détermine quels individus sont sélectionnés pour la reproduction
	et quels individus sont éliminés de la population.
	"""
	
	def __init__(
		self,
		selectionPressure: float = 1.5,
		elitismRatio: float = 0.1,
		tournamentSize: int = 3,
		minPopulationSize: int = 20,
		maxPopulationSize: int = 1000,
		noveltyBonus: float = 0.2,
		diversityWeight: float = 0.3,
		adaptabilityWeight: float = 0.2,
		fitnessShareSigma: float = 0.3,
		useFitnessSharing: bool = True,
		selectionMethod: str = "tournament",  # "tournament", "roulette", "rank"
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise le système de sélection.
		
		Args:
			selectionPressure: Intensité de la pression de sélection (plus élevé = plus forte)
			elitismRatio: Proportion des meilleurs individus à préserver automatiquement
			tournamentSize: Taille des tournois pour la sélection par tournoi
			minPopulationSize: Taille minimale de la population
			maxPopulationSize: Taille maximale de la population
			noveltyBonus: Bonus de fitness pour les comportements nouveaux
			diversityWeight: Poids de la diversité dans le calcul du fitness
			adaptabilityWeight: Poids de l'adaptabilité dans le calcul du fitness
			fitnessShareSigma: Paramètre sigma pour le partage de fitness
			useFitnessSharing: Activer/désactiver le partage de fitness
			selectionMethod: Méthode de sélection à utiliser
			seed: Graine pour le générateur de nombres aléatoires
		"""
		self.selectionPressure: float = selectionPressure
		self.elitismRatio: float = elitismRatio
		self.tournamentSize: int = tournamentSize
		self.minPopulationSize: int = minPopulationSize
		self.maxPopulationSize: int = maxPopulationSize
		self.noveltyBonus: float = noveltyBonus
		self.diversityWeight: float = diversityWeight
		self.adaptabilityWeight: float = adaptabilityWeight
		self.fitnessShareSigma: float = fitnessShareSigma
		self.useFitnessSharing: bool = useFitnessSharing
		self.selectionMethod: str = selectionMethod
		
		# Initialiser le générateur de nombres aléatoires
		self.rng: np.random.Generator = np.random.default_rng(seed)
		
		# Statistiques
		self.generationCount: int = 0
		self.totalIndividualsEvaluated: int = 0
		self.speciesStats: Dict[str, Dict[str, float]] = {}  # Statistiques par espèce
	
	def selectParents(self, population: List[Genome], count: int) -> List[Genome]:
		"""
		Sélectionne les parents pour la reproduction.
		
		Args:
			population: Liste des génomes dans la population actuelle
			count: Nombre de parents à sélectionner
			
		Returns:
			Liste des génomes sélectionnés comme parents
		"""
		if not population:
			return []
			
		# S'assurer que nous ne sélectionnons pas plus d'individus que disponibles
		count = min(count, len(population))
		
		# Calculer les fitness ajustés si nécessaire
		if self.useFitnessSharing:
			adjustedFitness = self._calculateSharedFitness(population)
		else:
			adjustedFitness = {genome.id: genome.fitness for genome in population}
			
		# Sélectionner les parents selon la méthode choisie
		selectedParents = []
		
		if self.selectionMethod == "tournament":
			selectedParents = self._tournamentSelection(population, adjustedFitness, count)
		elif self.selectionMethod == "roulette":
			selectedParents = self._rouletteWheelSelection(population, adjustedFitness, count)
		elif self.selectionMethod == "rank":
			selectedParents = self._rankSelection(population, count)
		else:
			# Méthode par défaut: tournoi
			selectedParents = self._tournamentSelection(population, adjustedFitness, count)
			
		return selectedParents
	
	def _tournamentSelection(
		self,
		population: List[Genome],
		fitnessDict: Dict[str, float],
		count: int
	) -> List[Genome]:
		"""
		Sélection par tournoi: pour chaque parent à sélectionner, on choisit aléatoirement
		tournamentSize individus et on sélectionne le meilleur.
		
		Args:
			population: Liste des génomes dans la population
			fitnessDict: Dictionnaire des valeurs de fitness (id -> fitness)
			count: Nombre de parents à sélectionner
			
		Returns:
			Liste des génomes sélectionnés comme parents
		"""
		selectedParents = []
		
		for _ in range(count):
			# Sélectionner aléatoirement tournamentSize individus
			if len(population) <= self.tournamentSize:
				tournament = population
			else:
				tournamentIndices = self.rng.choice(len(population), size=self.tournamentSize, replace=False)
				tournament = [population[i] for i in tournamentIndices]
			
			# Sélectionner le meilleur individu du tournoi
			bestIndividual = max(tournament, key=lambda genome: fitnessDict[genome.id])
			selectedParents.append(bestIndividual)
			
		return selectedParents
	
	def _rouletteWheelSelection(
		self,
		population: List[Genome],
		fitnessDict: Dict[str, float],
		count: int
	) -> List[Genome]:
		"""
		Sélection par roulette: la probabilité de sélection est proportionnelle à la fitness.
		
		Args:
			population: Liste des génomes dans la population
			fitnessDict: Dictionnaire des valeurs de fitness (id -> fitness)
			count: Nombre de parents à sélectionner
			
		Returns:
			Liste des génomes sélectionnés comme parents
		"""
		# Obtenir les fitness de tous les individus
		fitnesses = np.array([fitnessDict[genome.id] for genome in population])
		
		# Gérer le cas où toutes les fitness sont nulles ou négatives
		if np.sum(fitnesses) <= 0:
			# Utiliser des probabilités uniformes
			probabilities = np.ones(len(population)) / len(population)
		else:
			# Normaliser les fitness pour obtenir des probabilités
			probabilities = fitnesses / np.sum(fitnesses)
			
		# Sélectionner les parents en fonction de leurs probabilités
		selectedIndices = self.rng.choice(len(population), size=count, replace=True, p=probabilities)
		selectedParents = [population[i] for i in selectedIndices]
		
		return selectedParents
	
	def _rankSelection(self, population: List[Genome], count: int) -> List[Genome]:
		"""
		Sélection par rang: la probabilité de sélection dépend du rang (position)
		de l'individu quand la population est triée par fitness.
		
		Args:
			population: Liste des génomes dans la population
			count: Nombre de parents à sélectionner
			
		Returns:
			Liste des génomes sélectionnés comme parents
		"""
		# Trier la population par fitness (du plus faible au plus élevé)
		sortedPopulation = sorted(population, key=lambda genome: genome.fitness)
		
		# Calculer les rangs (1 pour le plus faible, N pour le plus élevé)
		ranks = np.arange(1, len(sortedPopulation) + 1)
		
		# Appliquer la pression de sélection (non-linéarité)
		ranksWithPressure = ranks ** self.selectionPressure
		
		# Normaliser pour obtenir des probabilités
		probabilities = ranksWithPressure / np.sum(ranksWithPressure)
		
		# Sélectionner les parents en fonction de leurs probabilités
		selectedIndices = self.rng.choice(len(sortedPopulation), size=count, replace=True, p=probabilities)
		selectedParents = [sortedPopulation[i] for i in selectedIndices]
		
		return selectedParents
	
	def selectSurvivors(
		self,
		currentPopulation: List[Genome],
		offspring: List[Genome],
		targetSize: int
	) -> List[Genome]:
		"""
		Sélectionne les individus qui survivent à la prochaine génération.
		
		Args:
			currentPopulation: Population actuelle
			offspring: Nouveaux individus créés par reproduction
			targetSize: Taille cible de la nouvelle population
			
		Returns:
			Liste des génomes sélectionnés pour la survie
		"""
		# S'assurer que la population cible respecte les limites
		targetSize = max(self.minPopulationSize, min(targetSize, self.maxPopulationSize))
		
		# Combiner la population actuelle et la progéniture
		combinedPopulation = currentPopulation + offspring
		
		# Si la population combinée est déjà inférieure à la cible, tout le monde survit
		if len(combinedPopulation) <= targetSize:
			return combinedPopulation
			
		# Elitisme: conserver les meilleurs individus automatiquement
		eliteCount = int(targetSize * self.elitismRatio)
		elites = sorted(combinedPopulation, key=lambda genome: genome.fitness, reverse=True)[:eliteCount]
		
		# Reste de la population à sélectionner
		remainingCount = targetSize - eliteCount
		
		# Exclure les élites de la sélection pour le reste
		nonElites = [genome for genome in combinedPopulation if genome not in elites]
		
		# Sélectionner le reste de la population par tournoi
		selectedRest = []
		if nonElites and remainingCount > 0:
			# Calculer les fitness ajustés si nécessaire
			if self.useFitnessSharing:
				adjustedFitness = self._calculateSharedFitness(nonElites)
			else:
				adjustedFitness = {genome.id: genome.fitness for genome in nonElites}
				
			selectedRest = self._tournamentSelection(nonElites, adjustedFitness, remainingCount)
			
		# Combiner les élites et le reste sélectionné
		survivors = elites + selectedRest
		
		return survivors
	
	def _calculateSharedFitness(self, population: List[Genome]) -> Dict[str, float]:
		"""
		Calcule les fitness ajustés selon la méthode de partage de fitness.
		Cette méthode réduit la fitness des individus similaires pour maintenir la diversité.
		
		Args:
			population: Liste des génomes dans la population
			
		Returns:
			Dictionnaire des fitness ajustés (id -> fitness ajusté)
		"""
		# Calculer la matrice de distance entre tous les individus
		n = len(population)
		distances = np.zeros((n, n))
		
		for i in range(n):
			for j in range(i+1, n):
				distance = population[i].calculateCompatibilityDistance(population[j])
				distances[i, j] = distance
				distances[j, i] = distance
				
		# Calculer les facteurs de niche (sharing factors)
		sharingFactors = np.zeros(n)
		
		for i in range(n):
			for j in range(n):
				# Distance normalisée
				d = distances[i, j] / self.fitnessShareSigma
				
				# Fonction de partage: 1 - (d/sigma)² si d < sigma, sinon 0
				if d < 1.0:
					sharingFactors[i] += 1.0 - d**2
					
		# Calculer les fitness ajustés
		adjustedFitness = {}
		
		for i, genome in enumerate(population):
			# Éviter la division par zéro
			if sharingFactors[i] > 0:
				adjustedFitness[genome.id] = genome.fitness / sharingFactors[i]
			else:
				adjustedFitness[genome.id] = genome.fitness
				
		return adjustedFitness
	
	def calculateNoveltyScore(
		self,
		genome: Genome,
		population: List[Genome],
		behaviorFunction: Callable[[Genome], np.ndarray]
	) -> float:
		"""
		Calcule un score de nouveauté pour un génome donné.
		
		Args:
			genome: Génome pour lequel calculer le score
			population: Population actuelle
			behaviorFunction: Fonction qui extrait le vecteur de comportement d'un génome
			
		Returns:
			Score de nouveauté
		"""
		# Extraire le vecteur de comportement du génome cible
		behavior = behaviorFunction(genome)
		
		# Extraire les vecteurs de comportement de toute la population
		behaviors = [behaviorFunction(g) for g in population if g.id != genome.id]
		
		if not behaviors:
			return 0.0  # Pas d'autres individus pour comparer
			
		# Calculer les distances au comportement cible
		distances = [np.linalg.norm(behavior - b) for b in behaviors]
		
		# Trier les distances et prendre la moyenne des k plus proches
		k = min(15, len(distances))
		nearestDistances = sorted(distances)[:k]
		
		# Score de nouveauté = moyenne des distances aux k plus proches voisins
		noveltyScore = np.mean(nearestDistances)
		
		return noveltyScore
	
	def evaluatePopulation(
		self,
		population: List[Genome],
		fitnessFunction: Callable[[Genome], float],
		behaviorFunction: Optional[Callable[[Genome], np.ndarray]] = None
	) -> None:
		"""
		Évalue toute la population en calculant les fitness et en mettant à jour les statistiques.
		
		Args:
			population: Liste des génomes dans la population
			fitnessFunction: Fonction qui calcule le fitness de base d'un génome
			behaviorFunction: Fonction qui extrait le vecteur de comportement d'un génome (pour la nouveauté)
		"""
		# Mettre à jour le compteur de générations
		self.generationCount += 1
		
		# Réinitialiser les statistiques par espèce pour cette génération
		self.speciesStats = {}
		
		# Calculer le fitness de base pour chaque individu
		for genome in population:
			genome.fitness = fitnessFunction(genome)
			self.totalIndividualsEvaluated += 1
			
			# Initialiser les statistiques pour cette espèce si nécessaire
			if genome.speciesId not in self.speciesStats:
				self.speciesStats[genome.speciesId] = {
					"count": 0,
					"totalFitness": 0.0,
					"maxFitness": float("-inf"),
					"minFitness": float("inf")
				}
				
			# Mettre à jour les statistiques de l'espèce
			speciesStats = self.speciesStats[genome.speciesId]
			speciesStats["count"] += 1
			speciesStats["totalFitness"] += genome.fitness
			speciesStats["maxFitness"] = max(speciesStats["maxFitness"], genome.fitness)
			speciesStats["minFitness"] = min(speciesStats["minFitness"], genome.fitness)
			
		# Calculer les moyennes pour chaque espèce
		for speciesId, stats in self.speciesStats.items():
			if stats["count"] > 0:
				stats["avgFitness"] = stats["totalFitness"] / stats["count"]
			else:
				stats["avgFitness"] = 0.0
				
		# Ajouter un bonus pour la nouveauté si la fonction de comportement est fournie
		if behaviorFunction and self.noveltyBonus > 0:
			for genome in population:
				noveltyScore = self.calculateNoveltyScore(genome, population, behaviorFunction)
				genome.fitness += noveltyScore * self.noveltyBonus
	
	def adjustFitnessBySpecies(self, population: List[Genome]) -> None:
		"""
		Ajuste les fitness en fonction des statistiques par espèce pour
		maintenir l'équilibre entre les espèces.
		
		Args:
			population: Liste des génomes dans la population
		"""
		if not self.speciesStats:
			return  # Pas de statistiques disponibles
			
		# Calculer le fitness moyen de toutes les espèces
		allSpeciesAvgFitness = []
		for stats in self.speciesStats.values():
			if stats.get("count", 0) > 0:
				allSpeciesAvgFitness.append(stats["avgFitness"])
				
		if not allSpeciesAvgFitness:
			return  # Pas d'espèces avec des individus
			
		globalAvgFitness = np.mean(allSpeciesAvgFitness)
		
		# Calculer les facteurs d'ajustement par espèce
		adjustmentFactors = {}
		for speciesId, stats in self.speciesStats.items():
			if stats.get("count", 0) > 0:
				# Facteur d'ajustement: rapport entre le fitness moyen global et celui de l'espèce
				adjustmentFactors[speciesId] = globalAvgFitness / max(1e-6, stats["avgFitness"])
			else:
				adjustmentFactors[speciesId] = 1.0
				
		# Appliquer les facteurs d'ajustement à chaque individu
		for genome in population:
			factor = adjustmentFactors.get(genome.speciesId, 1.0)
			genome.fitness *= factor
	
	def getStats(self) -> Dict[str, Any]:
		"""
		Retourne les statistiques sur les sélections effectuées.
		
		Returns:
			Dictionnaire des statistiques de sélection
		"""
		return {
			"generationCount": self.generationCount,
			"totalIndividualsEvaluated": self.totalIndividualsEvaluated,
			"speciesStats": self.speciesStats
		}
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet SelectionSystem en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du système de sélection
		"""
		return {
			"selectionPressure": self.selectionPressure,
			"elitismRatio": self.elitismRatio,
			"tournamentSize": self.tournamentSize,
			"minPopulationSize": self.minPopulationSize,
			"maxPopulationSize": self.maxPopulationSize,
			"noveltyBonus": self.noveltyBonus,
			"diversityWeight": self.diversityWeight,
			"adaptabilityWeight": self.adaptabilityWeight,
			"fitnessShareSigma": self.fitnessShareSigma,
			"useFitnessSharing": self.useFitnessSharing,
			"selectionMethod": self.selectionMethod,
			"generationCount": self.generationCount,
			"totalIndividualsEvaluated": self.totalIndividualsEvaluated,
			"speciesStats": self.speciesStats
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'SelectionSystem':
		"""
		Crée une instance de SelectionSystem à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du système de sélection
			
		Returns:
			Instance de SelectionSystem reconstruite
		"""
		system = cls(
			selectionPressure=data["selectionPressure"],
			elitismRatio=data["elitismRatio"],
			tournamentSize=data["tournamentSize"],
			minPopulationSize=data["minPopulationSize"],
			maxPopulationSize=data["maxPopulationSize"],
			noveltyBonus=data["noveltyBonus"],
			diversityWeight=data["diversityWeight"],
			adaptabilityWeight=data["adaptabilityWeight"],
			fitnessShareSigma=data["fitnessShareSigma"],
			useFitnessSharing=data["useFitnessSharing"],
			selectionMethod=data["selectionMethod"]
		)
		
		system.generationCount = data["generationCount"]
		system.totalIndividualsEvaluated = data["totalIndividualsEvaluated"]
		system.speciesStats = data["speciesStats"]
		
		return system