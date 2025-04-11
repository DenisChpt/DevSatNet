# learning/trainers/population_trainer.py
from typing import Dict, List, Tuple, Any, Optional, Callable
import numpy as np
import uuid
import copy
import time
import threading
import concurrent.futures
import torch
from collections import defaultdict

from core.genetics.genome import Genome, createInitialGenome
from core.genetics.mutation import MutationSystem
from core.genetics.crossover import CrossoverSystem
from core.genetics.selection import SelectionSystem
from learning.models.creature_brain import CreatureBrain
from utils.serialization import Serializable


class PopulationTrainer(Serializable):
	"""
	Système d'entraînement au niveau de la population qui gère l'évolution
	des créatures marines à travers la sélection naturelle et l'apprentissage.
	"""
	
	def __init__(
		self,
		populationSize: int = 100,
		speciesConfig: Dict[str, Dict[str, Any]] = None,
		evaluationSteps: int = 1000,
		generationsLimit: int = 100,
		fitnessThreshold: float = 100.0,
		parallelEvaluations: bool = True,
		maxWorkers: int = 4,
		selectionConfig: Dict[str, Any] = None,
		mutationConfig: Dict[str, Any] = None,
		crossoverConfig: Dict[str, Any] = None,
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise le système d'entraînement de population.
		
		Args:
			populationSize: Taille de la population totale
			speciesConfig: Configuration des espèces initiales
			evaluationSteps: Nombre d'étapes pour l'évaluation de chaque individu
			generationsLimit: Nombre maximal de générations
			fitnessThreshold: Seuil de fitness pour terminer l'évolution
			parallelEvaluations: Utiliser des évaluations parallèles
			maxWorkers: Nombre maximal de workers pour l'évaluation parallèle
			selectionConfig: Configuration du système de sélection
			mutationConfig: Configuration du système de mutation
			crossoverConfig: Configuration du système de croisement
			seed: Graine pour le générateur de nombres aléatoires
		"""
		self.id: str = str(uuid.uuid4())
		self.populationSize: int = populationSize
		self.speciesConfig: Dict[str, Dict[str, Any]] = speciesConfig if speciesConfig else {}
		self.evaluationSteps: int = evaluationSteps
		self.generationsLimit: int = generationsLimit
		self.fitnessThreshold: float = fitnessThreshold
		self.parallelEvaluations: bool = parallelEvaluations
		self.maxWorkers: int = maxWorkers
		
		# Initialiser le générateur de nombres aléatoires
		self.seed: int = seed if seed is not None else int(time.time())
		self.rng: np.random.Generator = np.random.default_rng(self.seed)
		
		# Systèmes génétiques
		self.selectionSystem: SelectionSystem = self._initSelectionSystem(selectionConfig)
		self.mutationSystem: MutationSystem = self._initMutationSystem(mutationConfig)
		self.crossoverSystem: CrossoverSystem = self._initCrossoverSystem(crossoverConfig)
		
		# État de la population
		self.population: List[Genome] = []
		self.species: Dict[str, Dict[str, Any]] = {}
		self.hallOfFame: List[Genome] = []  # Meilleurs individus de tous les temps
		
		# Statistiques d'évolution
		self.generationStats: List[Dict[str, Any]] = []
		self.currentGeneration: int = 0
		self.bestFitness: float = float("-inf")
		self.bestGenome: Optional[Genome] = None
		self.evaluationTime: float = 0.0
		self.reproductionTime: float = 0.0
		
		# Environnement et évaluation
		self.environmentFactory: Optional[Callable[[], Any]] = None
		self.brainFactory: Optional[Callable[[Genome], CreatureBrain]] = None
		
		# Verrou pour l'accès concurrent
		self.lock = threading.Lock()
	
	def _initSelectionSystem(self, config: Optional[Dict[str, Any]]) -> SelectionSystem:
		"""
		Initialise le système de sélection.
		
		Args:
			config: Configuration du système de sélection
			
		Returns:
			Instance de SelectionSystem configurée
		"""
		if config is None:
			config = {}
			
		return SelectionSystem(
			selectionPressure=config.get("selectionPressure", 1.5),
			elitismRatio=config.get("elitismRatio", 0.1),
			tournamentSize=config.get("tournamentSize", 3),
			minPopulationSize=config.get("minPopulationSize", 20),
			maxPopulationSize=config.get("maxPopulationSize", 1000),
			noveltyBonus=config.get("noveltyBonus", 0.2),
			diversityWeight=config.get("diversityWeight", 0.3),
			adaptabilityWeight=config.get("adaptabilityWeight", 0.2),
			fitnessShareSigma=config.get("fitnessShareSigma", 0.3),
			useFitnessSharing=config.get("useFitnessSharing", True),
			selectionMethod=config.get("selectionMethod", "tournament"),
			seed=self.seed
		)
	
	def _initMutationSystem(self, config: Optional[Dict[str, Any]]) -> MutationSystem:
		"""
		Initialise le système de mutation.
		
		Args:
			config: Configuration du système de mutation
			
		Returns:
			Instance de MutationSystem configurée
		"""
		if config is None:
			config = {}
			
		return MutationSystem(
			baseMutationRate=config.get("baseMutationRate", 0.1),
			geneMutationRate=config.get("geneMutationRate", 0.05),
			mutationMagnitude=config.get("mutationMagnitude", 0.1),
			structuralMutationRate=config.get("structuralMutationRate", 0.02),
			enableGeneAddition=config.get("enableGeneAddition", True),
			enableGeneDeletion=config.get("enableGeneDeletion", True),
			enableGeneDisabling=config.get("enableGeneDisabling", True),
			enableStructuralMutation=config.get("enableStructuralMutation", True),
			seed=self.seed
		)
	
	def _initCrossoverSystem(self, config: Optional[Dict[str, Any]]) -> CrossoverSystem:
		"""
		Initialise le système de croisement.
		
		Args:
			config: Configuration du système de croisement
			
		Returns:
			Instance de CrossoverSystem configurée
		"""
		if config is None:
			config = {}
			
		return CrossoverSystem(
			crossoverRate=config.get("crossoverRate", 0.7),
			uniformCrossoverProbability=config.get("uniformCrossoverProbability", 0.5),
			geneInheritanceRate=config.get("geneInheritanceRate", 0.5),
			inheritDisabledGenes=config.get("inheritDisabledGenes", True),
			hybridizationPenalty=config.get("hybridizationPenalty", 0.2),
			seed=self.seed
		)
	
	def initializePopulation(self) -> None:
		"""
		Initialise la population avec des génomes aléatoires pour chaque espèce.
		"""
		self.population = []
		
		# Si aucune espèce n'est configurée, en créer une par défaut
		if not self.speciesConfig:
			self.speciesConfig = {
				"default": {
					"initialCount": self.populationSize,
					"morphologyGenes": 3,
					"behaviorGenes": 2,
					"metabolismGenes": 2,
					"environmentalGenes": 2
				}
			}
			
		# Créer les génomes initiaux pour chaque espèce
		for speciesId, config in self.speciesConfig.items():
			# Nombre d'individus pour cette espèce
			initialCount = config.get("initialCount", 10)
			
			# Nombres de gènes par type
			morphologyCount = config.get("morphologyGenes", 3)
			behaviorCount = config.get("behaviorGenes", 2)
			metabolismCount = config.get("metabolismGenes", 2)
			environmentalCount = config.get("environmentalGenes", 2)
			
			# Créer les génomes
			for _ in range(initialCount):
				genome = createInitialGenome(
					speciesId=speciesId,
					morphologyCount=morphologyCount,
					behaviorCount=behaviorCount,
					metabolismCount=metabolismCount,
					environmentalCount=environmentalCount,
					rng=self.rng
				)
				
				self.population.append(genome)
				
			# Initialiser les statistiques de l'espèce
			self.species[speciesId] = {
				"count": initialCount,
				"avgFitness": 0.0,
				"bestFitness": float("-inf"),
				"bestGenomeId": None,
				"extinctionCounter": 0
			}
		
		# Mélanger la population
		self.rng.shuffle(self.population)
		
		# Initialiser les statistiques
		self.currentGeneration = 0
		self.bestFitness = float("-inf")
		self.bestGenome = None
		self.generationStats = []
	
	def setBrainFactory(self, factory: Callable[[Genome], CreatureBrain]) -> None:
		"""
		Définit la fonction de création de cerveaux à partir des génomes.
		
		Args:
			factory: Fonction qui prend un génome et retourne un cerveau
		"""
		self.brainFactory = factory
	
	def setEnvironmentFactory(self, factory: Callable[[], Any]) -> None:
		"""
		Définit la fonction de création d'environnements pour l'évaluation.
		
		Args:
			factory: Fonction qui crée et retourne un environnement
		"""
		self.environmentFactory = factory
	
	def evaluateGenome(self, genome: Genome) -> float:
		"""
		Évalue un génome en créant un cerveau et en le testant dans l'environnement.
		
		Args:
			genome: Génome à évaluer
			
		Returns:
			Score de fitness du génome
		"""
		if self.brainFactory is None or self.environmentFactory is None:
			raise ValueError("Brain factory and environment factory must be set before evaluation")
		
		# Créer le cerveau à partir du génome
		brain = self.brainFactory(genome)
		
		# Créer un environnement d'évaluation
		env = self.environmentFactory()
		
		# Évaluer le cerveau dans l'environnement
		totalReward = 0.0
		state = env.reset()
		
		for _ in range(self.evaluationSteps):
			# Convertir l'état en tensor
			stateTensor = torch.FloatTensor(state)
			
			# Obtenir l'action du cerveau
			with torch.no_grad():
				action = brain.selectAction(stateTensor)
				
			# Si le cerveau retourne un tensor, le convertir en numpy
			if torch.is_tensor(action):
				action = action.numpy()
				
			# Exécuter l'action dans l'environnement
			nextState, reward, done, info = env.step(action)
			
			# Accumuler la récompense
			totalReward += reward
			
			# Mettre à jour l'état
			state = nextState
			
			# Arrêter si l'épisode est terminé
			if done:
				break
		
		# Nettoyer les ressources
		env.close()
		
		# Mettre à jour le fitness du génome
		genome.fitness = totalReward
		
		return totalReward
	
	def evaluatePopulation(self) -> Dict[str, Any]:
		"""
		Évalue toute la population.
		
		Returns:
			Statistiques d'évaluation
		"""
		startTime = time.time()
		
		# Réinitialiser les fitness
		for genome in self.population:
			genome.fitness = 0.0
		
		# Évaluer chaque individu
		if self.parallelEvaluations:
			# Évaluation parallèle
			with concurrent.futures.ThreadPoolExecutor(max_workers=self.maxWorkers) as executor:
				# Soumettre les tâches d'évaluation
				futureToGenome = {executor.submit(self.evaluateGenome, genome): genome for genome in self.population}
				
				# Traiter les résultats au fur et à mesure de leur complétion
				for future in concurrent.futures.as_completed(futureToGenome):
					genome = futureToGenome[future]
					try:
						fitness = future.result()
					except Exception as e:
						print(f"Evaluation failed for genome {genome.id}: {e}")
						genome.fitness = float("-inf")
		else:
			# Évaluation séquentielle
			for genome in self.population:
				try:
					self.evaluateGenome(genome)
				except Exception as e:
					print(f"Evaluation failed for genome {genome.id}: {e}")
					genome.fitness = float("-inf")
		
		# Mettre à jour les statistiques par espèce
		speciesStats = defaultdict(lambda: {"count": 0, "totalFitness": 0.0, "bestFitness": float("-inf"), "bestGenomeId": None})
		
		for genome in self.population:
			stats = speciesStats[genome.speciesId]
			stats["count"] += 1
			stats["totalFitness"] += genome.fitness
			
			if genome.fitness > stats["bestFitness"]:
				stats["bestFitness"] = genome.fitness
				stats["bestGenomeId"] = genome.id
		
		# Calculer les moyennes et mettre à jour les statistiques d'espèce
		for speciesId, stats in speciesStats.items():
			if stats["count"] > 0:
				stats["avgFitness"] = stats["totalFitness"] / stats["count"]
			else:
				stats["avgFitness"] = 0.0
				
			if speciesId in self.species:
				self.species[speciesId].update(stats)
				
				# Incrémenter le compteur d'extinction si l'espèce est en voie de disparition
				if stats["count"] < 5:
					self.species[speciesId]["extinctionCounter"] += 1
				else:
					self.species[speciesId]["extinctionCounter"] = 0
			else:
				# Nouvelle espèce
				self.species[speciesId] = stats
				self.species[speciesId]["extinctionCounter"] = 0
		
		# Trouver le meilleur génome global
		bestGenome = max(self.population, key=lambda g: g.fitness)
		currentBestFitness = bestGenome.fitness
		
		if currentBestFitness > self.bestFitness:
			self.bestFitness = currentBestFitness
			self.bestGenome = copy.deepcopy(bestGenome)
			
			# Ajouter au Hall of Fame si c'est un des meilleurs de tous les temps
			if len(self.hallOfFame) < 10 or currentBestFitness > min(g.fitness for g in self.hallOfFame):
				self.hallOfFame.append(copy.deepcopy(bestGenome))
				self.hallOfFame.sort(key=lambda g: g.fitness, reverse=True)
				
				# Limiter la taille du Hall of Fame
				if len(self.hallOfFame) > 10:
					self.hallOfFame = self.hallOfFame[:10]
		
		# Calculer le temps d'évaluation
		evaluationTime = time.time() - startTime
		self.evaluationTime += evaluationTime
		
		# Statistiques d'évaluation
		evaluationStats = {
			"totalEvaluated": len(self.population),
			"avgFitness": sum(g.fitness for g in self.population) / len(self.population) if self.population else 0.0,
			"bestFitness": currentBestFitness,
			"bestGenomeId": bestGenome.id,
			"speciesStats": {sid: {k: v for k, v in s.items() if k != "bestGenomeId"} for sid, s in self.species.items()},
			"evaluationTime": evaluationTime
		}
		
		return evaluationStats
	
	def evolvePopulation(self) -> Dict[str, Any]:
		"""
		Fait évoluer la population vers une nouvelle génération.
		
		Returns:
			Statistiques de reproduction
		"""
		startTime = time.time()
		
		# Évaluer la population si ce n'est pas déjà fait
		if all(g.fitness == 0.0 for g in self.population):
			self.evaluatePopulation()
		
		# Ajuster les fitness en fonction des espèces pour maintenir la diversité
		self.selectionSystem.adjustFitnessBySpecies(self.population)
		
		# Nombre d'individus à sélectionner pour la reproduction
		numParents = self.populationSize // 2
		
		# Sélectionner les parents
		parents = self.selectionSystem.selectParents(self.population, numParents)
		
		# Effectuer le croisement et la mutation pour créer la nouvelle génération
		offspring = []
		
		# Créer des couples de parents pour la reproduction
		for i in range(0, len(parents), 2):
			if i + 1 < len(parents):
				parent1 = parents[i]
				parent2 = parents[i + 1]
				
				# Croisement
				child1, child2 = self.crossoverSystem.crossover(parent1, parent2)
				
				# Mutation
				self.mutationSystem.mutateGenome(child1)
				self.mutationSystem.mutateGenome(child2)
				
				offspring.extend([child1, child2])
		
		# Si nombre impair de parents, le dernier se reproduit avec un individu aléatoire
		if len(parents) % 2 != 0:
			lastParent = parents[-1]
			randomParent = self.rng.choice(parents[:-1])
			
			# Croisement
			child1, child2 = self.crossoverSystem.crossover(lastParent, randomParent)
			
			# Mutation
			self.mutationSystem.mutateGenome(child1)
			self.mutationSystem.mutateGenome(child2)
			
			offspring.extend([child1, child2])
		
		# Sélectionner les survivants pour la prochaine génération
		survivors = self.selectionSystem.selectSurvivors(
			self.population,
			offspring,
			self.populationSize
		)
		
		# Mettre à jour la population
		self.population = survivors
		
		# Incrémenter le compteur de générations
		self.currentGeneration += 1
		
		# Calculer le temps de reproduction
		reproductionTime = time.time() - startTime
		self.reproductionTime += reproductionTime
		
		# Statistiques de reproduction
		reproductionStats = {
			"parentsSelected": len(parents),
			"offspringCreated": len(offspring),
			"survivorsSelected": len(survivors),
			"mutationStats": self.mutationSystem.getStats(),
			"crossoverStats": self.crossoverSystem.getStats(),
			"reproductionTime": reproductionTime
		}
		
		return reproductionStats
	
	def runEvolution(
		self,
		numGenerations: Optional[int] = None,
		targetFitness: Optional[float] = None,
		callback: Optional[Callable[[Dict[str, Any]], None]] = None
	) -> Dict[str, Any]:
		"""
		Exécute le processus d'évolution pour plusieurs générations.
		
		Args:
			numGenerations: Nombre de générations à exécuter (None = utiliser generationsLimit)
			targetFitness: Fitness cible pour arrêter l'évolution
			callback: Fonction de rappel appelée après chaque génération
			
		Returns:
			Statistiques finales d'évolution
		"""
		if numGenerations is None:
			numGenerations = self.generationsLimit
			
		if targetFitness is None:
			targetFitness = self.fitnessThreshold
			
		# Initialiser la population si ce n'est pas déjà fait
		if not self.population:
			self.initializePopulation()
			
		# Heure de début
		startTime = time.time()
		
		# Statistiques globales
		evolutionStats = {
			"initialPopulationSize": len(self.population),
			"generations": [],
			"bestFitness": float("-inf"),
			"bestGenomeId": None,
			"converged": False,
			"evolutionTime": 0.0
		}
		
		# Boucle principale d'évolution
		for generation in range(numGenerations):
			# Évaluer la population
			evaluationStats = self.evaluatePopulation()
			
			# Vérifier si la cible de fitness est atteinte
			if evaluationStats["bestFitness"] >= targetFitness:
				print(f"Target fitness {targetFitness} reached at generation {self.currentGeneration}")
				evolutionStats["converged"] = True
				break
				
			# Faire évoluer la population
			reproductionStats = self.evolvePopulation()
			
			# Mettre à jour les statistiques de génération
			generationStats = {
				"generation": self.currentGeneration,
				"populationSize": len(self.population),
				"evaluation": evaluationStats,
				"reproduction": reproductionStats
			}
			
			self.generationStats.append(generationStats)
			evolutionStats["generations"].append(generationStats)
			
			# Mettre à jour le meilleur fitness
			if evaluationStats["bestFitness"] > evolutionStats["bestFitness"]:
				evolutionStats["bestFitness"] = evaluationStats["bestFitness"]
				evolutionStats["bestGenomeId"] = evaluationStats["bestGenomeId"]
			
			# Afficher les progrès
			print(f"Generation {self.currentGeneration}: Best fitness = {evaluationStats['bestFitness']:.2f}, Avg fitness = {evaluationStats['avgFitness']:.2f}")
			
			# Appeler la fonction de rappel
			if callback:
				callback(generationStats)
				
			# Vérifier si le nombre maximal de générations est atteint
			if self.currentGeneration >= self.generationsLimit:
				print(f"Maximum generations {self.generationsLimit} reached")
				break
		
		# Calculer le temps total d'évolution
		evolutionTime = time.time() - startTime
		evolutionStats["evolutionTime"] = evolutionTime
		
		return evolutionStats
	
	def getSpeciesStats(self) -> Dict[str, Dict[str, Any]]:
		"""
		Retourne les statistiques actuelles pour chaque espèce.
		
		Returns:
			Dictionnaire des statistiques par espèce
		"""
		return self.species
	
	def getStats(self) -> Dict[str, Any]:
		"""
		Retourne toutes les statistiques d'évolution.
		
		Returns:
			Dictionnaire des statistiques d'évolution
		"""
		return {
			"currentGeneration": self.currentGeneration,
			"populationSize": len(self.population),
			"bestFitness": self.bestFitness,
			"bestGenomeId": self.bestGenome.id if self.bestGenome else None,
			"speciesStats": self.getSpeciesStats(),
			"evaluationTime": self.evaluationTime,
			"reproductionTime": self.reproductionTime,
			"totalEvolutionTime": self.evaluationTime + self.reproductionTime,
			"mutationStats": self.mutationSystem.getStats(),
			"crossoverStats": self.crossoverSystem.getStats(),
			"selectionStats": self.selectionSystem.getStats(),
			"hallOfFameSize": len(self.hallOfFame)
		}
	
	def getBestGenome(self) -> Optional[Genome]:
		"""
		Retourne le meilleur génome de toutes les générations.
		
		Returns:
			Meilleur génome ou None si aucun génome n'a été évalué
		"""
		return self.bestGenome
	
	def getHallOfFame(self) -> List[Genome]:
		"""
		Retourne le hall of fame des meilleurs génomes.
		
		Returns:
			Liste des meilleurs génomes
		"""
		return self.hallOfFame
	
	def getGenomeById(self, genomeId: str) -> Optional[Genome]:
		"""
		Trouve un génome par son ID dans la population actuelle.
		
		Args:
			genomeId: ID du génome à trouver
			
		Returns:
			Génome correspondant ou None s'il n'est pas trouvé
		"""
		for genome in self.population:
			if genome.id == genomeId:
				return genome
				
		# Vérifier dans le hall of fame
		for genome in self.hallOfFame:
			if genome.id == genomeId:
				return genome
				
		return None
	
	def save(self, path: str) -> None:
		"""
		Sauvegarde l'état de l'évolution.
		
		Args:
			path: Chemin de sauvegarde
		"""
		# Convertir l'objet en dictionnaire
		data = self.toDict()
		
		# Sauvegarder au format JSON
		import json
		with open(path, 'w') as f:
			json.dump(data, f, indent=2)
	
	def load(self, path: str) -> None:
		"""
		Charge l'état d'évolution depuis un fichier.
		
		Args:
			path: Chemin du fichier de sauvegarde
		"""
		# Charger le dictionnaire depuis le fichier JSON
		import json
		with open(path, 'r') as f:
			data = json.load(f)
			
		# Reconstruire l'objet à partir du dictionnaire
		self._fromDict(data)
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet PopulationTrainer en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de l'entraîneur de population
		"""
		return {
			"id": self.id,
			"populationSize": self.populationSize,
			"speciesConfig": self.speciesConfig,
			"evaluationSteps": self.evaluationSteps,
			"generationsLimit": self.generationsLimit,
			"fitnessThreshold": self.fitnessThreshold,
			"parallelEvaluations": self.parallelEvaluations,
			"maxWorkers": self.maxWorkers,
			"seed": self.seed,
			"selectionSystem": self.selectionSystem.toDict(),
			"mutationSystem": self.mutationSystem.toDict(),
			"crossoverSystem": self.crossoverSystem.toDict(),
			"population": [genome.toDict() for genome in self.population],
			"species": self.species,
			"hallOfFame": [genome.toDict() for genome in self.hallOfFame],
			"generationStats": self.generationStats,
			"currentGeneration": self.currentGeneration,
			"bestFitness": self.bestFitness,
			"bestGenome": self.bestGenome.toDict() if self.bestGenome else None,
			"evaluationTime": self.evaluationTime,
			"reproductionTime": self.reproductionTime
		}
	
	def _fromDict(self, data: Dict[str, Any]) -> None:
		"""
		Remplit l'objet à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de l'entraîneur de population
		"""
		self.id = data["id"]
		self.populationSize = data["populationSize"]
		self.speciesConfig = data["speciesConfig"]
		self.evaluationSteps = data["evaluationSteps"]
		self.generationsLimit = data["generationsLimit"]
		self.fitnessThreshold = data["fitnessThreshold"]
		self.parallelEvaluations = data["parallelEvaluations"]
		self.maxWorkers = data["maxWorkers"]
		self.seed = data["seed"]
		
		# Recréer les systèmes
		self.selectionSystem = SelectionSystem.fromDict(data["selectionSystem"])
		self.mutationSystem = MutationSystem.fromDict(data["mutationSystem"])
		self.crossoverSystem = CrossoverSystem.fromDict(data["crossoverSystem"])
		
		# Recréer la population
		self.population = [Genome.fromDict(g) for g in data["population"]]
		
		# Restaurer les espèces
		self.species = data["species"]
		
		# Recréer le hall of fame
		self.hallOfFame = [Genome.fromDict(g) for g in data["hallOfFame"]]
		
		# Restaurer les statistiques
		self.generationStats = data["generationStats"]
		self.currentGeneration = data["currentGeneration"]
		self.bestFitness = data["bestFitness"]
		
		# Recréer le meilleur génome
		if data["bestGenome"]:
			self.bestGenome = Genome.fromDict(data["bestGenome"])
		else:
			self.bestGenome = None
			
		self.evaluationTime = data["evaluationTime"]
		self.reproductionTime = data["reproductionTime"]
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'PopulationTrainer':
		"""
		Crée une instance de PopulationTrainer à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de l'entraîneur
			
		Returns:
			Instance de PopulationTrainer reconstruite
		"""
		trainer = cls(
			populationSize=data["populationSize"],
			speciesConfig=data["speciesConfig"],
			evaluationSteps=data["evaluationSteps"],
			generationsLimit=data["generationsLimit"],
			fitnessThreshold=data["fitnessThreshold"],
			parallelEvaluations=data["parallelEvaluations"],
			maxWorkers=data["maxWorkers"],
			seed=data["seed"]
		)
		
		# Compléter la reconstruction
		trainer._fromDict(data)
		
		return trainer