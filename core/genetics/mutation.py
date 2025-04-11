# core/genetics/mutation.py
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np

from core.genetics.genome import Genome, Gene
from utils.serialization import Serializable


class MutationSystem(Serializable):
	"""
	Système gérant les mutations génétiques des créatures.
	Applique différents types de mutations et permet de contrôler leur fréquence et leur ampleur.
	"""
	
	def __init__(
		self,
		baseMutationRate: float = 0.1,
		geneMutationRate: float = 0.05,
		mutationMagnitude: float = 0.1,
		structuralMutationRate: float = 0.02,
		enableGeneAddition: bool = True,
		enableGeneDeletion: bool = True,
		enableGeneDisabling: bool = True,
		enableStructuralMutation: bool = True,
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise le système de mutation.
		
		Args:
			baseMutationRate: Probabilité globale qu'un génome subisse une mutation
			geneMutationRate: Probabilité qu'un gène individuel subisse une mutation
			mutationMagnitude: Ampleur des mutations des valeurs numériques
			structuralMutationRate: Probabilité de mutations structurelles (ajout/suppression de gènes)
			enableGeneAddition: Autoriser l'ajout de nouveaux gènes
			enableGeneDeletion: Autoriser la suppression de gènes
			enableGeneDisabling: Autoriser la désactivation de gènes
			enableStructuralMutation: Autoriser les mutations structurelles
			seed: Graine pour le générateur de nombres aléatoires
		"""
		self.baseMutationRate: float = baseMutationRate
		self.geneMutationRate: float = geneMutationRate
		self.mutationMagnitude: float = mutationMagnitude
		self.structuralMutationRate: float = structuralMutationRate
		self.enableGeneAddition: bool = enableGeneAddition
		self.enableGeneDeletion: bool = enableGeneDeletion
		self.enableGeneDisabling: bool = enableGeneDisabling
		self.enableStructuralMutation: bool = enableStructuralMutation
		
		# Initialiser le générateur de nombres aléatoires
		self.rng: np.random.Generator = np.random.default_rng(seed)
		
		# Statistiques de mutation
		self.totalMutations: int = 0
		self.structuralMutations: int = 0
		self.valueMutations: int = 0
		self.disableMutations: int = 0
	
	def mutateGenome(self, genome: Genome) -> int:
		"""
		Applique des mutations potentielles à un génome entier.
		
		Args:
			genome: Génome à muter
			
		Returns:
			Nombre de mutations appliquées
		"""
		# Déterminer si une mutation doit se produire
		if self.rng.random() > self.baseMutationRate:
			return 0
			
		mutationCount = 0
		
		# Appliquer potentiellement des mutations structurelles
		if self.enableStructuralMutation and self.rng.random() < self.structuralMutationRate:
			structuralMutations = self._applyStructuralMutation(genome)
			mutationCount += structuralMutations
			self.structuralMutations += structuralMutations
		
		# Muter potentiellement chaque gène individuellement
		valueMutations = 0
		for gene in genome.genes:
			if gene.enabled and self.rng.random() < self.geneMutationRate:
				# Mutation de valeurs
				if gene.mutate(self.rng):
					valueMutations += 1
					
				# Désactivation potentielle du gène
				if self.enableGeneDisabling and self.rng.random() < 0.05:  # 5% de chance
					gene.enabled = False
					self.disableMutations += 1
					mutationCount += 1
		
		mutationCount += valueMutations
		self.valueMutations += valueMutations
		self.totalMutations += mutationCount
		
		return mutationCount
	
	def _applyStructuralMutation(self, genome: Genome) -> int:
		"""
		Applique des mutations structurelles (ajout/suppression de gènes).
		
		Args:
			genome: Génome à muter
			
		Returns:
			Nombre de mutations structurelles appliquées
		"""
		mutationCount = 0
		
		# Ajout potentiel d'un nouveau gène
		if self.enableGeneAddition and self.rng.random() < 0.5:  # 50% de chance si mutation structurelle
			geneType = self.rng.choice(["morphology", "behavior", "metabolism", "environmental"])
			newGene = self._createRandomGene(geneType)
			genome.addGene(newGene)
			mutationCount += 1
		
		# Suppression potentielle d'un gène existant
		if self.enableGeneDeletion and len(genome.genes) > 1 and self.rng.random() < 0.3:  # 30% de chance
			# Choisir un gène aléatoire à supprimer
			geneIndex = self.rng.integers(0, len(genome.genes))
			geneId = genome.genes[geneIndex].id
			
			if genome.removeGene(geneId):
				mutationCount += 1
		
		return mutationCount
	
	def _createRandomGene(self, geneType: str) -> Gene:
		"""
		Crée un nouveau gène aléatoire d'un type spécifique.
		
		Args:
			geneType: Type de gène à créer
			
		Returns:
			Un nouveau gène aléatoire
		"""
		# Créer un gène selon son type
		if geneType == "morphology":
			return self._createRandomMorphologyGene()
		elif geneType == "behavior":
			return self._createRandomBehaviorGene()
		elif geneType == "metabolism":
			return self._createRandomMetabolismGene()
		elif geneType == "environmental":
			return self._createRandomEnvironmentalGene()
		else:
			raise ValueError(f"Type de gène inconnu: {geneType}")
	
	def _createRandomMorphologyGene(self) -> Gene:
		"""
		Crée un gène de morphologie aléatoire.
		
		Returns:
			Un nouveau gène de morphologie
		"""
		values = {
			"size": self.rng.uniform(0.5, 2.0),
			"bodyPlan": self.rng.choice(["fish", "jellyfish", "cephalopod", "crustacean", "eel"]),
			"limbCount": self.rng.integers(0, 8),
			"jointCount": self.rng.integers(1, 10),
			"muscleCount": self.rng.integers(2, 12),
			"sensorCount": self.rng.integers(1, 6),
			"symmetry": self.rng.uniform(0.5, 1.0),
			"density": self.rng.uniform(0.8, 1.2),
			"color": [self.rng.uniform(0, 1), self.rng.uniform(0, 1), self.rng.uniform(0, 1)],
			"texture": self.rng.choice(["smooth", "rough", "scaly", "slimy"]),
			"skeletonType": self.rng.choice(["cartilage", "bone", "exoskeleton", "hydrostatic"])
		}
		
		return Gene(
			geneType="morphology",
			values=values,
			dominance=self.rng.uniform(0.1, 1.0),
			mutationRate=self.geneMutationRate,
			mutationMagnitude=self.mutationMagnitude
		)
	
	def _createRandomBehaviorGene(self) -> Gene:
		"""
		Crée un gène de comportement aléatoire.
		
		Returns:
			Un nouveau gène de comportement
		"""
		values = {
			"aggressiveness": self.rng.uniform(0, 1),
			"sociability": self.rng.uniform(0, 1),
			"curiosity": self.rng.uniform(0, 1),
			"territoriality": self.rng.uniform(0, 1),
			"activity": self.rng.uniform(0, 1),
			"fearfulness": self.rng.uniform(0, 1),
			"explorativeness": self.rng.uniform(0, 1),
			"dietType": self.rng.choice(["carnivore", "herbivore", "omnivore", "filter_feeder"]),
			"diurnalActivity": self.rng.uniform(0, 1),
			"learningCapacity": self.rng.uniform(0, 1)
		}
		
		return Gene(
			geneType="behavior",
			values=values,
			dominance=self.rng.uniform(0.1, 1.0),
			mutationRate=self.geneMutationRate,
			mutationMagnitude=self.mutationMagnitude
		)
	
	def _createRandomMetabolismGene(self) -> Gene:
		"""
		Crée un gène de métabolisme aléatoire.
		
		Returns:
			Un nouveau gène de métabolisme
		"""
		# Générer des plages de température et de pression
		tempMin = self.rng.uniform(0, 20)
		tempMax = tempMin + self.rng.uniform(10, 20)
		
		pressureMin = self.rng.uniform(1, 5)
		pressureMax = pressureMin + self.rng.uniform(5, 20)
		
		values = {
			"basalMetabolicRate": self.rng.uniform(0.5, 1.5),
			"energyEfficiency": self.rng.uniform(0.5, 1.5),
			"temperatureTolerance": (tempMin, tempMax),
			"pressureTolerance": (pressureMin, pressureMax),
			"oxygenRequirement": self.rng.uniform(0.5, 1.5),
			"toxinResistance": self.rng.uniform(0, 1),
			"radiationResistance": self.rng.uniform(0, 1),
			"regenerationRate": self.rng.uniform(0, 0.5),
			"longevity": self.rng.uniform(0.5, 2.0),
			"maturationRate": self.rng.uniform(0.5, 1.5)
		}
		
		return Gene(
			geneType="metabolism",
			values=values,
			dominance=self.rng.uniform(0.1, 1.0),
			mutationRate=self.geneMutationRate,
			mutationMagnitude=self.mutationMagnitude
		)
	
	def _createRandomEnvironmentalGene(self) -> Gene:
		"""
		Crée un gène environnemental aléatoire.
		
		Returns:
			Un nouveau gène environnemental
		"""
		# Générer des plages de profondeur et de salinité
		depthMin = self.rng.uniform(0, 200)
		depthMax = depthMin + self.rng.uniform(50, 300)
		
		salinityMin = self.rng.uniform(25, 35)
		salinityMax = salinityMin + self.rng.uniform(5, 10)
		
		values = {
			"habitatPreference": self.rng.choice(["pelagic", "benthic", "reef", "deep_sea", "coastal"]),
			"depthRange": (depthMin, depthMax),
			"salinityTolerance": (salinityMin, salinityMax),
			"currentStrengthPreference": self.rng.uniform(0, 1),
			"substratumPreference": self.rng.choice(["sand", "mud", "rock", "coral", "seagrass"]),
			"lightSensitivity": self.rng.uniform(0, 1),
			"migratory": self.rng.uniform(0, 1),
			"seasonalAdaptation": self.rng.uniform(0, 1)
		}
		
		return Gene(
			geneType="environmental",
			values=values,
			dominance=self.rng.uniform(0.1, 1.0),
			mutationRate=self.geneMutationRate,
			mutationMagnitude=self.mutationMagnitude
		)
	
	def applyMorphologicalMutation(self, gene: Gene) -> bool:
		"""
		Applique une mutation spécifique à un gène de morphologie.
		
		Args:
			gene: Gène de morphologie à muter
			
		Returns:
			True si une mutation a été appliquée, False sinon
		"""
		if gene.geneType != "morphology" or not gene.enabled:
			return False
			
		# Sélectionner une caractéristique à muter
		availableKeys = list(gene.values.keys())
		if not availableKeys:
			return False
			
		keyToMutate = self.rng.choice(availableKeys)
		value = gene.values[keyToMutate]
		
		# Mutation dépendant du type de valeur
		if isinstance(value, (int, float)):
			if isinstance(value, int):
				# Pour les entiers, ajouter ou soustraire un petit nombre
				delta = self.rng.integers(-2, 3)  # -2, -1, 0, 1, 2
				gene.values[keyToMutate] = max(0, value + delta)
			else:
				# Pour les flottants, ajouter un bruit gaussien
				delta = self.rng.normal(0, self.mutationMagnitude)
				gene.values[keyToMutate] = value + delta
				
				# Si la valeur est normalisée (entre 0 et 1)
				if 0 <= value <= 1:
					gene.values[keyToMutate] = max(0.0, min(1.0, gene.values[keyToMutate]))
		elif isinstance(value, list) and len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
			# Pour les couleurs RGB
			for i in range(3):
				delta = self.rng.normal(0, self.mutationMagnitude)
				gene.values[keyToMutate][i] = max(0.0, min(1.0, value[i] + delta))
		elif isinstance(value, str):
			# Pour les valeurs catégorielles, changer à une autre catégorie possible
			if keyToMutate == "bodyPlan":
				options = ["fish", "jellyfish", "cephalopod", "crustacean", "eel"]
			elif keyToMutate == "texture":
				options = ["smooth", "rough", "scaly", "slimy"]
			elif keyToMutate == "skeletonType":
				options = ["cartilage", "bone", "exoskeleton", "hydrostatic"]
			else:
				return False
				
			# Exclure la valeur actuelle des options
			options = [opt for opt in options if opt != value]
			if options:
				gene.values[keyToMutate] = self.rng.choice(options)
			
		return True
	
	def getStats(self) -> Dict[str, int]:
		"""
		Retourne les statistiques sur les mutations appliquées.
		
		Returns:
			Dictionnaire des statistiques de mutation
		"""
		return {
			"totalMutations": self.totalMutations,
			"structuralMutations": self.structuralMutations,
			"valueMutations": self.valueMutations,
			"disableMutations": self.disableMutations
		}
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet MutationSystem en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du système de mutation
		"""
		return {
			"baseMutationRate": self.baseMutationRate,
			"geneMutationRate": self.geneMutationRate,
			"mutationMagnitude": self.mutationMagnitude,
			"structuralMutationRate": self.structuralMutationRate,
			"enableGeneAddition": self.enableGeneAddition,
			"enableGeneDeletion": self.enableGeneDeletion,
			"enableGeneDisabling": self.enableGeneDisabling,
			"enableStructuralMutation": self.enableStructuralMutation,
			"totalMutations": self.totalMutations,
			"structuralMutations": self.structuralMutations,
			"valueMutations": self.valueMutations,
			"disableMutations": self.disableMutations
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'MutationSystem':
		"""
		Crée une instance de MutationSystem à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du système de mutation
			
		Returns:
			Instance de MutationSystem reconstruite
		"""
		system = cls(
			baseMutationRate=data["baseMutationRate"],
			geneMutationRate=data["geneMutationRate"],
			mutationMagnitude=data["mutationMagnitude"],
			structuralMutationRate=data["structuralMutationRate"],
			enableGeneAddition=data["enableGeneAddition"],
			enableGeneDeletion=data["enableGeneDeletion"],
			enableGeneDisabling=data["enableGeneDisabling"],
			enableStructuralMutation=data["enableStructuralMutation"]
		)
		
		system.totalMutations = data["totalMutations"]
		system.structuralMutations = data["structuralMutations"]
		system.valueMutations = data["valueMutations"]
		system.disableMutations = data["disableMutations"]
		
		return system