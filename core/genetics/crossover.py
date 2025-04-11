# core/genetics/crossover.py
from typing import Dict, List, Tuple, Any, Optional, Set
import numpy as np
import uuid
import copy

from core.genetics.genome import Genome, Gene
from utils.serialization import Serializable


class CrossoverSystem(Serializable):
	"""
	Système gérant les croisements génétiques entre créatures.
	Permet de contrôler comment les génomes sont combinés durant la reproduction.
	"""
	
	def __init__(
		self,
		crossoverRate: float = 0.7,
		uniformCrossoverProbability: float = 0.5,
		geneInheritanceRate: float = 0.5,
		inheritDisabledGenes: bool = True,
		hybridizationPenalty: float = 0.2,
		seed: Optional[int] = None
	) -> None:
		"""
		Initialise le système de croisement.
		
		Args:
			crossoverRate: Probabilité qu'un croisement se produise
			uniformCrossoverProbability: Probabilité d'utiliser un croisement uniforme vs à un point
			geneInheritanceRate: Probabilité d'hériter d'un gène spécifique lors d'un croisement uniforme
			inheritDisabledGenes: Autoriser l'héritage de gènes désactivés
			hybridizationPenalty: Pénalité de fitness pour les hybrides entre espèces différentes
			seed: Graine pour le générateur de nombres aléatoires
		"""
		self.crossoverRate: float = crossoverRate
		self.uniformCrossoverProbability: float = uniformCrossoverProbability
		self.geneInheritanceRate: float = geneInheritanceRate
		self.inheritDisabledGenes: bool = inheritDisabledGenes
		self.hybridizationPenalty: float = hybridizationPenalty
		
		# Initialiser le générateur de nombres aléatoires
		self.rng: np.random.Generator = np.random.default_rng(seed)
		
		# Statistiques
		self.totalCrossovers: int = 0
		self.uniformCrossovers: int = 0
		self.onePointCrossovers: int = 0
		self.interspeciesCrossovers: int = 0
	
	def crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
		"""
		Réalise un croisement entre deux génomes parentaux.
		
		Args:
			parent1: Premier génome parental
			parent2: Second génome parental
			
		Returns:
			Deux génomes enfants issus du croisement
		"""
		# Vérifier si le croisement doit avoir lieu
		if self.rng.random() > self.crossoverRate:
			# Pas de croisement, retourner des clones des parents avec des mutations
			child1 = copy.deepcopy(parent1)
			child1.id = str(uuid.uuid4())
			child1.parentIds = [parent1.id]
			child1.childrenIds = []
			
			child2 = copy.deepcopy(parent2)
			child2.id = str(uuid.uuid4())
			child2.parentIds = [parent2.id]
			child2.childrenIds = []
			
			return child1, child2
		
		# Déterminer le type de croisement à appliquer
		useUniformCrossover = self.rng.random() < self.uniformCrossoverProbability
		
		# Mettre à jour les statistiques
		self.totalCrossovers += 1
		if useUniformCrossover:
			self.uniformCrossovers += 1
		else:
			self.onePointCrossovers += 1
			
		# Vérifier si c'est un croisement inter-espèces
		isInterspeciesCrossover = parent1.speciesId != parent2.speciesId
		if isInterspeciesCrossover:
			self.interspeciesCrossovers += 1
		
		# Préparer les génomes enfants
		child1 = Genome(
			speciesId=parent1.speciesId,  # L'enfant appartient à l'espèce du premier parent
			genes=[],
			generation=max(parent1.generation, parent2.generation) + 1
		)
		
		child2 = Genome(
			speciesId=parent2.speciesId,  # L'enfant appartient à l'espèce du second parent
			genes=[],
			generation=max(parent1.generation, parent2.generation) + 1
		)
		
		# Enregistrer les liens de parenté
		child1.parentIds = [parent1.id, parent2.id]
		child2.parentIds = [parent1.id, parent2.id]
		
		parent1.childrenIds.append(child1.id)
		parent1.childrenIds.append(child2.id)
		parent2.childrenIds.append(child1.id)
		parent2.childrenIds.append(child2.id)
		
		# Si c'est un croisement inter-espèces, les enfants peuvent hériter de traits des deux espèces
		if isInterspeciesCrossover:
			# Le premier enfant peut avoir une chance d'appartenir à l'espèce du second parent
			if self.rng.random() < 0.5:
				child1.speciesId = parent2.speciesId
				
			# Le second enfant peut avoir une chance d'appartenir à l'espèce du premier parent
			if self.rng.random() < 0.5:
				child2.speciesId = parent1.speciesId
				
			# Appliquer une pénalité de fitness pour les hybrides
			child1.metadata["hybridizationPenalty"] = self.hybridizationPenalty
			child2.metadata["hybridizationPenalty"] = self.hybridizationPenalty
		
		# Effectuer le croisement spécifique
		if useUniformCrossover:
			self._uniformCrossover(parent1, parent2, child1, child2)
		else:
			self._onePointCrossover(parent1, parent2, child1, child2)
			
		return child1, child2
	
	def _uniformCrossover(self, parent1: Genome, parent2: Genome, child1: Genome, child2: Genome) -> None:
		"""
		Réalise un croisement uniforme entre deux génomes parentaux.
		Chaque gène est hérité aléatoirement de l'un des parents.
		
		Args:
			parent1: Premier génome parental
			parent2: Second génome parental
			child1: Premier génome enfant à remplir
			child2: Second génome enfant à remplir
		"""
		# Regrouper les gènes par type
		parent1GenesByType = parent1.genesByType
		parent2GenesByType = parent2.genesByType
		
		# Liste de tous les types de gènes présents dans les deux génomes
		allGeneTypes = set(parent1GenesByType.keys()) | set(parent2GenesByType.keys())
		
		for geneType in allGeneTypes:
			parent1Genes = parent1GenesByType.get(geneType, [])
			parent2Genes = parent2GenesByType.get(geneType, [])
			
			# Filtrer les gènes désactivés si nécessaire
			if not self.inheritDisabledGenes:
				parent1Genes = [g for g in parent1Genes if g.enabled]
				parent2Genes = [g for g in parent2Genes if g.enabled]
				
			# Traiter séparément les cas où un parent n'a pas de gènes de ce type
			if not parent1Genes and not parent2Genes:
				continue
				
			if not parent1Genes:
				# Le second parent a des gènes mais pas le premier
				for gene in parent2Genes:
					# Chaque enfant a une chance indépendante d'hériter ce gène
					if self.rng.random() < self.geneInheritanceRate:
						child1.addGene(copy.deepcopy(gene))
					if self.rng.random() < self.geneInheritanceRate:
						child2.addGene(copy.deepcopy(gene))
				continue
				
			if not parent2Genes:
				# Le premier parent a des gènes mais pas le second
				for gene in parent1Genes:
					# Chaque enfant a une chance indépendante d'hériter ce gène
					if self.rng.random() < self.geneInheritanceRate:
						child1.addGene(copy.deepcopy(gene))
					if self.rng.random() < self.geneInheritanceRate:
						child2.addGene(copy.deepcopy(gene))
				continue
			
			# Les deux parents ont des gènes de ce type
			# Créer un pool combiné de gènes
			allGenes = parent1Genes + parent2Genes
			
			# Pour chaque gène, décider s'il est hérité par chaque enfant
			for gene in allGenes:
				inherit1 = self.rng.random() < self.geneInheritanceRate
				inherit2 = self.rng.random() < self.geneInheritanceRate
				
				if inherit1:
					child1.addGene(copy.deepcopy(gene))
				if inherit2:
					child2.addGene(copy.deepcopy(gene))
	
	def _onePointCrossover(self, parent1: Genome, parent2: Genome, child1: Genome, child2: Genome) -> None:
		"""
		Réalise un croisement à un point entre deux génomes parentaux.
		Les gènes sont divisés à un point aléatoire, le premier enfant hérite
		des gènes avant ce point du premier parent et après du second,
		tandis que le second enfant fait l'inverse.
		
		Args:
			parent1: Premier génome parental
			parent2: Second génome parental
			child1: Premier génome enfant à remplir
			child2: Second génome enfant à remplir
		"""
		# Regrouper les gènes par type
		parent1GenesByType = parent1.genesByType
		parent2GenesByType = parent2.genesByType
		
		# Liste de tous les types de gènes présents dans les deux génomes
		allGeneTypes = set(parent1GenesByType.keys()) | set(parent2GenesByType.keys())
		
		for geneType in allGeneTypes:
			parent1Genes = parent1GenesByType.get(geneType, [])
			parent2Genes = parent2GenesByType.get(geneType, [])
			
			# Filtrer les gènes désactivés si nécessaire
			if not self.inheritDisabledGenes:
				parent1Genes = [g for g in parent1Genes if g.enabled]
				parent2Genes = [g for g in parent2Genes if g.enabled]
				
			# Traiter séparément les cas où un parent n'a pas de gènes de ce type
			if not parent1Genes and not parent2Genes:
				continue
				
			if not parent1Genes:
				# Le second parent a des gènes mais pas le premier
				# Point de croisement pour le second parent
				crossoverPoint2 = self.rng.integers(0, len(parent2Genes) + 1)
				
				# Premier enfant: aucun gène du premier segment, tous les gènes du second segment
				for i, gene in enumerate(parent2Genes):
					if i >= crossoverPoint2:
						child1.addGene(copy.deepcopy(gene))
						
				# Second enfant: tous les gènes du premier segment, aucun gène du second segment
				for i, gene in enumerate(parent2Genes):
					if i < crossoverPoint2:
						child2.addGene(copy.deepcopy(gene))
				continue
				
			if not parent2Genes:
				# Le premier parent a des gènes mais pas le second
				# Point de croisement pour le premier parent
				crossoverPoint1 = self.rng.integers(0, len(parent1Genes) + 1)
				
				# Premier enfant: tous les gènes du premier segment, aucun gène du second segment
				for i, gene in enumerate(parent1Genes):
					if i < crossoverPoint1:
						child1.addGene(copy.deepcopy(gene))
						
				# Second enfant: aucun gène du premier segment, tous les gènes du second segment
				for i, gene in enumerate(parent1Genes):
					if i >= crossoverPoint1:
						child2.addGene(copy.deepcopy(gene))
				continue
			
			# Les deux parents ont des gènes de ce type
			# Points de croisement
			crossoverPoint1 = self.rng.integers(0, len(parent1Genes) + 1)
			crossoverPoint2 = self.rng.integers(0, len(parent2Genes) + 1)
			
			# Premier enfant: gènes du premier parent jusqu'au point, puis du second parent
			for i, gene in enumerate(parent1Genes):
				if i < crossoverPoint1:
					child1.addGene(copy.deepcopy(gene))
			
			for i, gene in enumerate(parent2Genes):
				if i >= crossoverPoint2:
					child1.addGene(copy.deepcopy(gene))
					
			# Second enfant: gènes du second parent jusqu'au point, puis du premier parent
			for i, gene in enumerate(parent2Genes):
				if i < crossoverPoint2:
					child2.addGene(copy.deepcopy(gene))
			
			for i, gene in enumerate(parent1Genes):
				if i >= crossoverPoint1:
					child2.addGene(copy.deepcopy(gene))
	
	def performGeneCrossover(self, geneA: Gene, geneB: Gene) -> Tuple[Gene, Gene]:
		"""
		Réalise un croisement au niveau des gènes individuels.
		
		Args:
			geneA: Premier gène parental
			geneB: Second gène parental
			
		Returns:
			Deux gènes enfants issus du croisement
		"""
		try:
			return geneA.crossover(geneB, self.rng)
		except ValueError as e:
			# En cas d'incompatibilité, retourner des copies des parents
			return copy.deepcopy(geneA), copy.deepcopy(geneB)
	
	def calculateCompatibility(self, genome1: Genome, genome2: Genome) -> float:
		"""
		Calcule la compatibilité entre deux génomes pour la reproduction.
		
		Args:
			genome1: Premier génome
			genome2: Second génome
			
		Returns:
			Score de compatibilité (0.0 = incompatible, 1.0 = parfaitement compatible)
		"""
		# La compatibilité de base dépend de l'appartenance à la même espèce
		if genome1.speciesId != genome2.speciesId:
			baseCompatibility = 0.3  # Compatibilité réduite pour les croisements inter-espèces
		else:
			baseCompatibility = 1.0
			
		# Calculer la distance génétique
		geneticDistance = genome1.calculateCompatibilityDistance(genome2)
		
		# La compatibilité diminue avec la distance génétique
		compatibilityScore = baseCompatibility * max(0.0, 1.0 - geneticDistance)
		
		return compatibilityScore
	
	def getStats(self) -> Dict[str, int]:
		"""
		Retourne les statistiques sur les croisements effectués.
		
		Returns:
			Dictionnaire des statistiques de croisement
		"""
		return {
			"totalCrossovers": self.totalCrossovers,
			"uniformCrossovers": self.uniformCrossovers,
			"onePointCrossovers": self.onePointCrossovers,
			"interspeciesCrossovers": self.interspeciesCrossovers
		}
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet CrossoverSystem en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du système de croisement
		"""
		return {
			"crossoverRate": self.crossoverRate,
			"uniformCrossoverProbability": self.uniformCrossoverProbability,
			"geneInheritanceRate": self.geneInheritanceRate,
			"inheritDisabledGenes": self.inheritDisabledGenes,
			"hybridizationPenalty": self.hybridizationPenalty,
			"totalCrossovers": self.totalCrossovers,
			"uniformCrossovers": self.uniformCrossovers,
			"onePointCrossovers": self.onePointCrossovers,
			"interspeciesCrossovers": self.interspeciesCrossovers
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'CrossoverSystem':
		"""
		Crée une instance de CrossoverSystem à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du système de croisement
			
		Returns:
			Instance de CrossoverSystem reconstruite
		"""
		system = cls(
			crossoverRate=data["crossoverRate"],
			uniformCrossoverProbability=data["uniformCrossoverProbability"],
			geneInheritanceRate=data["geneInheritanceRate"],
			inheritDisabledGenes=data["inheritDisabledGenes"],
			hybridizationPenalty=data["hybridizationPenalty"]
		)
		
		system.totalCrossovers = data["totalCrossovers"]
		system.uniformCrossovers = data["uniformCrossovers"]
		system.onePointCrossovers = data["onePointCrossovers"]
		system.interspeciesCrossovers = data["interspeciesCrossovers"]
		
		return system