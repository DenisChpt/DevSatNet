# core/genetics/genome.py
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import numpy as np
import uuid
import copy

from utils.serialization import Serializable


class Gene(Serializable):
	"""
	Représente un gène individuel contenant une ou plusieurs caractéristiques héritables.
	"""
	
	def __init__(
		self,
		geneType: str,
		values: Dict[str, Union[float, int, str, bool]],
		dominance: float = 0.5,
		mutationRate: float = 0.05,
		mutationMagnitude: float = 0.1,
		enabled: bool = True
	) -> None:
		"""
		Initialise un nouveau gène.
		
		Args:
			geneType: Type de gène (morphology, behavior, metabolism, etc.)
			values: Dictionnaire des valeurs du gène
			dominance: Niveau de dominance (0-1) affectant l'expression
			mutationRate: Probabilité de mutation lors de la reproduction
			mutationMagnitude: Amplitude des mutations lorsqu'elles se produisent
			enabled: Indique si le gène est activé
		"""
		self.id: str = str(uuid.uuid4())
		self.geneType: str = geneType
		self.values: Dict[str, Union[float, int, str, bool]] = values
		self.dominance: float = dominance
		self.mutationRate: float = mutationRate
		self.mutationMagnitude: float = mutationMagnitude
		self.enabled: bool = enabled
		
		# Historique d'évolution
		self.generation: int = 0
		self.parentIds: List[str] = []
		self.modificationHistory: List[Dict[str, Any]] = []
	
	def mutate(self, rng: np.random.Generator) -> bool:
		"""
		Applique une mutation potentielle au gène.
		
		Args:
			rng: Générateur de nombres aléatoires
			
		Returns:
			True si une mutation s'est produite, False sinon
		"""
		if not self.enabled:
			return False
			
		# Déterminer si une mutation se produit
		if rng.random() > self.mutationRate:
			return False
			
		# Déterminer quelles valeurs muter
		mutatedKeys = []
		
		for key, value in self.values.items():
			# Chaque valeur a une chance indépendante de muter
			if rng.random() <= self.mutationRate:
				oldValue = self.values[key]
				
				# Mutation dépend du type de valeur
				if isinstance(value, (int, float)):
					# Mutation continue (addition d'une valeur aléatoire)
					mutationStrength = rng.normal(0, self.mutationMagnitude)
					
					if isinstance(value, int):
						# Pour les entiers, arrondir et assurer une valeur minimale de 0
						newValue = max(0, int(round(value + mutationStrength * 10)))
					else:
						# Pour les flottants, simplement ajouter
						newValue = value + mutationStrength
						
						# Si la valeur est censée être normalisée (entre 0-1)
						if 0 <= value <= 1:
							newValue = max(0.0, min(1.0, newValue))
							
				elif isinstance(value, bool):
					# Mutation discrète (inversion avec une certaine probabilité)
					if rng.random() < self.mutationMagnitude:
						newValue = not value
					else:
						newValue = value
				elif isinstance(value, str):
					# Pour les chaînes, pas de mutation (ou implémentation spécifique)
					newValue = value
				else:
					# Type inconnu, pas de mutation
					newValue = value
				
				# Mettre à jour la valeur si elle a changé
				if newValue != oldValue:
					self.values[key] = newValue
					mutatedKeys.append(key)
		
		# Enregistrer la mutation dans l'historique si au moins une valeur a muté
		if mutatedKeys:
			self.modificationHistory.append({
				"type": "mutation",
				"timestamp": 0,  # Sera mis à jour par le système d'évolution
				"modifiedKeys": mutatedKeys
			})
			return True
			
		return False
	
	def crossover(self, otherGene: 'Gene', rng: np.random.Generator) -> Tuple['Gene', 'Gene']:
		"""
		Réalise un croisement entre ce gène et un autre gène du même type.
		
		Args:
			otherGene: Autre gène pour le croisement
			rng: Générateur de nombres aléatoires
			
		Returns:
			Deux nouveaux gènes issus du croisement
		"""
		if self.geneType != otherGene.geneType:
			raise ValueError(f"Incompatible gene types: {self.geneType} vs {otherGene.geneType}")
			
		# Créer deux nouveaux gènes (enfants)
		child1 = Gene(
			geneType=self.geneType,
			values={},
			dominance=(self.dominance + otherGene.dominance) / 2,
			mutationRate=(self.mutationRate + otherGene.mutationRate) / 2,
			mutationMagnitude=(self.mutationMagnitude + otherGene.mutationMagnitude) / 2,
			enabled=self.enabled and otherGene.enabled
		)
		
		child2 = Gene(
			geneType=self.geneType,
			values={},
			dominance=(self.dominance + otherGene.dominance) / 2,
			mutationRate=(self.mutationRate + otherGene.mutationRate) / 2,
			mutationMagnitude=(self.mutationMagnitude + otherGene.mutationMagnitude) / 2,
			enabled=self.enabled and otherGene.enabled
		)
		
		# Effectuer le croisement pour chaque valeur
		# Si les deux parents ont la même clé, mélanger les valeurs
		allKeys = set(self.values.keys()) | set(otherGene.values.keys())
		
		for key in allKeys:
			if key in self.values and key in otherGene.values:
				# Les deux parents ont cette valeur, mélanger selon leur dominance
				value1 = self.values[key]
				value2 = otherGene.values[key]
				
				if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
					# Pour les nombres, faire une moyenne pondérée
					dominanceFactor = (self.dominance) / (self.dominance + otherGene.dominance)
					blendedValue = value1 * dominanceFactor + value2 * (1 - dominanceFactor)
					
					# Ajouter une petite variation pour chaque enfant
					if isinstance(value1, int):
						child1.values[key] = max(0, int(round(blendedValue + rng.normal(0, 0.5))))
						child2.values[key] = max(0, int(round(blendedValue + rng.normal(0, 0.5))))
					else:
						child1.values[key] = blendedValue + rng.normal(0, 0.01)
						child2.values[key] = blendedValue + rng.normal(0, 0.01)
						
						# Si la valeur est censée être normalisée (entre 0-1)
						if 0 <= value1 <= 1 and 0 <= value2 <= 1:
							child1.values[key] = max(0.0, min(1.0, child1.values[key]))
							child2.values[key] = max(0.0, min(1.0, child2.values[key]))
							
				elif isinstance(value1, bool) and isinstance(value2, bool):
					# Pour les booléens, choisir en fonction du niveau de dominance
					if rng.random() < self.dominance:
						child1.values[key] = value1
					else:
						child1.values[key] = value2
						
					if rng.random() < otherGene.dominance:
						child2.values[key] = value2
					else:
						child2.values[key] = value1
						
				elif isinstance(value1, str) and isinstance(value2, str):
					# Pour les chaînes, choisir l'une ou l'autre
					if rng.random() < 0.5:
						child1.values[key] = value1
						child2.values[key] = value2
					else:
						child1.values[key] = value2
						child2.values[key] = value1
				else:
					# Types incompatibles, choisir l'un ou l'autre
					if rng.random() < 0.5:
						if key in self.values:
							child1.values[key] = self.values[key]
						if key in otherGene.values:
							child2.values[key] = otherGene.values[key]
					else:
						if key in otherGene.values:
							child1.values[key] = otherGene.values[key]
						if key in self.values:
							child2.values[key] = self.values[key]
			elif key in self.values:
				# Seulement le premier parent a cette valeur
				child1.values[key] = self.values[key]
				# 50% de chance de l'hériter pour le second enfant
				if rng.random() < 0.5:
					child2.values[key] = self.values[key]
			else:
				# Seulement le second parent a cette valeur
				child2.values[key] = otherGene.values[key]
				# 50% de chance de l'hériter pour le premier enfant
				if rng.random() < 0.5:
					child1.values[key] = otherGene.values[key]
		
		# Mettre à jour les informations d'héritage
		child1.generation = max(self.generation, otherGene.generation) + 1
		child2.generation = max(self.generation, otherGene.generation) + 1
		
		child1.parentIds = [self.id, otherGene.id]
		child2.parentIds = [self.id, otherGene.id]
		
		child1.modificationHistory.append({
			"type": "crossover",
			"timestamp": 0,  # Sera mis à jour par le système d'évolution
			"parentIds": [self.id, otherGene.id]
		})
		
		child2.modificationHistory.append({
			"type": "crossover",
			"timestamp": 0,  # Sera mis à jour par le système d'évolution
			"parentIds": [self.id, otherGene.id]
		})
		
		return child1, child2
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Gene en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du gène
		"""
		return {
			"id": self.id,
			"geneType": self.geneType,
			"values": self.values,
			"dominance": self.dominance,
			"mutationRate": self.mutationRate,
			"mutationMagnitude": self.mutationMagnitude,
			"enabled": self.enabled,
			"generation": self.generation,
			"parentIds": self.parentIds,
			"modificationHistory": self.modificationHistory
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Gene':
		"""
		Crée une instance de Gene à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du gène
			
		Returns:
			Instance de Gene reconstruite
		"""
		gene = cls(
			geneType=data["geneType"],
			values=data["values"],
			dominance=data["dominance"],
			mutationRate=data["mutationRate"],
			mutationMagnitude=data["mutationMagnitude"],
			enabled=data["enabled"]
		)
		
		gene.id = data["id"]
		gene.generation = data["generation"]
		gene.parentIds = data["parentIds"]
		gene.modificationHistory = data["modificationHistory"]
		
		return gene


class Genome(Serializable):
	"""
	Classe représentant le génome complet d'une créature, contenant plusieurs gènes.
	"""
	
	def __init__(
		self,
		speciesId: str,
		genes: Optional[List[Gene]] = None,
		fitness: float = 0.0,
		generation: int = 0,
		metadata: Optional[Dict[str, Any]] = None
	) -> None:
		"""
		Initialise un nouveau génome.
		
		Args:
			speciesId: Identifiant de l'espèce à laquelle appartient ce génome
			genes: Liste des gènes constituant le génome
			fitness: Valeur d'adaptation (fitness) de ce génome
			generation: Génération à laquelle appartient ce génome
			metadata: Métadonnées supplémentaires
		"""
		self.id: str = str(uuid.uuid4())
		self.speciesId: str = speciesId
		self.genes: List[Gene] = genes if genes is not None else []
		self.fitness: float = fitness
		self.generation: int = generation
		self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
		
		# Regrouper les gènes par type pour un accès plus facile
		self.genesByType: Dict[str, List[Gene]] = {}
		self._updateGenesByType()
		
		# Historique de reproduction
		self.parentIds: List[str] = []
		self.childrenIds: List[str] = []
		self.mutationCount: int = 0
	
	def _updateGenesByType(self) -> None:
		"""
		Met à jour le dictionnaire de gènes par type.
		"""
		self.genesByType = {}
		for gene in self.genes:
			if gene.geneType not in self.genesByType:
				self.genesByType[gene.geneType] = []
				
			self.genesByType[gene.geneType].append(gene)
	
	def addGene(self, gene: Gene) -> None:
		"""
		Ajoute un gène au génome.
		
		Args:
			gene: Gène à ajouter
		"""
		self.genes.append(gene)
		
		if gene.geneType not in self.genesByType:
			self.genesByType[gene.geneType] = []
			
		self.genesByType[gene.geneType].append(gene)
	
	def removeGene(self, geneId: str) -> bool:
		"""
		Supprime un gène du génome.
		
		Args:
			geneId: Identifiant du gène à supprimer
			
		Returns:
			True si le gène a été trouvé et supprimé, False sinon
		"""
		geneToRemove = None
		
		for gene in self.genes:
			if gene.id == geneId:
				geneToRemove = gene
				break
				
		if geneToRemove is None:
			return False
			
		self.genes.remove(geneToRemove)
		
		# Mettre à jour le dictionnaire par type
		if geneToRemove.geneType in self.genesByType:
			self.genesByType[geneToRemove.geneType] = [
				g for g in self.genesByType[geneToRemove.geneType] if g.id != geneId
			]
			
		return True
	
	def getGenesByType(self, geneType: str) -> List[Gene]:
		"""
		Retourne les gènes d'un type spécifique.
		
		Args:
			geneType: Type de gène à récupérer
			
		Returns:
			Liste des gènes du type spécifié
		"""
		return self.genesByType.get(geneType, [])
	
	def mutate(self, rng: np.random.Generator, mutationProbability: float = 0.3) -> int:
		"""
		Applique des mutations potentielles à tout le génome.
		
		Args:
			rng: Générateur de nombres aléatoires
			mutationProbability: Probabilité globale de mutation
			
		Returns:
			Nombre de gènes qui ont muté
		"""
		mutatedCount = 0
		
		# Pour chaque gène, tenter une mutation
		for gene in self.genes:
			# Deux niveaux de probabilité: global et spécifique au gène
			if rng.random() < mutationProbability:
				if gene.mutate(rng):
					mutatedCount += 1
		
		self.mutationCount += mutatedCount
		return mutatedCount
	
	def crossover(
		self,
		otherGenome: 'Genome',
		rng: np.random.Generator,
		uniformCrossover: bool = False
	) -> Tuple['Genome', 'Genome']:
		"""
		Réalise un croisement entre ce génome et un autre pour produire deux descendants.
		
		Args:
			otherGenome: Autre génome pour le croisement
			rng: Générateur de nombres aléatoires
			uniformCrossover: Si True, utiliser le croisement uniforme, sinon croisement à un point
			
		Returns:
			Deux nouveaux génomes issus du croisement
		"""
		# Vérifier que les génomes sont de la même espèce
		if self.speciesId != otherGenome.speciesId:
			raise ValueError(f"Incompatible species for crossover: {self.speciesId} vs {otherGenome.speciesId}")
			
		# Créer deux nouveaux génomes (enfants)
		child1 = Genome(
			speciesId=self.speciesId,
			genes=[],
			generation=max(self.generation, otherGenome.generation) + 1
		)
		
		child2 = Genome(
			speciesId=self.speciesId,
			genes=[],
			generation=max(self.generation, otherGenome.generation) + 1
		)
		
		# Enregistrer les parents
		child1.parentIds = [self.id, otherGenome.id]
		child2.parentIds = [self.id, otherGenome.id]
		
		# Mettre à jour les enfants des parents
		self.childrenIds.append(child1.id)
		self.childrenIds.append(child2.id)
		otherGenome.childrenIds.append(child1.id)
		otherGenome.childrenIds.append(child2.id)
		
		# Regrouper les gènes par type
		selfGenesByType = self.genesByType
		otherGenesByType = otherGenome.genesByType
		
		# Liste de tous les types de gènes présents dans les deux génomes
		allGeneTypes = set(selfGenesByType.keys()) | set(otherGenesByType.keys())
		
		for geneType in allGeneTypes:
			selfGenes = selfGenesByType.get(geneType, [])
			otherGenes = otherGenesByType.get(geneType, [])
			
			# Si aucun parent n'a ce type de gène, continuer
			if not selfGenes and not otherGenes:
				continue
				
			# Si un seul parent a ce type de gène, 50% de chance pour chaque enfant de l'hériter
			if not selfGenes:
				for gene in otherGenes:
					if rng.random() < 0.5:
						child1.addGene(copy.deepcopy(gene))
					if rng.random() < 0.5:
						child2.addGene(copy.deepcopy(gene))
				continue
				
			if not otherGenes:
				for gene in selfGenes:
					if rng.random() < 0.5:
						child1.addGene(copy.deepcopy(gene))
					if rng.random() < 0.5:
						child2.addGene(copy.deepcopy(gene))
				continue
			
			# Les deux parents ont ce type de gène, faire un croisement
			if uniformCrossover:
				# Croisement uniforme: chaque gène est hérité aléatoirement de l'un des parents
				allParentGenes = selfGenes + otherGenes
				
				for gene in allParentGenes:
					if rng.random() < 0.5:
						child1.addGene(copy.deepcopy(gene))
					else:
						child2.addGene(copy.deepcopy(gene))
			else:
				# Croisement à un point: le premier enfant hérite des gènes du premier parent jusqu'à un point,
				# puis des gènes du second parent; le second enfant fait l'inverse
				
				# Mélanger les gènes pour éviter les biais
				shuffledSelfGenes = selfGenes.copy()
				shuffledOtherGenes = otherGenes.copy()
				rng.shuffle(shuffledSelfGenes)
				rng.shuffle(shuffledOtherGenes)
				
				# Déterminer les points de croisement
				crossoverPoint1 = rng.integers(0, len(shuffledSelfGenes) + 1)
				crossoverPoint2 = rng.integers(0, len(shuffledOtherGenes) + 1)
				
				# Premier enfant: gènes du premier parent jusqu'au point, puis du second
				for i, gene in enumerate(shuffledSelfGenes):
					if i < crossoverPoint1:
						child1.addGene(copy.deepcopy(gene))
					else:
						child2.addGene(copy.deepcopy(gene))
						
				for i, gene in enumerate(shuffledOtherGenes):
					if i < crossoverPoint2:
						child2.addGene(copy.deepcopy(gene))
					else:
						child1.addGene(copy.deepcopy(gene))
		
		# Faire aussi des croisements au niveau des gènes individuels pour les gènes communs
		for geneType in allGeneTypes:
			selfGenes = selfGenesByType.get(geneType, [])
			otherGenes = otherGenesByType.get(geneType, [])
			
			# Si les deux parents ont des gènes du même type, faire des croisements au niveau des gènes
			if selfGenes and otherGenes:
				# Sélectionner aléatoirement quelques paires de gènes pour le croisement
				numCrossovers = min(len(selfGenes), len(otherGenes), 3)  # Maximum 3 croisements par type
				
				for _ in range(numCrossovers):
					if rng.random() < 0.3:  # 30% de chance de faire un croisement au niveau des gènes
						geneA = rng.choice(selfGenes)
						geneB = rng.choice(otherGenes)
						
						# Faire le croisement si les gènes sont compatibles
						try:
							newGene1, newGene2 = geneA.crossover(geneB, rng)
							child1.addGene(newGene1)
							child2.addGene(newGene2)
						except ValueError:
							# Gènes incompatibles, ignorer le croisement
							pass
		
		return child1, child2
	
	def calculateCompatibilityDistance(self, otherGenome: 'Genome') -> float:
		"""
		Calcule une mesure de distance génétique entre ce génome et un autre.
		
		Args:
			otherGenome: Autre génome à comparer
			
		Returns:
			Mesure de distance (plus faible = plus compatible)
		"""
		if self.speciesId != otherGenome.speciesId:
			return float('inf')  # Incompatible
			
		# Regrouper les gènes par type
		selfGenesByType = self.genesByType
		otherGenesByType = otherGenome.genesByType
		
		# Liste de tous les types de gènes présents dans les deux génomes
		allGeneTypes = set(selfGenesByType.keys()) | set(otherGenesByType.keys())
		
		if not allGeneTypes:
			return 0.0  # Les deux génomes sont vides
			
		totalDistance = 0.0
		
		for geneType in allGeneTypes:
			selfGenes = selfGenesByType.get(geneType, [])
			otherGenes = otherGenesByType.get(geneType, [])
			
			# Différence de nombre de gènes
			numDifference = abs(len(selfGenes) - len(otherGenes))
			totalDistance += numDifference * 0.5
			
			# Différence de valeurs pour les gènes communs
			# On calcule la distance entre les valeurs moyennes pour chaque type
			if selfGenes and otherGenes:
				# Calculer les valeurs moyennes pour les gènes numériques
				selfValues = {}
				otherValues = {}
				
				for gene in selfGenes:
					for key, value in gene.values.items():
						if isinstance(value, (int, float)):
							if key not in selfValues:
								selfValues[key] = []
							selfValues[key].append(value)
				
				for gene in otherGenes:
					for key, value in gene.values.items():
						if isinstance(value, (int, float)):
							if key not in otherValues:
								otherValues[key] = []
							otherValues[key].append(value)
				
				# Calculer la différence pour chaque clé
				commonKeys = set(selfValues.keys()) & set(otherValues.keys())
				for key in commonKeys:
					selfMean = sum(selfValues[key]) / len(selfValues[key])
					otherMean = sum(otherValues[key]) / len(otherValues[key])
					
					# Normaliser la différence
					maxValue = max(max(selfValues[key]), max(otherValues[key]))
					minValue = min(min(selfValues[key]), min(otherValues[key]))
					
					if maxValue > minValue:
						normalizedDiff = abs(selfMean - otherMean) / (maxValue - minValue)
					else:
						normalizedDiff = 0.0
						
					totalDistance += normalizedDiff
		
		# Normaliser la distance totale
		return totalDistance / len(allGeneTypes)
	
	def expressPhysicalTraits(self) -> Dict[str, Any]:
		"""
		Exprime les caractéristiques physiques de la créature à partir du génome.
		Cela transforme l'information génétique en caractéristiques concrètes.
		
		Returns:
			Dictionnaire des caractéristiques physiques exprimées
		"""
		# Les caractéristiques physiques sont déterminées par les gènes de morphologie
		morphologyGenes = self.getGenesByType("morphology")
		
		# Valeurs par défaut
		traits = {
			"size": 1.0,
			"bodyPlan": "fish",
			"limbCount": 4,
			"jointCount": 5,
			"muscleCount": 6,
			"sensorCount": 4,
			"symmetry": 1.0,  # 1.0 = symétrie bilatérale complète
			"density": 1.0,   # Densité relative à l'eau
			"color": [0.5, 0.5, 0.8],  # RGB
			"texture": "smooth",
			"skeletonType": "cartilage"
		}
		
		# Combinaison des valeurs de tous les gènes de morphologie
		# Les gènes avec une dominance plus élevée ont plus d'influence
		numericalTraits = ["size", "limbCount", "jointCount", "muscleCount", "sensorCount", "symmetry", "density"]
		colorTraits = ["color"]
		
		weightedSum = {trait: 0.0 for trait in numericalTraits}
		weightedSumColors = {i: 0.0 for i in range(3)}
		totalWeight = 0.0
		
		for gene in morphologyGenes:
			weight = gene.dominance
			totalWeight += weight
			
			for trait in numericalTraits:
				if trait in gene.values:
					if trait in ["limbCount", "jointCount", "muscleCount", "sensorCount"]:
						# Convertir en entier pour les compteurs
						value = int(gene.values[trait])
					else:
						value = gene.values[trait]
					weightedSum[trait] += value * weight
			
			# Traitement spécial pour la couleur (RGB)
			if "color" in gene.values and isinstance(gene.values["color"], list) and len(gene.values["color"]) == 3:
				for i in range(3):
					weightedSumColors[i] += gene.values["color"][i] * weight
			
			# Pour les traits catégoriels, utiliser le gène le plus dominant
			if "bodyPlan" in gene.values and gene.dominance > traits.get("bodyPlan_dominance", 0):
				traits["bodyPlan"] = gene.values["bodyPlan"]
				traits["bodyPlan_dominance"] = gene.dominance
				
			if "texture" in gene.values and gene.dominance > traits.get("texture_dominance", 0):
				traits["texture"] = gene.values["texture"]
				traits["texture_dominance"] = gene.dominance
				
			if "skeletonType" in gene.values and gene.dominance > traits.get("skeletonType_dominance", 0):
				traits["skeletonType"] = gene.values["skeletonType"]
				traits["skeletonType_dominance"] = gene.dominance
		
		# Calculer les valeurs finales si des gènes ont été trouvés
		if totalWeight > 0:
			for trait in numericalTraits:
				traits[trait] = weightedSum[trait] / totalWeight
				
			# Arrondir les compteurs à des entiers
			for trait in ["limbCount", "jointCount", "muscleCount", "sensorCount"]:
				traits[trait] = max(1, int(round(traits[trait])))
				
			# Calculer la couleur finale
			traits["color"] = [weightedSumColors[i] / totalWeight for i in range(3)]
		
		# Nettoyer les traits temporaires
		for temp_trait in ["bodyPlan_dominance", "texture_dominance", "skeletonType_dominance"]:
			if temp_trait in traits:
				del traits[temp_trait]
		
		return traits
	
	def expressBehavioralTraits(self) -> Dict[str, Any]:
		"""
		Exprime les caractéristiques comportementales de la créature à partir du génome.
		
		Returns:
			Dictionnaire des caractéristiques comportementales exprimées
		"""
		# Les caractéristiques comportementales sont déterminées par les gènes de comportement
		behaviorGenes = self.getGenesByType("behavior")
		
		# Valeurs par défaut
		traits = {
			"aggressiveness": 0.5,
			"sociability": 0.5,
			"curiosity": 0.5,
			"territoriality": 0.5,
			"activity": 0.5,
			"fearfulness": 0.5,
			"explorativeness": 0.5,
			"dietType": "omnivore",
			"diurnalActivity": 0.5,  # 0 = nocturne, 1 = diurne
			"learningCapacity": 0.5
		}
		
		# Combinaison des valeurs de tous les gènes de comportement
		numericalTraits = ["aggressiveness", "sociability", "curiosity", "territoriality", 
						  "activity", "fearfulness", "explorativeness", "diurnalActivity", "learningCapacity"]
		
		weightedSum = {trait: 0.0 for trait in numericalTraits}
		totalWeight = 0.0
		
		for gene in behaviorGenes:
			weight = gene.dominance
			totalWeight += weight
			
			for trait in numericalTraits:
				if trait in gene.values:
					weightedSum[trait] += gene.values[trait] * weight
			
			# Pour les traits catégoriels, utiliser le gène le plus dominant
			if "dietType" in gene.values and gene.dominance > traits.get("dietType_dominance", 0):
				traits["dietType"] = gene.values["dietType"]
				traits["dietType_dominance"] = gene.dominance
		
		# Calculer les valeurs finales si des gènes ont été trouvés
		if totalWeight > 0:
			for trait in numericalTraits:
				traits[trait] = weightedSum[trait] / totalWeight
		
		# Nettoyer les traits temporaires
		if "dietType_dominance" in traits:
			del traits["dietType_dominance"]
		
		return traits
	
	def expressMetabolicTraits(self) -> Dict[str, Any]:
		"""
		Exprime les caractéristiques métaboliques de la créature à partir du génome.
		
		Returns:
			Dictionnaire des caractéristiques métaboliques exprimées
		"""
		# Les caractéristiques métaboliques sont déterminées par les gènes de métabolisme
		metabolismGenes = self.getGenesByType("metabolism")
		
		# Valeurs par défaut
		traits = {
			"basalMetabolicRate": 1.0,
			"energyEfficiency": 1.0,
			"temperatureTolerance": (10.0, 30.0),  # Plage de température (°C)
			"pressureTolerance": (1.0, 10.0),      # Plage de pression (bar)
			"oxygenRequirement": 1.0,
			"toxinResistance": 0.5,
			"radiationResistance": 0.5,
			"regenerationRate": 0.1,
			"longevity": 1.0,                      # Facteur d'espérance de vie
			"maturationRate": 1.0                  # Vitesse de maturation
		}
		
		# Combinaison des valeurs de tous les gènes de métabolisme
		numericalTraits = ["basalMetabolicRate", "energyEfficiency", "oxygenRequirement", 
						  "toxinResistance", "radiationResistance", "regenerationRate", 
						  "longevity", "maturationRate"]
		
		rangeTraits = ["temperatureTolerance", "pressureTolerance"]
		
		weightedSum = {trait: 0.0 for trait in numericalTraits}
		
		# Pour les traits de plage (min, max)
		weightedSumRangeMin = {trait: 0.0 for trait in rangeTraits}
		weightedSumRangeMax = {trait: 0.0 for trait in rangeTraits}
		
		totalWeight = 0.0
		
		for gene in metabolismGenes:
			weight = gene.dominance
			totalWeight += weight
			
			for trait in numericalTraits:
				if trait in gene.values:
					weightedSum[trait] += gene.values[trait] * weight
			
			# Traitement spécial pour les traits de plage (min, max)
			for trait in rangeTraits:
				if trait in gene.values and isinstance(gene.values[trait], tuple) and len(gene.values[trait]) == 2:
					minVal, maxVal = gene.values[trait]
					weightedSumRangeMin[trait] += minVal * weight
					weightedSumRangeMax[trait] += maxVal * weight
		
		# Calculer les valeurs finales si des gènes ont été trouvés
		if totalWeight > 0:
			for trait in numericalTraits:
				traits[trait] = weightedSum[trait] / totalWeight
				
			# Calculer les valeurs de plage finales
			for trait in rangeTraits:
				minVal = weightedSumRangeMin[trait] / totalWeight
				maxVal = weightedSumRangeMax[trait] / totalWeight
				
				# S'assurer que min <= max
				if minVal > maxVal:
					minVal, maxVal = maxVal, minVal
					
				traits[trait] = (minVal, maxVal)
		
		return traits
	
	def expressEnvironmentalTraits(self) -> Dict[str, Any]:
		"""
		Exprime les préférences et adaptations environnementales de la créature.
		
		Returns:
			Dictionnaire des traits environnementaux exprimés
		"""
		# Les traits environnementaux sont déterminés par les gènes d'environnement
		environmentalGenes = self.getGenesByType("environmental")
		
		# Valeurs par défaut
		traits = {
			"habitatPreference": "pelagic",  # pelagic, benthic, reef, deep_sea, coastal
			"depthRange": (0.0, 100.0),      # Plage de profondeur préférée (m)
			"salinityTolerance": (30.0, 40.0),  # Plage de salinité (PSU)
			"currentStrengthPreference": 0.5,   # 0 = eau calme, 1 = forts courants
			"substratumPreference": "sand",     # sand, mud, rock, coral, seagrass
			"lightSensitivity": 0.5,            # 0 = préfère l'obscurité, 1 = préfère la lumière
			"migratory": 0.0,                   # Tendance migratoire
			"seasonalAdaptation": 0.5           # Adaptation aux changements saisonniers
		}
		
		# Combinaison des valeurs de tous les gènes environnementaux
		numericalTraits = ["currentStrengthPreference", "lightSensitivity", "migratory", "seasonalAdaptation"]
		
		rangeTraits = ["depthRange", "salinityTolerance"]
		
		weightedSum = {trait: 0.0 for trait in numericalTraits}
		
		# Pour les traits de plage (min, max)
		weightedSumRangeMin = {trait: 0.0 for trait in rangeTraits}
		weightedSumRangeMax = {trait: 0.0 for trait in rangeTraits}
		
		totalWeight = 0.0
		
		for gene in environmentalGenes:
			weight = gene.dominance
			totalWeight += weight
			
			for trait in numericalTraits:
				if trait in gene.values:
					weightedSum[trait] += gene.values[trait] * weight
			
			# Traitement spécial pour les traits de plage (min, max)
			for trait in rangeTraits:
				if trait in gene.values and isinstance(gene.values[trait], tuple) and len(gene.values[trait]) == 2:
					minVal, maxVal = gene.values[trait]
					weightedSumRangeMin[trait] += minVal * weight
					weightedSumRangeMax[trait] += maxVal * weight
			
			# Pour les traits catégoriels, utiliser le gène le plus dominant
			for categoricalTrait in ["habitatPreference", "substratumPreference"]:
				if categoricalTrait in gene.values and gene.dominance > traits.get(f"{categoricalTrait}_dominance", 0):
					traits[categoricalTrait] = gene.values[categoricalTrait]
					traits[f"{categoricalTrait}_dominance"] = gene.dominance
		
		# Calculer les valeurs finales si des gènes ont été trouvés
		if totalWeight > 0:
			for trait in numericalTraits:
				traits[trait] = weightedSum[trait] / totalWeight
				
			# Calculer les valeurs de plage finales
			for trait in rangeTraits:
				minVal = weightedSumRangeMin[trait] / totalWeight
				maxVal = weightedSumRangeMax[trait] / totalWeight
				
				# S'assurer que min <= max
				if minVal > maxVal:
					minVal, maxVal = maxVal, minVal
					
				traits[trait] = (minVal, maxVal)
		
		# Nettoyer les traits temporaires
		for temp_trait in ["habitatPreference_dominance", "substratumPreference_dominance"]:
			if temp_trait in traits:
				del traits[temp_trait]
		
		return traits
	
	def expressAllTraits(self) -> Dict[str, Dict[str, Any]]:
		"""
		Exprime tous les traits de la créature à partir du génome.
		
		Returns:
			Dictionnaire de tous les traits exprimés, organisés par catégorie
		"""
		return {
			"physical": self.expressPhysicalTraits(),
			"behavioral": self.expressBehavioralTraits(),
			"metabolic": self.expressMetabolicTraits(),
			"environmental": self.expressEnvironmentalTraits()
		}
	
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet Genome en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état du génome
		"""
		return {
			"id": self.id,
			"speciesId": self.speciesId,
			"genes": [gene.toDict() for gene in self.genes],
			"fitness": self.fitness,
			"generation": self.generation,
			"metadata": self.metadata,
			"parentIds": self.parentIds,
			"childrenIds": self.childrenIds,
			"mutationCount": self.mutationCount
		}
	
	@classmethod
	def fromDict(cls, data: Dict[str, Any]) -> 'Genome':
		"""
		Crée une instance de Genome à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données du génome
			
		Returns:
			Instance de Genome reconstruite
		"""
		# Créer le génome de base
		genome = cls(
			speciesId=data["speciesId"],
			genes=[],
			fitness=data["fitness"],
			generation=data["generation"],
			metadata=data["metadata"]
		)
		
		# Reconstruire l'ID
		genome.id = data["id"]
		
		# Reconstruire les gènes
		for geneData in data["genes"]:
			gene = Gene.fromDict(geneData)
			genome.addGene(gene)
		
		# Reconstruire les relations
		genome.parentIds = data["parentIds"]
		genome.childrenIds = data["childrenIds"]
		genome.mutationCount = data["mutationCount"]
		
		return genome


def createInitialGenome(
	speciesId: str,
	morphologyCount: int = 3,
	behaviorCount: int = 2,
	metabolismCount: int = 2,
	environmentalCount: int = 2,
	rng: Optional[np.random.Generator] = None
) -> Genome:
	"""
	Crée un génome initial avec un ensemble aléatoire de gènes.
	
	Args:
		speciesId: Identifiant de l'espèce
		morphologyCount: Nombre de gènes de morphologie
		behaviorCount: Nombre de gènes de comportement
		metabolismCount: Nombre de gènes de métabolisme
		environmentalCount: Nombre de gènes environnementaux
		rng: Générateur de nombres aléatoires
		
	Returns:
		Un nouveau génome avec des gènes aléatoires
	"""
	if rng is None:
		rng = np.random.default_rng()
		
	genome = Genome(
		speciesId=speciesId,
		genes=[],
		generation=0
	)
	
	# Créer des gènes de morphologie
	for _ in range(morphologyCount):
		gene = _createRandomMorphologyGene(rng)
		genome.addGene(gene)
		
	# Créer des gènes de comportement
	for _ in range(behaviorCount):
		gene = _createRandomBehaviorGene(rng)
		genome.addGene(gene)
		
	# Créer des gènes de métabolisme
	for _ in range(metabolismCount):
		gene = _createRandomMetabolismGene(rng)
		genome.addGene(gene)
		
	# Créer des gènes environnementaux
	for _ in range(environmentalCount):
		gene = _createRandomEnvironmentalGene(rng)
		genome.addGene(gene)
		
	return genome


def _createRandomMorphologyGene(rng: np.random.Generator) -> Gene:
	"""
	Crée un gène de morphologie aléatoire.
	
	Args:
		rng: Générateur de nombres aléatoires
		
	Returns:
		Un nouveau gène de morphologie
	"""
	values = {
		"size": rng.uniform(0.5, 2.0),
		"bodyPlan": rng.choice(["fish", "jellyfish", "cephalopod", "crustacean", "eel"]),
		"limbCount": rng.integers(0, 8),
		"jointCount": rng.integers(1, 10),
		"muscleCount": rng.integers(2, 12),
		"sensorCount": rng.integers(1, 6),
		"symmetry": rng.uniform(0.5, 1.0),
		"density": rng.uniform(0.8, 1.2),
		"color": [rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1)],
		"texture": rng.choice(["smooth", "rough", "scaly", "slimy"]),
		"skeletonType": rng.choice(["cartilage", "bone", "exoskeleton", "hydrostatic"])
	}
	
	return Gene(
		geneType="morphology",
		values=values,
		dominance=rng.uniform(0.1, 1.0),
		mutationRate=rng.uniform(0.01, 0.1),
		mutationMagnitude=rng.uniform(0.05, 0.2)
	)


def _createRandomBehaviorGene(rng: np.random.Generator) -> Gene:
	"""
	Crée un gène de comportement aléatoire.
	
	Args:
		rng: Générateur de nombres aléatoires
		
	Returns:
		Un nouveau gène de comportement
	"""
	values = {
		"aggressiveness": rng.uniform(0, 1),
		"sociability": rng.uniform(0, 1),
		"curiosity": rng.uniform(0, 1),
		"territoriality": rng.uniform(0, 1),
		"activity": rng.uniform(0, 1),
		"fearfulness": rng.uniform(0, 1),
		"explorativeness": rng.uniform(0, 1),
		"dietType": rng.choice(["carnivore", "herbivore", "omnivore", "filter_feeder"]),
		"diurnalActivity": rng.uniform(0, 1),
		"learningCapacity": rng.uniform(0, 1)
	}
	
	return Gene(
		geneType="behavior",
		values=values,
		dominance=rng.uniform(0.1, 1.0),
		mutationRate=rng.uniform(0.01, 0.1),
		mutationMagnitude=rng.uniform(0.05, 0.2)
	)


def _createRandomMetabolismGene(rng: np.random.Generator) -> Gene:
	"""
	Crée un gène de métabolisme aléatoire.
	
	Args:
		rng: Générateur de nombres aléatoires
		
	Returns:
		Un nouveau gène de métabolisme
	"""
	# Générer des plages de température et de pression
	tempMin = rng.uniform(0, 20)
	tempMax = tempMin + rng.uniform(10, 20)
	
	pressureMin = rng.uniform(1, 5)
	pressureMax = pressureMin + rng.uniform(5, 20)
	
	values = {
		"basalMetabolicRate": rng.uniform(0.5, 1.5),
		"energyEfficiency": rng.uniform(0.5, 1.5),
		"temperatureTolerance": (tempMin, tempMax),
		"pressureTolerance": (pressureMin, pressureMax),
		"oxygenRequirement": rng.uniform(0.5, 1.5),
		"toxinResistance": rng.uniform(0, 1),
		"radiationResistance": rng.uniform(0, 1),
		"regenerationRate": rng.uniform(0, 0.5),
		"longevity": rng.uniform(0.5, 2.0),
		"maturationRate": rng.uniform(0.5, 1.5)
	}
	
	return Gene(
		geneType="metabolism",
		values=values,
		dominance=rng.uniform(0.1, 1.0),
		mutationRate=rng.uniform(0.01, 0.1),
		mutationMagnitude=rng.uniform(0.05, 0.2)
	)


def _createRandomEnvironmentalGene(rng: np.random.Generator) -> Gene:
	"""
	Crée un gène environnemental aléatoire.
	
	Args:
		rng: Générateur de nombres aléatoires
		
	Returns:
		Un nouveau gène environnemental
	"""
	# Générer des plages de profondeur et de salinité
	depthMin = rng.uniform(0, 200)
	depthMax = depthMin + rng.uniform(50, 300)
	
	salinityMin = rng.uniform(25, 35)
	salinityMax = salinityMin + rng.uniform(5, 10)
	
	values = {
		"habitatPreference": rng.choice(["pelagic", "benthic", "reef", "deep_sea", "coastal"]),
		"depthRange": (depthMin, depthMax),
		"salinityTolerance": (salinityMin, salinityMax),
		"currentStrengthPreference": rng.uniform(0, 1),
		"substratumPreference": rng.choice(["sand", "mud", "rock", "coral", "seagrass"]),
		"lightSensitivity": rng.uniform(0, 1),
		"migratory": rng.uniform(0, 1),
		"seasonalAdaptation": rng.uniform(0, 1)
	}
	
	return Gene(
		geneType="environmental",
		values=values,
		dominance=rng.uniform(0.1, 1.0),
		mutationRate=rng.uniform(0.01, 0.1),
		mutationMagnitude=rng.uniform(0.05, 0.2)
	)