import os
import glob
import json
import torch
import shutil
from typing import Dict, List, Any, Optional
import re

class CheckpointManager:
	"""
	Gère les points de contrôle (checkpoints) pour la sauvegarde et le chargement des modèles.
	"""
	
	def __init__(self, checkpointDir: str, maxToKeep: int = 5):
		"""
		Initialise le gestionnaire de points de contrôle.
		
		Args:
			checkpointDir: Répertoire pour les points de contrôle
			maxToKeep: Nombre maximum de points de contrôle à conserver
		"""
		self.checkpointDir = checkpointDir
		self.maxToKeep = maxToKeep
		
		# Créer le répertoire s'il n'existe pas
		os.makedirs(checkpointDir, exist_ok=True)
		
		# Informations sur les checkpoints existants
		self.checkpoints = []
		self._scanCheckpoints()
	
	def _scanCheckpoints(self) -> None:
		"""
		Analyse les points de contrôle existants dans le répertoire.
		"""
		# Motif de recherche pour les fichiers de points de contrôle
		pattern = os.path.join(self.checkpointDir, "checkpoint_*.pt")
		
		# Chercher tous les fichiers correspondants
		checkpointFiles = glob.glob(pattern)
		
		# Extraire les étapes et les trier
		self.checkpoints = []
		
		for filePath in checkpointFiles:
			# Extraire le numéro d'étape du nom de fichier
			try:
				fileName = os.path.basename(filePath)
				match = re.search(r'checkpoint_(\d+)\.pt', fileName)
				if match:
					step = int(match.group(1))
					self.checkpoints.append((step, filePath))
			except Exception:
				continue
		
		# Trier par étape croissante
		self.checkpoints.sort(key=lambda x: x[0])
	
	def save(self, agent: Any, step: int, extraInfo: Optional[Dict[str, Any]] = None, isBest: bool = False) -> str:
		"""
		Sauvegarde un point de contrôle.
		
		Args:
			agent: Agent à sauvegarder
			step: Étape de temps actuelle
			extraInfo: Informations supplémentaires à sauvegarder
			isBest: Si True, sauvegarde également ce checkpoint comme le meilleur
			
		Returns:
			Chemin du fichier de point de contrôle
		"""
		# Générer le nom du fichier
		fileName = f"checkpoint_{step}.pt"
		filePath = os.path.join(self.checkpointDir, fileName)
		
		# Sauvegarder l'agent
		agent.save(filePath)
		
		# Sauvegarder les informations supplémentaires
		if extraInfo is not None:
			infoPath = os.path.join(self.checkpointDir, f"info_{step}.json")
			with open(infoPath, 'w') as f:
				json.dump(extraInfo, f, indent=4)
		
		# Si c'est le meilleur modèle, le copier
		if isBest:
			bestPath = os.path.join(self.checkpointDir, "best_model.pt")
			shutil.copy(filePath, bestPath)
			
			# Sauvegarder les informations supplémentaires pour le meilleur modèle
			if extraInfo is not None:
				bestInfoPath = os.path.join(self.checkpointDir, "best_info.json")
				with open(bestInfoPath, 'w') as f:
					json.dump(extraInfo, f, indent=4)
		
		# Ajouter ce point de contrôle à la liste
		self.checkpoints.append((step, filePath))
		
		# Nettoyer les points de contrôle en excès
		self._cleanupCheckpoints()
		
		return filePath
	
	def _cleanupCheckpoints(self) -> None:
		"""
		Supprime les points de contrôle en excès, en gardant uniquement les plus récents.
		"""
		if len(self.checkpoints) <= self.maxToKeep:
			return
		
		# Trier par étape en ordre décroissant
		sortedCheckpoints = sorted(self.checkpoints, key=lambda x: x[0], reverse=True)
		
		# Garder les maxToKeep plus récents
		checkpointsToKeep = sortedCheckpoints[:self.maxToKeep]
		checkpointsToRemove = sortedCheckpoints[self.maxToKeep:]
		
		# Supprimer les points de contrôle en excès
		for step, filePath in checkpointsToRemove:
			try:
				os.remove(filePath)
				
				# Supprimer également le fichier d'informations associé
				infoPath = os.path.join(self.checkpointDir, f"info_{step}.json")
				if os.path.exists(infoPath):
					os.remove(infoPath)
			except Exception as e:
				print(f"Erreur lors de la suppression du point de contrôle {filePath}: {e}")
		
		# Mettre à jour la liste des points de contrôle
		self.checkpoints = checkpointsToKeep
	
	def getLatestCheckpoint(self) -> Optional[str]:
		"""
		Récupère le chemin du point de contrôle le plus récent.
		
		Returns:
			Chemin du fichier ou None si aucun point de contrôle n'existe
		"""
		if not self.checkpoints:
			return None
		
		# Trier par étape en ordre décroissant et prendre le premier
		sortedCheckpoints = sorted(self.checkpoints, key=lambda x: x[0], reverse=True)
		return sortedCheckpoints[0][1]
	
	def getBestCheckpoint(self) -> Optional[str]:
		"""
		Récupère le chemin du meilleur point de contrôle.
		
		Returns:
			Chemin du fichier ou None si aucun point de contrôle "best" n'existe
		"""
		bestPath = os.path.join(self.checkpointDir, "best_model.pt")
		
		if os.path.exists(bestPath):
			return bestPath
		else:
			return None
	
	def getCheckpointAtStep(self, step: int) -> Optional[str]:
		"""
		Récupère le chemin du point de contrôle à une étape spécifique.
		
		Args:
			step: Étape de temps
			
		Returns:
			Chemin du fichier ou None si aucun point de contrôle n'existe à cette étape
		"""
		# Chercher un point de contrôle avec l'étape correspondante
		for checkpointStep, filePath in self.checkpoints:
			if checkpointStep == step:
				return filePath
		
		return None
	
	def loadAgent(self, agent: Any, checkpoint: Optional[str] = None) -> Optional[Dict[str, Any]]:
		"""
		Charge un agent à partir d'un point de contrôle.
		
		Args:
			agent: Agent à charger
			checkpoint: Chemin du point de contrôle (ou mot-clé 'latest', 'best', ou un nombre d'étape)
			
		Returns:
			Informations supplémentaires du point de contrôle ou None si la charge a échoué
		"""
		checkpointPath = None
		
		# Déterminer le chemin du point de contrôle
		if checkpoint is None or checkpoint == 'latest':
			checkpointPath = self.getLatestCheckpoint()
		elif checkpoint == 'best':
			checkpointPath = self.getBestCheckpoint()
		elif isinstance(checkpoint, int):
			checkpointPath = self.getCheckpointAtStep(checkpoint)
		else:
			checkpointPath = checkpoint
		
		if not checkpointPath or not os.path.exists(checkpointPath):
			print(f"Point de contrôle {checkpoint} non trouvé.")
			return None
		
		# Charger l'agent
		try:
			agent.load(checkpointPath)
			print(f"Agent chargé à partir de {checkpointPath}")
			
			# Charger les informations supplémentaires
			infoPath = None
			
			if checkpoint == 'best':
				infoPath = os.path.join(self.checkpointDir, "best_info.json")
			else:
				# Extraire le numéro d'étape du nom de fichier
				match = re.search(r'checkpoint_(\d+)\.pt', os.path.basename(checkpointPath))
				if match:
					step = match.group(1)
					infoPath = os.path.join(self.checkpointDir, f"info_{step}.json")
			
			if infoPath and os.path.exists(infoPath):
				with open(infoPath, 'r') as f:
					return json.load(f)
			
			return {}
		except Exception as e:
			print(f"Erreur lors du chargement de l'agent: {e}")
			return None