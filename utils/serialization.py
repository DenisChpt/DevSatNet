from typing import Dict, Any, Type, TypeVar, Protocol
import json
import os
import inspect
import numpy as np
from abc import ABC, abstractmethod

# Type variable for generic type hints
T = TypeVar('T')

class Serializable(ABC):
	"""
	Interface abstraite pour les objets qui peuvent être sérialisés et désérialisés.
	Tous les objets qui doivent être sauvegardés et chargés doivent implémenter cette interface.
	"""
	
	@abstractmethod
	def toDict(self) -> Dict[str, Any]:
		"""
		Convertit l'objet en dictionnaire pour la sérialisation.
		
		Returns:
			Dictionnaire représentant l'état de l'objet
		"""
		pass
	
	@classmethod
	@abstractmethod
	def fromDict(cls: Type[T], data: Dict[str, Any]) -> T:
		"""
		Crée une instance de la classe à partir d'un dictionnaire.
		
		Args:
			data: Dictionnaire contenant les données de l'objet
			
		Returns:
			Instance reconstruite de la classe
		"""
		pass


class NumpyEncoder(json.JSONEncoder):
	"""
	Encodeur JSON personnalisé pour gérer les types numpy.
	"""
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		elif isinstance(obj, np.integer):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.bool_):
			return bool(obj)
		return super(NumpyEncoder, self).default(obj)


def save_object(obj: Serializable, filepath: str) -> None:
	"""
	Sauvegarde un objet sérialisable dans un fichier JSON.
	
	Args:
		obj: Objet sérialisable à sauvegarder
		filepath: Chemin du fichier de destination
	"""
	# Créer le répertoire si nécessaire
	os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
	
	# Convertir l'objet en dictionnaire
	data = obj.toDict()
	
	# Ajouter la classe de l'objet pour la désérialisation
	data['__class__'] = obj.__class__.__name__
	data['__module__'] = obj.__class__.__module__
	
	# Écrire le dictionnaire au format JSON dans le fichier
	with open(filepath, 'w') as f:
		json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_object(filepath: str) -> Serializable:
	"""
	Charge un objet sérialisable depuis un fichier JSON.
	
	Args:
		filepath: Chemin du fichier source
		
	Returns:
		Objet désérialisé
	"""
	# Lire le fichier JSON
	with open(filepath, 'r') as f:
		data = json.load(f)
	
	# Récupérer les informations de classe
	class_name = data.pop('__class__')
	module_name = data.pop('__module__')
	
	# Importer le module et obtenir la classe
	module = __import__(module_name, fromlist=[class_name])
	cls = getattr(module, class_name)
	
	# Vérifier que la classe implémente l'interface Serializable
	if not issubclass(cls, Serializable):
		raise TypeError(f"La classe {class_name} n'implémente pas l'interface Serializable")
	
	# Créer et retourner l'objet
	return cls.fromDict(data)


def save_simulation_state(simulation, filepath: str, include_history: bool = True) -> None:
	"""
	Sauvegarde l'état complet d'une simulation.
	
	Args:
		simulation: Objet de simulation à sauvegarder
		filepath: Chemin du fichier de destination
		include_history: Si True, inclut l'historique complet de la simulation
	"""
	# Vérifier que la simulation est sérialisable
	if not isinstance(simulation, Serializable):
		raise TypeError("L'objet de simulation doit implémenter l'interface Serializable")
	
	# Convertir la simulation en dictionnaire
	data = simulation.toDict()
	
	# Si l'historique ne doit pas être inclus, le supprimer
	if not include_history and 'history' in data:
		data['history'] = []
	
	# Ajouter des métadonnées
	data['__version__'] = '1.0.0'
	data['__timestamp__'] = import_datetime_and_get_now()
	data['__class__'] = simulation.__class__.__name__
	data['__module__'] = simulation.__class__.__module__
	
	# Créer le répertoire si nécessaire
	os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
	
	# Écrire le dictionnaire au format JSON dans le fichier
	with open(filepath, 'w') as f:
		json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_simulation_state(filepath: str) -> Serializable:
	"""
	Charge l'état complet d'une simulation.
	
	Args:
		filepath: Chemin du fichier source
		
	Returns:
		Objet de simulation désérialisé
	"""
	# Lire le fichier JSON
	with open(filepath, 'r') as f:
		data = json.load(f)
	
	# Vérifier la version
	version = data.pop('__version__', None)
	if version != '1.0.0':
		print(f"Attention: Chargement d'un fichier de version {version}, la version actuelle est 1.0.0")
	
	# Supprimer les autres métadonnées
	data.pop('__timestamp__', None)
	
	# Récupérer les informations de classe
	class_name = data.pop('__class__')
	module_name = data.pop('__module__')
	
	# Importer le module et obtenir la classe
	module = __import__(module_name, fromlist=[class_name])
	cls = getattr(module, class_name)
	
	# Vérifier que la classe implémente l'interface Serializable
	if not issubclass(cls, Serializable):
		raise TypeError(f"La classe {class_name} n'implémente pas l'interface Serializable")
	
	# Créer et retourner l'objet
	return cls.fromDict(data)


def import_datetime_and_get_now():
	"""
	Importe le module datetime et retourne la date et l'heure actuelles.
	
	Returns:
		Date et heure actuelles au format ISO
	"""
	import datetime
	return datetime.datetime.now().isoformat()