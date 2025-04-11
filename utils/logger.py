import logging
import os
import sys
import time
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import json
import threading

# Configuration des niveaux de journalisation
class LogLevel(Enum):
	DEBUG = logging.DEBUG
	INFO = logging.INFO
	WARNING = logging.WARNING
	ERROR = logging.ERROR
	CRITICAL = logging.CRITICAL


class SimulationLogger:
	"""
	Système de journalisation pour la simulation d'évolution marine.
	Permet de journaliser des messages avec différents niveaux de gravité,
	ainsi que des métriques et des événements spécifiques à la simulation.
	"""
	
	_instance = None
	_lock = threading.Lock()
	
	def __new__(cls, *args, **kwargs):
		with cls._lock:
			if cls._instance is None:
				cls._instance = super(SimulationLogger, cls).__new__(cls)
				cls._instance._initialized = False
			return cls._instance
	
	def __init__(self, 
				log_dir: str = "logs", 
				log_level: LogLevel = LogLevel.INFO,
				console_output: bool = True,
				file_output: bool = True,
				log_prefix: str = "marine_evolution",
				metrics_enabled: bool = True,
				metrics_interval: int = 60) -> None:
		"""
		Initialise le système de journalisation.
		
		Args:
			log_dir: Répertoire où stocker les fichiers de log
			log_level: Niveau de journalisation minimum
			console_output: Activer la sortie vers la console
			file_output: Activer la sortie vers un fichier
			log_prefix: Préfixe pour les noms de fichiers de log
			metrics_enabled: Activer la collecte de métriques
			metrics_interval: Intervalle en secondes pour l'enregistrement périodique des métriques
		"""
		# Éviter la réinitialisation si le singleton est déjà initialisé
		if self._initialized:
			return
			
		self.log_dir = log_dir
		self.log_level = log_level
		self.console_output = console_output
		self.file_output = file_output
		self.log_prefix = log_prefix
		self.metrics_enabled = metrics_enabled
		self.metrics_interval = metrics_interval
		
		# Créer le répertoire de logs s'il n'existe pas
		os.makedirs(log_dir, exist_ok=True)
		
		# Générer un nom de fichier de log basé sur la date et l'heure
		timestamp = time.strftime("%Y%m%d_%H%M%S")
		self.log_filename = f"{log_prefix}_{timestamp}.log"
		self.metrics_filename = f"{log_prefix}_metrics_{timestamp}.json"
		
		# Configurer le logger principal
		self.logger = logging.getLogger("marine_evolution")
		self.logger.setLevel(log_level.value)
		self.logger.handlers = []  # Supprimer les handlers existants
		
		# Format de log détaillé
		formatter = logging.Formatter(
			fmt='%(asctime)s [%(levelname)s] [%(name)s] - %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S'
		)
		
		# Ajouter un handler pour la console si demandé
		if console_output:
			console_handler = logging.StreamHandler(sys.stdout)
			console_handler.setFormatter(formatter)
			self.logger.addHandler(console_handler)
		
		# Ajouter un handler pour le fichier si demandé
		if file_output:
			file_path = os.path.join(log_dir, self.log_filename)
			file_handler = logging.FileHandler(file_path)
			file_handler.setFormatter(formatter)
			self.logger.addHandler(file_handler)
		
		# Initialiser le stockage de métriques
		self.metrics: Dict[str, List[Dict[str, Any]]] = {}
		self.last_metrics_time = time.time()
		
		# Initialiser les loggers spécialisés
		self._init_specialized_loggers()
		
		# Marquer comme initialisé
		self._initialized = True
		
		# Log d'initialisation
		self.logger.info(f"Logger initialisé avec niveau {log_level.name}")
		if file_output:
			self.logger.info(f"Logs écrits dans {os.path.join(log_dir, self.log_filename)}")
	
	def _init_specialized_loggers(self) -> None:
		"""
		Initialise des loggers spécialisés pour différentes parties du système.
		"""
		# Créer des loggers pour chaque module principal
		module_names = [
			"core.entities", "core.environment", "core.physics", "core.genetics",
			"learning.models", "learning.trainers", "learning.rewards",
			"visualization", "evolution", "simulation"
		]
		
		self.module_loggers = {}
		
		for module_name in module_names:
			logger = logging.getLogger(f"marine_evolution.{module_name}")
			logger.setLevel(self.log_level.value)
			# Ne pas ajouter de handlers, ils héritent des handlers du logger parent
			logger.propagate = True
			self.module_loggers[module_name] = logger
	
	def get_module_logger(self, module_name: str) -> logging.Logger:
		"""
		Récupère un logger pour un module spécifique.
		
		Args:
			module_name: Nom du module (ex: 'core.entities')
			
		Returns:
			Logger pour le module spécifié
		"""
		if module_name in self.module_loggers:
			return self.module_loggers[module_name]
		
		# Si le module n'existe pas encore, le créer
		logger = logging.getLogger(f"marine_evolution.{module_name}")
		logger.setLevel(self.log_level.value)
		logger.propagate = True
		self.module_loggers[module_name] = logger
		
		return logger
	
	def debug(self, message: str, module: Optional[str] = None) -> None:
		"""
		Log un message de niveau DEBUG.
		
		Args:
			message: Message à journaliser
			module: Module concerné (optionnel)
		"""
		if module:
			self.get_module_logger(module).debug(message)
		else:
			self.logger.debug(message)
	
	def info(self, message: str, module: Optional[str] = None) -> None:
		"""
		Log un message de niveau INFO.
		
		Args:
			message: Message à journaliser
			module: Module concerné (optionnel)
		"""
		if module:
			self.get_module_logger(module).info(message)
		else:
			self.logger.info(message)
	
	def warning(self, message: str, module: Optional[str] = None) -> None:
		"""
		Log un message de niveau WARNING.
		
		Args:
			message: Message à journaliser
			module: Module concerné (optionnel)
		"""
		if module:
			self.get_module_logger(module).warning(message)
		else:
			self.logger.warning(message)
	
	def error(self, message: str, module: Optional[str] = None, exc_info: bool = False) -> None:
		"""
		Log un message de niveau ERROR.
		
		Args:
			message: Message à journaliser
			module: Module concerné (optionnel)
			exc_info: Si True, inclut les informations d'exception
		"""
		if module:
			self.get_module_logger(module).error(message, exc_info=exc_info)
		else:
			self.logger.error(message, exc_info=exc_info)
	
	def critical(self, message: str, module: Optional[str] = None, exc_info: bool = True) -> None:
		"""
		Log un message de niveau CRITICAL.
		
		Args:
			message: Message à journaliser
			module: Module concerné (optionnel)
			exc_info: Si True, inclut les informations d'exception
		"""
		if module:
			self.get_module_logger(module).critical(message, exc_info=exc_info)
		else:
			self.logger.critical(message, exc_info=exc_info)
	
	def log_evolution_event(self, 
						  event_type: str, 
						  generation: int, 
						  details: Dict[str, Any]) -> None:
		"""
		Journalise un événement d'évolution spécifique.
		
		Args:
			event_type: Type d'événement (ex: 'reproduction', 'mutation', 'selection')
			generation: Numéro de génération
			details: Détails de l'événement
		"""
		event_message = f"Evolution[{generation}] {event_type.upper()}: " + json.dumps(details, default=str)
		self.info(event_message, module="evolution")
		
		# Si c'est un événement important, on le journalise aussi en tant que métrique
		if event_type in ['new_species', 'extinction', 'significant_mutation']:
			self.log_metric('evolution_events', {
				'type': event_type,
				'generation': generation,
				'details': details,
				'timestamp': time.time()
			})
	
	def log_simulation_status(self, 
							status: str, 
							step: int, 
							metrics: Dict[str, Any]) -> None:
		"""
		Journalise l'état actuel de la simulation.
		
		Args:
			status: État de la simulation (ex: 'running', 'paused', 'completed')
			step: Étape actuelle de la simulation
			metrics: Métriques actuelles de la simulation
		"""
		status_message = f"Simulation[{step}] {status.upper()}: " + ', '.join([f"{k}={v}" for k, v in metrics.items()])
		self.info(status_message, module="simulation")
		
		# Ajouter aux métriques
		metrics_data = metrics.copy()
		metrics_data.update({
			'step': step,
			'status': status,
			'timestamp': time.time()
		})
		self.log_metric('simulation_status', metrics_data)
	
	def log_metric(self, category: str, data: Dict[str, Any]) -> None:
		"""
		Journalise une métrique.
		
		Args:
			category: Catégorie de la métrique
			data: Données de la métrique
		"""
		if not self.metrics_enabled:
			return
			
		if category not in self.metrics:
			self.metrics[category] = []
			
		# Ajouter un timestamp si non présent
		if 'timestamp' not in data:
			data['timestamp'] = time.time()
			
		self.metrics[category].append(data)
		
		# Sauvegarder périodiquement les métriques
		current_time = time.time()
		if current_time - self.last_metrics_time > self.metrics_interval:
			self.save_metrics()
			self.last_metrics_time = current_time
	
	def save_metrics(self, filepath: Optional[str] = None) -> None:
		"""
		Sauvegarde les métriques collectées dans un fichier JSON.
		
		Args:
			filepath: Chemin du fichier (optionnel, utilise le chemin par défaut si non spécifié)
		"""
		if not self.metrics_enabled or not self.metrics:
			return
			
		if filepath is None:
			filepath = os.path.join(self.log_dir, self.metrics_filename)
			
		try:
			with open(filepath, 'w') as f:
				json.dump(self.metrics, f, indent=2, default=str)
			self.debug(f"Métriques sauvegardées dans {filepath}")
		except Exception as e:
			self.error(f"Erreur lors de la sauvegarde des métriques: {str(e)}", exc_info=True)
	
	def set_log_level(self, level: LogLevel) -> None:
		"""
		Change le niveau de journalisation.
		
		Args:
			level: Nouveau niveau de journalisation
		"""
		self.log_level = level
		self.logger.setLevel(level.value)
		
		# Mettre à jour les loggers des modules
		for logger in self.module_loggers.values():
			logger.setLevel(level.value)
			
		self.info(f"Niveau de journalisation changé à {level.name}")
	
	def close(self) -> None:
		"""
		Ferme proprement le logger et sauvegarde les métriques restantes.
		"""
		self.save_metrics()
		self.info("Logger fermé")
		
		# Fermer les handlers
		for handler in self.logger.handlers:
			handler.close()
			
		# Réinitialiser l'instance singleton pour permettre une nouvelle initialisation
		SimulationLogger._instance = None
		self._initialized = False


# Fonction d'accès rapide au logger
def get_logger() -> SimulationLogger:
	"""
	Obtient l'instance unique du logger de simulation.
	Si le logger n'est pas encore initialisé, l'initialise avec les paramètres par défaut.
	
	Returns:
		Instance du SimulationLogger
	"""
	if SimulationLogger._instance is None or not SimulationLogger._instance._initialized:
		return SimulationLogger()
	return SimulationLogger._instance