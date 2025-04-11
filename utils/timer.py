import time
import threading
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
import statistics

class Timer:
	"""
	Chronomètre pour mesurer les performances et durées d'exécution.
	Permet de mesurer plusieurs sections de code indépendamment,
	et de générer des statistiques sur ces mesures.
	"""
	
	def __init__(self, name: str = "Timer") -> None:
		"""
		Initialise un nouveau chronomètre.
		
		Args:
			name: Nom du chronomètre (pour l'identification dans les logs)
		"""
		self.name = name
		self.start_time: Optional[float] = None
		self.end_time: Optional[float] = None
		self.elapsed_time: Optional[float] = None
		self.is_running: bool = False
		
		# Pour les mesures de sections
		self.sections: Dict[str, List[float]] = {}
		self.current_sections: Dict[str, float] = {}
		
		# Pour les mesures périodiques
		self.periodic_measurements: Dict[str, List[float]] = {}
		self.periodic_timers: Dict[str, threading.Timer] = {}
	
	def start(self) -> None:
		"""
		Démarre le chronomètre principal.
		"""
		if self.is_running:
			return
			
		self.start_time = time.time()
		self.is_running = True
		self.elapsed_time = None
	
	def stop(self) -> float:
		"""
		Arrête le chronomètre principal et retourne le temps écoulé.
		
		Returns:
			Temps écoulé en secondes depuis le démarrage
		"""
		if not self.is_running:
			return 0.0
			
		self.end_time = time.time()
		self.is_running = False
		self.elapsed_time = self.end_time - self.start_time
		return self.elapsed_time
	
	def reset(self) -> None:
		"""
		Réinitialise le chronomètre principal.
		"""
		self.start_time = None
		self.end_time = None
		self.elapsed_time = None
		self.is_running = False
	
	def get_elapsed(self) -> float:
		"""
		Retourne le temps écoulé depuis le démarrage, sans arrêter le chronomètre.
		
		Returns:
			Temps écoulé en secondes
		"""
		if not self.is_running:
			return self.elapsed_time or 0.0
			
		return time.time() - self.start_time
	
	def start_section(self, section_name: str) -> None:
		"""
		Démarre la mesure d'une section spécifique.
		
		Args:
			section_name: Nom de la section
		"""
		self.current_sections[section_name] = time.time()
	
	def stop_section(self, section_name: str) -> float:
		"""
		Arrête la mesure d'une section et enregistre la durée.
		
		Args:
			section_name: Nom de la section
			
		Returns:
			Temps écoulé pour cette section en secondes
		"""
		if section_name not in self.current_sections:
			return 0.0
			
		elapsed = time.time() - self.current_sections[section_name]
		
		if section_name not in self.sections:
			self.sections[section_name] = []
			
		self.sections[section_name].append(elapsed)
		del self.current_sections[section_name]
		
		return elapsed
	
	@contextmanager
	def section(self, section_name: str):
		"""
		Gestionnaire de contexte pour mesurer une section de code.
		
		Args:
			section_name: Nom de la section
			
		Yields:
			None
		"""
		self.start_section(section_name)
		try:
			yield
		finally:
			self.stop_section(section_name)
	
	def get_section_stats(self, section_name: str) -> Dict[str, float]:
		"""
		Retourne les statistiques pour une section spécifique.
		
		Args:
			section_name: Nom de la section
			
		Returns:
			Dictionnaire des statistiques (moyenne, min, max, etc.)
		"""
		if section_name not in self.sections or not self.sections[section_name]:
			return {
				"count": 0,
				"total": 0.0,
				"average": 0.0,
				"min": 0.0,
				"max": 0.0,
				"median": 0.0,
				"std_dev": 0.0
			}
			
		measurements = self.sections[section_name]
		
		return {
			"count": len(measurements),
			"total": sum(measurements),
			"average": statistics.mean(measurements),
			"min": min(measurements),
			"max": max(measurements),
			"median": statistics.median(measurements),
			"std_dev": statistics.stdev(measurements) if len(measurements) > 1 else 0.0
		}
	
	def get_all_section_stats(self) -> Dict[str, Dict[str, float]]:
		"""
		Retourne les statistiques pour toutes les sections.
		
		Returns:
			Dictionnaire des statistiques par section
		"""
		return {section: self.get_section_stats(section) for section in self.sections}
	
	def start_periodic_measurement(self, 
								  name: str, 
								  interval: float, 
								  callback: Optional[Callable[[str, float], None]] = None) -> None:
		"""
		Démarre une mesure périodique d'une fonction ou métrique.
		
		Args:
			name: Nom de la mesure
			interval: Intervalle en secondes entre les mesures
			callback: Fonction à appeler avec le résultat de chaque mesure
		"""
		if name in self.periodic_timers:
			# Arrêter le timer existant
			self.stop_periodic_measurement(name)
			
		# Initialiser la liste de mesures
		if name not in self.periodic_measurements:
			self.periodic_measurements[name] = []
			
		# Fonction de mesure périodique
		def measure():
			elapsed = self.get_elapsed()
			self.periodic_measurements[name].append(elapsed)
			
			if callback:
				callback(name, elapsed)
				
			# Reprogrammer la prochaine mesure
			if self.is_running:
				timer = threading.Timer(interval, measure)
				timer.daemon = True
				timer.start()
				self.periodic_timers[name] = timer
		
		# Démarrer le timer
		timer = threading.Timer(interval, measure)
		timer.daemon = True
		timer.start()
		self.periodic_timers[name] = timer
	
	def stop_periodic_measurement(self, name: str) -> None:
		"""
		Arrête une mesure périodique.
		
		Args:
			name: Nom de la mesure à arrêter
		"""
		if name in self.periodic_timers:
			self.periodic_timers[name].cancel()
			del self.periodic_timers[name]
	
	def get_periodic_measurement_stats(self, name: str) -> Dict[str, float]:
		"""
		Retourne les statistiques pour une mesure périodique spécifique.
		
		Args:
			name: Nom de la mesure
			
		Returns:
			Dictionnaire des statistiques
		"""
		if name not in self.periodic_measurements or not self.periodic_measurements[name]:
			return {
				"count": 0,
				"average": 0.0,
				"min": 0.0,
				"max": 0.0,
				"last": 0.0
			}
			
		measurements = self.periodic_measurements[name]
		
		return {
			"count": len(measurements),
			"average": statistics.mean(measurements),
			"min": min(measurements),
			"max": max(measurements),
			"last": measurements[-1]
		}
	
	def clear_section(self, section_name: str) -> None:
		"""
		Efface les mesures enregistrées pour une section spécifique.
		
		Args:
			section_name: Nom de la section à effacer
		"""
		if section_name in self.sections:
			self.sections[section_name] = []
	
	def clear_all_sections(self) -> None:
		"""
		Efface les mesures enregistrées pour toutes les sections.
		"""
		for section in self.sections:
			self.sections[section] = []
	
	def __enter__(self) -> 'Timer':
		"""
		Support pour l'utilisation avec le gestionnaire de contexte (with).
		
		Returns:
			L'instance du timer
		"""
		self.start()
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb) -> None:
		"""
		Méthode de sortie pour le gestionnaire de contexte.
		
		Args:
			exc_type: Type d'exception (si une exception s'est produite)
			exc_val: Valeur de l'exception
			exc_tb: Traceback de l'exception
		"""
		self.stop()
	
	def get_formatted_time(self, seconds: float) -> str:
		"""
		Convertit un temps en secondes en une chaîne formatée.
		
		Args:
			seconds: Temps en secondes
			
		Returns:
			Temps formaté (ex: "2h 30m 45.3s")
		"""
		hours, remainder = divmod(seconds, 3600)
		minutes, seconds = divmod(remainder, 60)
		
		if hours > 0:
			return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"
		elif minutes > 0:
			return f"{int(minutes)}m {seconds:.1f}s"
		else:
			return f"{seconds:.3f}s"
	
	def get_summary(self) -> str:
		"""
		Génère un résumé du timer et de ses sections.
		
		Returns:
			Chaîne formatée contenant le résumé du timer
		"""
		summary = [f"Timer '{self.name}':"]
		
		# Temps total
		elapsed = self.get_elapsed()
		summary.append(f"  Total time: {self.get_formatted_time(elapsed)}")
		
		# Statistiques des sections
		if self.sections:
			summary.append("  Sections:")
			section_stats = self.get_all_section_stats()
			
			# Trier les sections par temps total
			sorted_sections = sorted(
				section_stats.items(), 
				key=lambda x: x[1]["total"], 
				reverse=True
			)
			
			for section_name, stats in sorted_sections:
				total = stats["total"]
				avg = stats["average"]
				count = stats["count"]
				percentage = (total / elapsed * 100) if elapsed > 0 else 0
				summary.append(f"    {section_name}: {self.get_formatted_time(total)} "
							  f"({percentage:.1f}%) - {count} calls, avg: {self.get_formatted_time(avg)}")
				
		return "\n".join(summary)


class PerformanceMonitor:
	"""
	Moniteur de performance pour suivre l'utilisation des ressources et
	les métriques de performance de la simulation.
	"""
	
	def __init__(self, sampling_interval: float = 1.0) -> None:
		"""
		Initialise le moniteur de performance.
		
		Args:
			sampling_interval: Intervalle d'échantillonnage en secondes
		"""
		self.sampling_interval = sampling_interval
		self.timers: Dict[str, Timer] = {}
		self.metrics: Dict[str, List[float]] = {}
		self.is_monitoring = False
		self.monitor_thread = None
	
	def add_timer(self, name: str) -> Timer:
		"""
		Ajoute un nouveau timer au moniteur.
		
		Args:
			name: Nom du timer
			
		Returns:
			Le timer créé
		"""
		timer = Timer(name)
		self.timers[name] = timer
		return timer
	
	def get_timer(self, name: str) -> Timer:
		"""
		Récupère un timer existant, ou en crée un nouveau s'il n'existe pas.
		
		Args:
			name: Nom du timer
			
		Returns:
			Le timer demandé
		"""
		if name not in self.timers:
			return self.add_timer(name)
		return self.timers[name]
	
	def start_monitoring(self) -> None:
		"""
		Démarre la surveillance des performances système.
		Collecte les métriques d'utilisation CPU, mémoire, etc.
		"""
		if self.is_monitoring:
			return
			
		self.is_monitoring = True
		
		# Créer et démarrer le thread de surveillance
		self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
		self.monitor_thread.start()
	
	def stop_monitoring(self) -> None:
		"""
		Arrête la surveillance des performances système.
		"""
		self.is_monitoring = False
		if self.monitor_thread:
			self.monitor_thread.join(timeout=1.0)
			self.monitor_thread = None
	
	def _monitor_loop(self) -> None:
		"""
		Boucle de surveillance des performances système.
		Collecte périodiquement les métriques.
		"""
		try:
			# Importer psutil si disponible (pour les métriques système)
			import psutil
			has_psutil = True
		except ImportError:
			has_psutil = False
			
		while self.is_monitoring:
			if has_psutil:
				# Collecter les métriques d'utilisation CPU
				cpu_percent = psutil.cpu_percent(interval=None)
				self.log_metric("cpu_percent", cpu_percent)
				
				# Collecter les métriques d'utilisation mémoire
				memory = psutil.virtual_memory()
				self.log_metric("memory_percent", memory.percent)
				self.log_metric("memory_used_mb", memory.used / (1024 * 1024))
				
				# Collecter les métriques de l'application
				process = psutil.Process()
				self.log_metric("process_cpu_percent", process.cpu_percent(interval=None))
				self.log_metric("process_memory_mb", process.memory_info().rss / (1024 * 1024))
				
				# Collecter les métriques de disque
				disk = psutil.disk_usage('/')
				self.log_metric("disk_percent", disk.percent)
				
			# Collecter les métriques des timers
			for name, timer in self.timers.items():
				for section, stats in timer.get_all_section_stats().items():
					metric_name = f"{name}_{section}_avg"
					self.log_metric(metric_name, stats["average"])
			
			# Attendre le prochain échantillonnage
			time.sleep(self.sampling_interval)
	
	def log_metric(self, name: str, value: float) -> None:
		"""
		Enregistre une métrique de performance.
		
		Args:
			name: Nom de la métrique
			value: Valeur de la métrique
		"""
		if name not in self.metrics:
			self.metrics[name] = []
			
		self.metrics[name].append(value)
		
		# Limiter la taille de l'historique (garder 1000 dernières valeurs)
		if len(self.metrics[name]) > 1000:
			self.metrics[name] = self.metrics[name][-1000:]
	
	def get_metric_stats(self, name: str) -> Dict[str, float]:
		"""
		Récupère les statistiques pour une métrique spécifique.
		
		Args:
			name: Nom de la métrique
			
		Returns:
			Dictionnaire des statistiques
		"""
		if name not in self.metrics or not self.metrics[name]:
			return {
				"count": 0,
				"current": 0.0,
				"average": 0.0,
				"min": 0.0,
				"max": 0.0
			}
			
		measurements = self.metrics[name]
		
		return {
			"count": len(measurements),
			"current": measurements[-1],
			"average": statistics.mean(measurements),
			"min": min(measurements),
			"max": max(measurements)
		}
	
	def get_all_metric_stats(self) -> Dict[str, Dict[str, float]]:
		"""
		Récupère les statistiques pour toutes les métriques.
		
		Returns:
			Dictionnaire des statistiques par métrique
		"""
		return {metric: self.get_metric_stats(metric) for metric in self.metrics}
	
	def get_summary(self) -> str:
		"""
		Génère un résumé des performances.
		
		Returns:
			Chaîne formatée contenant le résumé des performances
		"""
		summary = ["Performance Summary:"]
		
		# Résumé des timers
		for name, timer in self.timers.items():
			summary.append(timer.get_summary())
			
		# Résumé des métriques
		summary.append("\nMetrics:")
		metric_stats = self.get_all_metric_stats()
		
		# Grouper les métriques par catégorie
		categorized_metrics = {}
		for metric_name, stats in metric_stats.items():
			category = metric_name.split('_')[0] if '_' in metric_name else 'other'
			if category not in categorized_metrics:
				categorized_metrics[category] = []
			categorized_metrics[category].append((metric_name, stats))
		
		# Afficher les métriques par catégorie
		for category, metrics in categorized_metrics.items():
			summary.append(f"  {category.upper()}:")
			for metric_name, stats in metrics:
				name = metric_name[len(category)+1:] if category != 'other' else metric_name
				summary.append(f"    {name}: current={stats['current']:.2f}, avg={stats['average']:.2f}, "
							  f"min={stats['min']:.2f}, max={stats['max']:.2f}")
				
		return "\n".join(summary)