import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Any, Optional
import time
import threading
import os

class SimulationVisualizer:
	"""
	Visualiseur de simulation pour la constellation de satellites.
	Prend en charge les visualisations 2D et 3D.
	"""
	
	def __init__(self, mode: str = '3d', figsize: Tuple[int, int] = (12, 10)):
		"""
		Initialise le visualiseur.
		
		Args:
			mode: Mode de visualisation ('2d' ou '3d')
			figsize: Taille de la figure (largeur, hauteur)
		"""
		self.mode = mode
		self.figsize = figsize
		
		# État de la simulation
		self.satellites = []
		self.groundStations = []
		self.earthModel = None
		self.metrics = {}
		self.time = 0.0
		
		# Figure et axes
		self.fig = None
		self.axes = {}
		self.plots = {}
		
		# Animation
		self.animationRunning = False
		self.animationThread = None
		
		# Initialiser la visualisation
		self._setupVisualization()
	
	def _setupVisualization(self) -> None:
		"""
		Configure la visualisation selon le mode.
		"""
		# Créer la figure
		self.fig = plt.figure(figsize=self.figsize)
		
		if self.mode == '3d':
			# Vue 3D principale
			self.axes['3d'] = self.fig.add_subplot(2, 2, 1, projection='3d')
			self.axes['3d'].set_title('Constellation de satellites')
			
			# Vue 2D de la couverture (projection)
			self.axes['coverage'] = self.fig.add_subplot(2, 2, 2)
			self.axes['coverage'].set_title('Carte de couverture')
			
			# Vue des métriques au fil du temps
			self.axes['metrics'] = self.fig.add_subplot(2, 2, 3)
			self.axes['metrics'].set_title('Métriques')
			
			# Vue du réseau de connectivité
			self.axes['network'] = self.fig.add_subplot(2, 2, 4)
			self.axes['network'].set_title('Graphe de connectivité')
		else:
			# Vue 2D principale (projection)
			self.axes['2d'] = self.fig.add_subplot(2, 2, 1)
			self.axes['2d'].set_title('Projection de la constellation')
			
			# Vue de la couverture
			self.axes['coverage'] = self.fig.add_subplot(2, 2, 2)
			self.axes['coverage'].set_title('Carte de couverture')
			
			# Vue des métriques au fil du temps
			self.axes['metrics'] = self.fig.add_subplot(2, 2, 3)
			self.axes['metrics'].set_title('Métriques')
			
			# Informations textuelles
			self.axes['info'] = self.fig.add_subplot(2, 2, 4)
			self.axes['info'].set_title('Informations')
			self.axes['info'].axis('off')
		
		self.fig.tight_layout()
		self.fig.subplots_adjust(hspace=0.3, wspace=0.3)
		
		# Historique des métriques
		self.metricHistory = {
			'time': [],
			'coverage': [],
			'data_throughput': [],
			'energy_efficiency': [],
			'user_satisfaction': []
		}
	
	def reset(self, satellites: List[Any], groundStations: List[Any], earthModel: Any) -> None:
		"""
		Réinitialise la visualisation avec de nouveaux objets.
		
		Args:
			satellites: Liste des satellites
			groundStations: Liste des stations au sol
			earthModel: Modèle de la Terre
		"""
		self.satellites = satellites
		self.groundStations = groundStations
		self.earthModel = earthModel
		self.metrics = {}
		self.time = 0.0
		
		# Réinitialiser l'historique des métriques
		self.metricHistory = {
			'time': [],
			'coverage': [],
			'data_throughput': [],
			'energy_efficiency': [],
			'user_satisfaction': []
		}
		
		# Mettre à jour la visualisation
		self.update(satellites, groundStations, {}, 0.0)
	
	def update(self, satellites: List[Any], groundStations: List[Any], metrics: Dict[str, float], time: float) -> None:
		"""
		Met à jour la visualisation avec les données actuelles.
		
		Args:
			satellites: Liste des satellites
			groundStations: Liste des stations au sol
			metrics: Métriques globales
			time: Temps de simulation
		"""
		self.satellites = satellites
		self.groundStations = groundStations
		self.metrics = metrics
		self.time = time
		
		# Mettre à jour l'historique des métriques
		self.metricHistory['time'].append(time)
		self.metricHistory['coverage'].append(metrics.get('coverage', 0.0))
		self.metricHistory['data_throughput'].append(metrics.get('data_throughput', 0.0))
		self.metricHistory['energy_efficiency'].append(metrics.get('energy_efficiency', 0.0))
		self.metricHistory['user_satisfaction'].append(metrics.get('user_satisfaction', 0.0))
		
		# Limiter l'historique à 1000 points
		if len(self.metricHistory['time']) > 1000:
			for key in self.metricHistory:
				self.metricHistory[key] = self.metricHistory[key][-1000:]
	
	def render(self, mode: str = 'human') -> Optional[np.ndarray]:
		"""
		Rend la visualisation actuelle.
		
		Args:
			mode: Mode de rendu ('human', 'rgb_array')
			
		Returns:
			Image rendue en mode 'rgb_array', None sinon
		"""
		# Effacer toutes les axes
		for ax in self.axes.values():
			ax.clear()
		
		# Rendre selon le mode de visualisation
		if self.mode == '3d':
			self._render3D()
		else:
			self._render2D()
		
		# Ajuster la mise en page
		self.fig.tight_layout()
		
		# Afficher ou retourner l'image
		if mode == 'human':
			plt.pause(0.01)
			return None
		elif mode == 'rgb_array':
			# Rendre la figure en tableau numpy
			canvas = FigureCanvasAgg(self.fig)
			canvas.draw()
			img = np.array(canvas.renderer.buffer_rgba())
			return img
	
	def _render3D(self) -> None:
		"""
		Effectue le rendu en mode 3D.
		"""
		# 1. Vue 3D
		ax = self.axes['3d']
		ax.set_title('Constellation de satellites')
		ax.set_xlabel('X (km)')
		ax.set_ylabel('Y (km)')
		ax.set_zlabel('Z (km)')
		
		# Dessiner la Terre
		self._drawEarth(ax)
		
		# Dessiner les satellites
		self._drawSatellites(ax)
		
		# Dessiner les stations au sol
		self._drawGroundStations(ax)
		
		# Dessiner les connections entre satellites visibles
		self._drawConnections(ax)
		
		# Configurer la vue
		earthRadius = 6371.0  # km
		ax.set_xlim(-earthRadius*2, earthRadius*2)
		ax.set_ylim(-earthRadius*2, earthRadius*2)
		ax.set_zlim(-earthRadius*2, earthRadius*2)
		
		# Égaliser la vue
		ax.set_box_aspect([1, 1, 1])
		
		# 2. Carte de couverture
		ax = self.axes['coverage']
		ax.set_title('Carte de couverture')
		self._drawCoverageMap(ax)
		
		# 3. Métriques
		ax = self.axes['metrics']
		ax.set_title('Métriques au fil du temps')
		self._drawMetricHistory(ax)
		
		# 4. Réseau de connectivité
		ax = self.axes['network']
		ax.set_title('Graphe de connectivité')
		self._drawNetworkGraph(ax)
	
	def _render2D(self) -> None:
		"""
		Effectue le rendu en mode 2D.
		"""
		# 1. Vue 2D (projection)
		ax = self.axes['2d']
		ax.set_title('Projection de la constellation')
		ax.set_xlabel('Longitude (°)')
		ax.set_ylabel('Latitude (°)')
		
		# Dessiner la carte du monde
		self._drawWorldMap(ax)
		
		# Dessiner les satellites (projection sur la carte)
		self._drawSatellitesProjection(ax)
		
		# Dessiner les stations au sol
		self._drawGroundStationsProjection(ax)
		
		# 2. Carte de couverture
		ax = self.axes['coverage']
		ax.set_title('Carte de couverture')
		self._drawCoverageMap(ax)
		
		# 3. Métriques
		ax = self.axes['metrics']
		ax.set_title('Métriques au fil du temps')
		self._drawMetricHistory(ax)
		
		# 4. Informations textuelles
		ax = self.axes['info']
		ax.set_title('Informations')
		ax.axis('off')
		self._drawTextInfo(ax)
	
	def _drawEarth(self, ax: plt.Axes) -> None:
		"""
		Dessine le globe terrestre en 3D.
		
		Args:
			ax: Axes où dessiner
		"""
		# Rayon de la Terre
		earthRadius = 6371.0  # km
		
		# Créer le wireframe de la Terre
		u = np.linspace(0, 2 * np.pi, 30)
		v = np.linspace(0, np.pi, 30)
		x = earthRadius * np.outer(np.cos(u), np.sin(v))
		y = earthRadius * np.outer(np.sin(u), np.sin(v))
		z = earthRadius * np.outer(np.ones(np.size(u)), np.cos(v))
		
		# Dessiner la Terre semi-transparente
		ax.plot_surface(x, y, z, color='blue', alpha=0.2)
		
		# Dessiner les axes de référence
		ax.quiver(0, 0, 0, earthRadius * 1.5, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
		ax.quiver(0, 0, 0, 0, earthRadius * 1.5, 0, color='g', arrow_length_ratio=0.1, label='Y')
		ax.quiver(0, 0, 0, 0, 0, earthRadius * 1.5, color='b', arrow_length_ratio=0.1, label='Z')
	
	def _drawSatellites(self, ax: plt.Axes) -> None:
		"""
		Dessine les satellites en 3D.
		
		Args:
			ax: Axes où dessiner
		"""
		for satellite in self.satellites:
			position = satellite.getPosition()
			
			# Couleur dépendant du niveau de batterie
			batteryLevel = satellite.state.batteryLevel
			color = plt.cm.RdYlGn(batteryLevel)  # Rouge à faible niveau, vert à niveau élevé
			
			# Dessiner le satellite
			ax.scatter(position[0], position[1], position[2], color=color, s=20, edgecolors='black')
			
			# Dessiner une petite traînée pour visualiser la direction
			if hasattr(satellite, 'stateHistory') and len(satellite.stateHistory) > 1:
				history = satellite.stateHistory[-10:]  # Dernières 10 positions
				xs = [s.position[0] for s in history]
				ys = [s.position[1] for s in history]
				zs = [s.position[2] for s in history]
				ax.plot(xs, ys, zs, color=color, alpha=0.3, linewidth=1)
	
	def _drawGroundStations(self, ax: plt.Axes) -> None:
		"""
		Dessine les stations au sol en 3D.
		
		Args:
			ax: Axes où dessiner
		"""
		earthRadius = 6371.0  # km
		
		for station in self.groundStations:
			position = station.getPosition()
			
			# Dessiner la station au sol
			ax.scatter(position[0], position[1], position[2], color='orange', s=30, marker='^', edgecolors='black')
			
			# Dessiner une ligne de la station au centre de la Terre
			ax.plot([0, position[0]], [0, position[1]], [0, position[2]], 'k-', alpha=0.1)
	
	def _drawConnections(self, ax: plt.Axes) -> None:
		"""
		Dessine les connexions entre satellites visibles.
		
		Args:
			ax: Axes où dessiner
		"""
		for satellite in self.satellites:
			position = satellite.getPosition()
			
			# Connexions vers les autres satellites
			for visibleId in satellite.visibleSatellites:
				# Trouver le satellite visible
				for otherSat in self.satellites:
					if otherSat.satelliteId == visibleId:
						otherPos = otherSat.getPosition()
						
						# Dessiner la connexion
						ax.plot([position[0], otherPos[0]],
								[position[1], otherPos[1]],
								[position[2], otherPos[2]],
								'g-', alpha=0.2, linewidth=0.5)
						break
			
			# Connexions vers les stations au sol
			for visibleId in satellite.visibleGroundStations:
				# Trouver la station visible
				for station in self.groundStations:
					if station.stationId == visibleId:
						stationPos = station.getPosition()
						
						# Dessiner la connexion
						ax.plot([position[0], stationPos[0]],
								[position[1], stationPos[1]],
								[position[2], stationPos[2]],
								'y-', alpha=0.5, linewidth=1)
						break
	
	def _drawWorldMap(self, ax: plt.Axes) -> None:
		"""
		Dessine une carte du monde simple en 2D.
		
		Args:
			ax: Axes où dessiner
		"""
		# Configuré les limites de la carte
		ax.set_xlim(-180, 180)
		ax.set_ylim(-90, 90)
		
		# Ajouter un grid pour les coordonnées
		ax.grid(True, linestyle='--', alpha=0.5)
		
		# Ajouter les lignes d'équateur et des méridiens principaux
		ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Équateur
		ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)  # Méridien de Greenwich
		
		# Ajouter les tropiques
		ax.axhline(y=23.5, color='k', linestyle='--', alpha=0.3)  # Tropique du Cancer
		ax.axhline(y=-23.5, color='k', linestyle='--', alpha=0.3)  # Tropique du Capricorne
		
		# Ajouter les cercles polaires
		ax.axhline(y=66.5, color='k', linestyle='--', alpha=0.3)  # Cercle polaire arctique
		ax.axhline(y=-66.5, color='k', linestyle='--', alpha=0.3)  # Cercle polaire antarctique
	
	def _drawSatellitesProjection(self, ax: plt.Axes) -> None:
		"""
		Dessine les satellites projetés sur la carte 2D.
		
		Args:
			ax: Axes où dessiner
		"""
		for satellite in self.satellites:
			position = satellite.getPosition()
			
			# Convertir les coordonnées cartésiennes en coordonnées géographiques
			if self.earthModel is not None:
				lat, lon, alt = self.earthModel.cartesianToGeodetic(position)
			else:
				# Conversion simple si earthModel n'est pas disponible
				r = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
				lon = np.degrees(np.arctan2(position[1], position[0]))
				lat = np.degrees(np.arcsin(position[2] / r))
				alt = r - 6371.0  # km au-dessus de la surface
			
			# Couleur dépendant du niveau de batterie
			batteryLevel = satellite.state.batteryLevel
			color = plt.cm.RdYlGn(batteryLevel)  # Rouge à faible niveau, vert à niveau élevé
			
			# Dessiner le satellite
			ax.scatter(lon, lat, color=color, s=20, edgecolors='black')
			
			# Dessiner l'empreinte du satellite
			# Simplification: cercle autour de la position du satellite
			footprint_radius = np.degrees(np.arccos(6371.0 / (6371.0 + alt)))
			
			# Dessiner le cercle d'empreinte
			if footprint_radius > 0:
				circle = plt.Circle((lon, lat), footprint_radius, color=color, fill=False, alpha=0.2)
				ax.add_patch(circle)
	
	def _drawGroundStationsProjection(self, ax: plt.Axes) -> None:
		"""
		Dessine les stations au sol sur la carte 2D.
		
		Args:
			ax: Axes où dessiner
		"""
		for station in self.groundStations:
			# Si la station a des coordonnées géographiques directes
			if hasattr(station, 'latitude') and hasattr(station, 'longitude'):
				lat = station.latitude
				lon = station.longitude
			else:
				position = station.getPosition()
				
				# Convertir les coordonnées cartésiennes en coordonnées géographiques
				if self.earthModel is not None:
					lat, lon, _ = self.earthModel.cartesianToGeodetic(position)
				else:
					# Conversion simple si earthModel n'est pas disponible
					r = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
					lon = np.degrees(np.arctan2(position[1], position[0]))
					lat = np.degrees(np.arcsin(position[2] / r))
			
			# Dessiner la station au sol
			ax.scatter(lon, lat, color='orange', s=30, marker='^', edgecolors='black')
			
			# Ajouter le nom si disponible
			if hasattr(station, 'name'):
				ax.text(lon, lat, station.name, fontsize=8, ha='right')
	
	def _drawCoverageMap(self, ax: plt.Axes) -> None:
		"""
		Dessine la carte de couverture.
		
		Args:
			ax: Axes où dessiner
		"""
		# Configurer les limites de la carte
		ax.set_xlim(-180, 180)
		ax.set_ylim(-90, 90)
		
		# Ajouter un grid pour les coordonnées
		ax.grid(True, linestyle='--', alpha=0.5)
		
		# Générer une carte de couverture à partir de la constellation
		resolution = 60  # divisions par hémisphère
		coverageMask = np.zeros((2*resolution, 2*resolution))
		
		# Si nous avons un objet constellation, utiliser sa méthode getCoverageMask
		if hasattr(self.satellites, 'getCoverageMask'):
			coverageMask = self.satellites.getCoverageMask(resolution)
		else:
			# Sinon, calculer la couverture manuellement
			for satellite in self.satellites:
				position = satellite.getPosition()
				
				# Convertir les coordonnées cartésiennes en coordonnées géographiques
				if self.earthModel is not None:
					lat, lon, alt = self.earthModel.cartesianToGeodetic(position)
				else:
					# Conversion simple si earthModel n'est pas disponible
					r = np.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
					lon = np.degrees(np.arctan2(position[1], position[0]))
					lat = np.degrees(np.arcsin(position[2] / r))
					alt = r - 6371.0  # km au-dessus de la surface
				
				# Calculer l'empreinte du satellite
				footprint_radius = np.degrees(np.arccos(6371.0 / (6371.0 + alt)))
				
				# Mettre à jour la matrice de couverture
				for i in range(2*resolution):
					lat_grid = 90 - i * 180 / (2*resolution)  # De 90° à -90°
					
					for j in range(2*resolution):
						lon_grid = -180 + j * 360 / (2*resolution)  # De -180° à 180°
						
						# Calcul de la distance sphérique
						dlat = np.radians(lat_grid - lat)
						dlon = np.radians(lon_grid - lon)
						a = np.sin(dlat/2)**2 + np.cos(np.radians(lat)) * np.cos(np.radians(lat_grid)) * np.sin(dlon/2)**2
						distance = 2 * np.arcsin(np.sqrt(a))
						distance_degrees = np.degrees(distance)
						
						# Si la distance est inférieure au rayon d'empreinte, le point est couvert
						if distance_degrees < footprint_radius:
							coverageMask[i, j] = 1
		
		# Créer une colormap personnalisée
		cmap = LinearSegmentedColormap.from_list('coverage', ['black', 'blue'], N=2)
		
		# Afficher la carte de couverture
		extent = [-180, 180, -90, 90]
		ax.imshow(coverageMask, extent=extent, origin='upper', cmap=cmap, alpha=0.7)
		
		# Ajouter les lignes de référence
		ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Équateur
		ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)  # Méridien de Greenwich
	
	def _drawMetricHistory(self, ax: plt.Axes) -> None:
		"""
		Dessine l'historique des métriques.
		
		Args:
			ax: Axes où dessiner
		"""
		# Nettoyer l'axe
		ax.set_xlabel('Temps de simulation (s)')
		ax.set_ylabel('Valeur')
		
		# S'assurer qu'il y a des données
		if len(self.metricHistory['time']) == 0:
			return
		
		# Extraire les données
		time = np.array(self.metricHistory['time'])
		coverage = np.array(self.metricHistory['coverage']) * 100  # Convertir en pourcentage
		data_throughput = np.array(self.metricHistory['data_throughput']) * 100  # Convertir en pourcentage
		energy_efficiency = np.array(self.metricHistory['energy_efficiency']) * 100  # Convertir en pourcentage
		user_satisfaction = np.array(self.metricHistory['user_satisfaction']) * 100  # Convertir en pourcentage
		
		# Tracer les métriques
		ax.plot(time, coverage, 'b-', label='Couverture (%)')
		ax.plot(time, data_throughput, 'g-', label='Débit (%)')
		ax.plot(time, energy_efficiency, 'r-', label='Efficacité (%)')
		ax.plot(time, user_satisfaction, 'y-', label='Satisfaction (%)')
		
		# Ajouter la légende
		ax.legend(loc='upper left', fontsize='small')
		
		# Limiter l'axe Y entre 0 et 100 (pourcentage)
		ax.set_ylim(0, 100)
	
	def _drawNetworkGraph(self, ax: plt.Axes) -> None:
		"""
		Dessine le graphe de connectivité du réseau.
		
		Args:
			ax: Axes où dessiner
		"""
		# Nettoyer l'axe
		ax.set_title('Graphe de connectivité')
		ax.axis('off')
		
		# Extraire les nœuds et les arêtes
		nodes = []
		edges = []
		
		# Si nous avons une méthode getConnectivityGraph, l'utiliser
		if hasattr(self.satellites, 'getConnectivityGraph'):
			nodes, edges = self.satellites.getConnectivityGraph()
		else:
			# Sinon, construire manuellement le graphe
			nodes = [sat.satelliteId for sat in self.satellites]
			edges = []
			
			for satellite in self.satellites:
				for visibleId in satellite.visibleSatellites:
					# Ajouter une arête seulement dans une direction pour éviter les doublons
					if satellite.satelliteId < visibleId:
						edges.append((satellite.satelliteId, visibleId))
		
		# Disposition des nœuds en cercle
		node_positions = {}
		n = len(nodes)
		
		if n > 0:
			for i, node in enumerate(nodes):
				angle = 2 * np.pi * i / n
				node_positions[node] = (np.cos(angle), np.sin(angle))
		
			# Dessiner les arêtes
			for source, target in edges:
				x1, y1 = node_positions[source]
				x2, y2 = node_positions[target]
				ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.6, linewidth=0.5)
			
			# Dessiner les nœuds
			for node in nodes:
				x, y = node_positions[node]
				
				# Rechercher le satellite correspondant
				satellite = None
				for sat in self.satellites:
					if sat.satelliteId == node:
						satellite = sat
						break
				
				# Couleur dépendant du niveau de batterie si disponible
				if satellite is not None:
					batteryLevel = satellite.state.batteryLevel
					color = plt.cm.RdYlGn(batteryLevel)
				else:
					color = 'blue'
				
				ax.scatter(x, y, color=color, s=100, edgecolors='black')
				ax.text(x, y, str(node), ha='center', va='center', fontsize=8, color='white')
	
	def _drawTextInfo(self, ax: plt.Axes) -> None:
		"""
		Affiche les informations textuelles.
		
		Args:
			ax: Axes où dessiner
		"""
		# Informations sur la simulation
		simInfo = [
			f"Temps de simulation: {self.time:.1f} s",
			f"Nombre de satellites: {len(self.satellites)}",
			f"Nombre de stations au sol: {len(self.groundStations)}",
			"",
			"Métriques globales:",
			f"- Couverture: {self.metrics.get('coverage', 0.0)*100:.1f}%",
			f"- Débit de données: {self.metrics.get('data_throughput', 0.0)*100:.1f}%",
			f"- Efficacité énergétique: {self.metrics.get('energy_efficiency', 0.0)*100:.1f}%",
			f"- Résilience du réseau: {self.metrics.get('network_resilience', 0.0)*100:.1f}%",
			f"- Satisfaction utilisateur: {self.metrics.get('user_satisfaction', 0.0)*100:.1f}%",
			"",
			"Statistiques des satellites:",
			f"- Satellites actifs: {len([s for s in self.satellites if not s.activeFaults])}",
			f"- Satellites en défaut: {len([s for s in self.satellites if s.activeFaults])}"
		]
		
		# Ajouter les 5 premiers satellites avec leurs niveaux de batterie
		top5_satellites = sorted(self.satellites, key=lambda s: s.satelliteId)[:5]
		if top5_satellites:
			simInfo.append("")
			simInfo.append("Échantillon de satellites:")
			for sat in top5_satellites:
				batteryStr = f"{sat.state.batteryLevel*100:.1f}%"
				eclipsedStr = "éclipsé" if sat.isEclipsed else "exposé"
				faultsStr = ", ".join(sat.activeFaults) if sat.activeFaults else "aucun"
				simInfo.append(f"  Sat #{sat.satelliteId}: Batterie: {batteryStr}, {eclipsedStr}, Défauts: {faultsStr}")
		
		# Afficher le texte
		ax.text(0.02, 0.98, "\n".join(simInfo), 
				ha='left', va='top', fontsize=9, 
				transform=ax.transAxes,
				bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
	
	def startAnimation(self) -> None:
		"""
		Démarre l'animation en arrière-plan.
		"""
		if self.animationRunning:
			return
		
		self.animationRunning = True
		self.animationThread = threading.Thread(target=self._animationLoop)
		self.animationThread.daemon = True
		self.animationThread.start()
	
	def stopAnimation(self) -> None:
		"""
		Arrête l'animation en arrière-plan.
		"""
		self.animationRunning = False
		if self.animationThread is not None:
			self.animationThread.join(timeout=1.0)
	
	def _animationLoop(self) -> None:
		"""
		Boucle d'animation en arrière-plan.
		"""
		while self.animationRunning:
			self.render(mode='human')
			time.sleep(0.1)
	
	def saveFrame(self, filePath: str) -> None:
		"""
		Enregistre une image de la visualisation actuelle.
		
		Args:
			filePath: Chemin du fichier où enregistrer l'image
		"""
		# Rendre la figure
		self.render(mode='human')
		
		# Enregistrer l'image
		self.fig.savefig(filePath, dpi=150, bbox_inches='tight')
		print(f"Image enregistrée dans {filePath}")
	
	def close(self) -> None:
		"""
		Ferme la visualisation et libère les ressources.
		"""
		self.stopAnimation()
		plt.close(self.fig)
		self.fig = None
		self.axes = {}
		self.plots = {}