import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.artist import Artist
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict, Any, Optional, Tuple

from environment.utils.OrbitalElements import OrbitalElements
from environment.Satellite import Satellite

class OrbitRenderer:
	"""
	Classe pour le rendu des orbites des satellites.
	"""
	
	def __init__(self, earthRadius: float = 6371.0):
		"""
		Initialise le renderer d'orbites.
		
		Args:
			earthRadius: Rayon de la Terre en km
		"""
		self.earthRadius = earthRadius
		self.orbitCache: Dict[int, Dict[str, Any]] = {}  # Cache pour les orbites déjà calculées
		self.colormap = plt.cm.viridis  # Colormap pour distinguer les différentes orbites
	
	def renderOrbits(self, ax: Axes3D, satellites: List[Satellite], 
				   numPoints: int = 100, showLabels: bool = False,
				   colors: Optional[List[Any]] = None) -> List[Artist]:
		"""
		Dessine les orbites complètes des satellites sur un axe 3D.
		
		Args:
			ax: Axe 3D Matplotlib
			satellites: Liste des satellites à tracer
			numPoints: Nombre de points pour échantillonner chaque orbite
			showLabels: Afficher les étiquettes des satellites
			colors: Liste des couleurs à utiliser pour les orbites
			
		Returns:
			Liste des artistes matplotlib créés
		"""
		artists = []
		
		for i, satellite in enumerate(satellites):
			color = colors[i] if colors and i < len(colors) else self.colormap(i / max(1, len(satellites) - 1))
			
			# Vérifier si cette orbite est dans le cache
			satelliteId = satellite.satelliteId
			
			if satelliteId in self.orbitCache:
				orbitData = self.orbitCache[satelliteId]
				x, y, z = orbitData["points"]
				elements = orbitData["elements"]
			else:
				# Obtenir les éléments orbitaux actuels
				elements = satellite.currentOrbitalElements
				
				# Calculer les points le long de l'orbite
				x, y, z = self._calculateOrbitPoints(elements, numPoints)
				
				# Stocker dans le cache
				self.orbitCache[satelliteId] = {
					"points": (x, y, z),
					"elements": elements
				}
			
			# Tracer l'orbite
			orbit_artists = self._plotOrbit(ax, x, y, z, satellite, color, showLabels)
			artists.extend(orbit_artists)
		
		return artists
	
	def renderPlanarOrbits(self, ax: plt.Axes, satellites: List[Satellite], 
						  numPoints: int = 100, showLabels: bool = False,
						  colors: Optional[List[Any]] = None) -> List[Artist]:
		"""
		Dessine les projections des orbites sur un plan 2D.
		
		Args:
			ax: Axe 2D Matplotlib
			satellites: Liste des satellites à tracer
			numPoints: Nombre de points pour échantillonner chaque orbite
			showLabels: Afficher les étiquettes des satellites
			colors: Liste des couleurs à utiliser pour les orbites
			
		Returns:
			Liste des artistes matplotlib créés
		"""
		artists = []
		
		for i, satellite in enumerate(satellites):
			color = colors[i] if colors and i < len(colors) else self.colormap(i / max(1, len(satellites) - 1))
			
			# Vérifier si cette orbite est dans le cache
			satelliteId = satellite.satelliteId
			
			if satelliteId in self.orbitCache:
				orbitData = self.orbitCache[satelliteId]
				x, y, z = orbitData["points"]
				elements = orbitData["elements"]
			else:
				# Obtenir les éléments orbitaux actuels
				elements = satellite.currentOrbitalElements
				
				# Calculer les points le long de l'orbite
				x, y, z = self._calculateOrbitPoints(elements, numPoints)
				
				# Stocker dans le cache
				self.orbitCache[satelliteId] = {
					"points": (x, y, z),
					"elements": elements
				}
			
			# Tracer la projection de l'orbite (plan XY)
			orbit_line = ax.plot(x, y, color=color, alpha=0.7, lw=1.0)[0]
			artists.append(orbit_line)
			
			# Position actuelle du satellite
			pos = satellite.getPosition()
			satellite_marker = ax.scatter(pos[0], pos[1], color=color, edgecolor='black', s=40, zorder=10)[0]
			artists.append(satellite_marker)
			
			# Étiquette
			if showLabels:
				label = ax.annotate(f"Sat {satelliteId}", (pos[0], pos[1]), 
								   xytext=(5, 5), textcoords='offset points',
								   fontsize=8, color=color)
				artists.append(label)
		
		return artists
	
	def renderOrbitParameters(self, ax: plt.Axes, satellites: List[Satellite], 
							 colors: Optional[List[Any]] = None) -> List[Artist]:
		"""
		Affiche les paramètres orbitaux des satellites sur un graphique.
		
		Args:
			ax: Axe 2D Matplotlib
			satellites: Liste des satellites à représenter
			colors: Liste des couleurs à utiliser
			
		Returns:
			Liste des artistes matplotlib créés
		"""
		artists = []
		
		# Paramètres à afficher
		params = ['SMA', 'Ecc', 'Inc', 'RAAN', 'ArgP', 'TA']
		param_descriptions = {
			'SMA': 'Demi-grand axe (km)',
			'Ecc': 'Excentricité',
			'Inc': 'Inclinaison (°)',
			'RAAN': 'Asc. Node (°)',
			'ArgP': 'Arg. Péri. (°)',
			'TA': 'Anom. Vraie (°)'
		}
		
		# Créer un tableau de données
		data = []
		for i, satellite in enumerate(satellites):
			elements = satellite.currentOrbitalElements
			color = colors[i] if colors and i < len(colors) else self.colormap(i / max(1, len(satellites) - 1))
			
			# Convertir certains paramètres en degrés pour la lisibilité
			row = [
				elements.semimajorAxis,
				elements.eccentricity,
				np.degrees(elements.inclination),
				np.degrees(elements.longitudeOfAscendingNode),
				np.degrees(elements.argumentOfPeriapsis),
				np.degrees(elements.trueAnomaly)
			]
			data.append((satellite.satelliteId, row, color))
		
		# Trier par ID de satellite
		data.sort(key=lambda x: x[0])
		
		# Effacer l'axe existant
		ax.clear()
		ax.set_axis_off()
		
		# Créer une table
		table = ax.table(
			cellText=[[f"Sat #{sat_id}"] + [f"{val:.2f}" for val in row] for sat_id, row, _ in data],
			colLabels=["Satellite"] + params,
			cellColours=[[color] + [color + (0.2,) for _ in params] for _, _, color in data],
			cellLoc='center',
			loc='center'
		)
		
		# Ajuster la taille de la table
		table.scale(1, 1.5)
		table.auto_set_font_size(False)
		table.set_fontsize(8)
		
		# Ajouter la description des paramètres
		description = "\n".join([f"{param}: {desc}" for param, desc in param_descriptions.items()])
		footnote = ax.text(0.5, 0.05, description, fontsize=8, ha='center', transform=ax.transAxes)
		
		artists.append(table)
		artists.append(footnote)
		
		return artists
	
	def clearCache(self) -> None:
		"""
		Efface le cache des orbites calculées.
		"""
		self.orbitCache.clear()
	
	def _calculateOrbitPoints(self, elements: OrbitalElements, numPoints: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
		"""
		Calcule les points d'une orbite à partir des éléments orbitaux.
		
		Args:
			elements: Éléments orbitaux
			numPoints: Nombre de points pour échantillonner l'orbite
			
		Returns:
			Tuple des coordonnées (x, y, z) des points de l'orbite
		"""
		# Échantillonner les anomalies vraies uniformément sur [0, 2π]
		true_anomalies = np.linspace(0, 2 * np.pi, numPoints)
		
		# Sauvegarder l'anomalie vraie actuelle
		current_true_anomaly = elements.trueAnomaly
		
		# Initialiser les tableaux pour les positions
		x = np.zeros(numPoints)
		y = np.zeros(numPoints)
		z = np.zeros(numPoints)
		
		# Calculer chaque point
		for i, true_anomaly in enumerate(true_anomalies):
			# Mettre à jour temporairement l'anomalie vraie
			elements.trueAnomaly = true_anomaly
			
			# Calculer la position
			position, _ = elements.toPosVel()
			x[i], y[i], z[i] = position
		
		# Restaurer l'anomalie vraie originale
		elements.trueAnomaly = current_true_anomaly
		
		return x, y, z
	
	def _plotOrbit(self, ax: Axes3D, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
				 satellite: Satellite, color: Any, showLabels: bool) -> List[Artist]:
		"""
		Dessine une orbite individuelle.
		
		Args:
			ax: Axe 3D Matplotlib
			x, y, z: Coordonnées des points de l'orbite
			satellite: Satellite associé à l'orbite
			color: Couleur de l'orbite
			showLabels: Afficher l'étiquette du satellite
			
		Returns:
			Liste des artistes matplotlib créés
		"""
		artists = []
		
		# Tracer l'orbite avec un dégradé de transparence
		# Le segment où se trouve actuellement le satellite est plus visible
		position = satellite.getPosition()
		
		# Tracer la ligne d'orbite
		orbit_line = ax.plot(x, y, z, color=color, alpha=0.5, lw=1.5)[0]
		artists.append(orbit_line)
		
		# Point représentant le satellite
		satellite_marker = ax.scatter(position[0], position[1], position[2], color=color, edgecolor='black', s=50, zorder=10)[0]
		artists.append(satellite_marker)
		
		# Ajouter une étiquette
		if showLabels:
			label = ax.text(position[0], position[1], position[2], f"Sat {satellite.satelliteId}", 
						   size=8, zorder=11, color=color)
			artists.append(label)
		
		return artists
	
	def generateOrbitDiagram(self, satellite: Satellite, figsize: Tuple[float, float] = (8, 6)) -> Figure:
		"""
		Génère un diagramme détaillé de l'orbite d'un satellite.
		
		Args:
			satellite: Satellite à représenter
			figsize: Taille de la figure
			
		Returns:
			Figure matplotlib
		"""
		fig = plt.figure(figsize=figsize)
		
		# Panneau principal: vue 3D de l'orbite
		ax1 = fig.add_subplot(2, 2, 1, projection='3d')
		ax1.set_title(f"Orbite du satellite #{satellite.satelliteId}")
		
		# Dessiner la Terre
		self._drawEarth(ax1)
		
		# Calculer et dessiner l'orbite
		elements = satellite.currentOrbitalElements
		x, y, z = self._calculateOrbitPoints(elements, 200)
		orbit_color = self.colormap(0.5)
		self._plotOrbit(ax1, x, y, z, satellite, orbit_color, True)
		
		# Configurer la vue 3D
		ax1.set_xlabel('X (km)')
		ax1.set_ylabel('Y (km)')
		ax1.set_zlabel('Z (km)')
		ax1.set_box_aspect([1, 1, 1])
		
		# Limite des axes
		limit = max(np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))) * 1.1
		ax1.set_xlim(-limit, limit)
		ax1.set_ylim(-limit, limit)
		ax1.set_zlim(-limit, limit)
		
		# Panneau 2: Projection XY
		ax2 = fig.add_subplot(2, 2, 2)
		ax2.set_title("Projection XY")
		ax2.plot(x, y, color=orbit_color)
		ax2.scatter(satellite.getPosition()[0], satellite.getPosition()[1], color=orbit_color, edgecolor='black', s=50)
		ax2.set_xlabel('X (km)')
		ax2.set_ylabel('Y (km)')
		ax2.grid(True, alpha=0.3)
		
		# Dessiner le cercle représentant la Terre
		earth_circle = plt.Circle((0, 0), self.earthRadius, color='blue', alpha=0.2)
		ax2.add_artist(earth_circle)
		
		# Limites égales
		ax2.set_aspect('equal')
		ax2.set_xlim(-limit, limit)
		ax2.set_ylim(-limit, limit)
		
		# Panneau 3: Projection XZ
		ax3 = fig.add_subplot(2, 2, 3)
		ax3.set_title("Projection XZ")
		ax3.plot(x, z, color=orbit_color)
		ax3.scatter(satellite.getPosition()[0], satellite.getPosition()[2], color=orbit_color, edgecolor='black', s=50)
		ax3.set_xlabel('X (km)')
		ax3.set_ylabel('Z (km)')
		ax3.grid(True, alpha=0.3)
		
		# Limites égales
		ax3.set_aspect('equal')
		ax3.set_xlim(-limit, limit)
		ax3.set_ylim(-limit, limit)
		
		# Panneau 4: Informations sur l'orbite
		ax4 = fig.add_subplot(2, 2, 4)
		ax4.axis('off')
		
		# Récupérer les éléments orbitaux
		elements = satellite.currentOrbitalElements
		
		# Texte avec les informations
		info_text = (
			f"Paramètres orbitaux:\n"
			f"Demi-grand axe: {elements.semimajorAxis:.1f} km\n"
			f"Excentricité: {elements.eccentricity:.6f}\n"
			f"Inclinaison: {np.degrees(elements.inclination):.2f}°\n"
			f"Long. nœud asc.: {np.degrees(elements.longitudeOfAscendingNode):.2f}°\n"
			f"Arg. périapsis: {np.degrees(elements.argumentOfPeriapsis):.2f}°\n"
			f"Anomalie vraie: {np.degrees(elements.trueAnomaly):.2f}°\n\n"
			f"Paramètres dérivés:\n"
			f"Période: {self._calculateOrbitalPeriod(elements):.2f} min\n"
			f"Périgée: {self._calculatePerigee(elements):.1f} km\n"
			f"Apogée: {self._calculateApogee(elements):.1f} km\n"
			f"Altitude actuelle: {np.linalg.norm(satellite.getPosition()) - self.earthRadius:.1f} km"
		)
		
		ax4.text(0.05, 0.95, info_text, va='top', fontsize=9)
		
		# Informations sur l'état du satellite
		state_text = (
			f"État du satellite:\n"
			f"Batterie: {satellite.state.batteryLevel * 100:.1f}%\n"
			f"Production solaire: {satellite.state.solarPanelOutput:.1f} W\n"
			f"Température: {satellite.state.temperature:.1f} K\n"
			f"Carburant: {satellite.state.fuelRemaining:.2f} kg\n"
			f"Éclipse: {'Oui' if satellite.isEclipsed else 'Non'}"
		)
		
		ax4.text(0.05, 0.35, state_text, va='top', fontsize=9)
		
		# Ajuster la mise en page
		plt.tight_layout()
		
		return fig
	
	def _drawEarth(self, ax: Axes3D) -> None:
		"""
		Dessine une représentation simple de la Terre sur un axe 3D.
		
		Args:
			ax: Axe 3D Matplotlib
		"""
		# Créer une sphère wireframe
		u = np.linspace(0, 2 * np.pi, 20)
		v = np.linspace(0, np.pi, 20)
		x = self.earthRadius * np.outer(np.cos(u), np.sin(v))
		y = self.earthRadius * np.outer(np.sin(u), np.sin(v))
		z = self.earthRadius * np.outer(np.ones(np.size(u)), np.cos(v))
		
		# Dessiner la Terre semi-transparente
		ax.plot_surface(x, y, z, color='blue', alpha=0.2)
		
		# Ajouter des lignes pour l'équateur et les méridiens principaux
		# Équateur
		theta = np.linspace(0, 2 * np.pi, 100)
		x_eq = self.earthRadius * np.cos(theta)
		y_eq = self.earthRadius * np.sin(theta)
		z_eq = np.zeros_like(theta)
		ax.plot(x_eq, y_eq, z_eq, 'k-', alpha=0.3)
		
		# Méridien de Greenwich (0°)
		phi = np.linspace(-np.pi/2, np.pi/2, 100)
		x_m0 = self.earthRadius * np.zeros_like(phi)
		y_m0 = self.earthRadius * np.cos(phi)
		z_m0 = self.earthRadius * np.sin(phi)
		ax.plot(x_m0, y_m0, z_m0, 'k-', alpha=0.3)
		
		# Méridien 90° Est
		x_m90 = self.earthRadius * np.cos(phi)
		y_m90 = self.earthRadius * np.zeros_like(phi)
		z_m90 = self.earthRadius * np.sin(phi)
		ax.plot(x_m90, y_m90, z_m90, 'k-', alpha=0.3)
	
	def _calculateOrbitalPeriod(self, elements: OrbitalElements) -> float:
		"""
		Calcule la période orbitale en minutes.
		
		Args:
			elements: Éléments orbitaux
			
		Returns:
			Période orbitale en minutes
		"""
		# Période = 2π * sqrt(a³/μ)
		a_meters = elements.semimajorAxis * 1000  # Convertir en mètres
		mu = 3.986004418e14  # m³/s²
		
		period_seconds = 2 * np.pi * np.sqrt(a_meters**3 / mu)
		return period_seconds / 60  # Convertir en minutes
	
	def _calculatePerigee(self, elements: OrbitalElements) -> float:
		"""
		Calcule l'altitude du périgée en km.
		
		Args:
			elements: Éléments orbitaux
			
		Returns:
			Altitude du périgée en km
		"""
		return elements.semimajorAxis * (1 - elements.eccentricity) - self.earthRadius
	
	def _calculateApogee(self, elements: OrbitalElements) -> float:
		"""
		Calcule l'altitude de l'apogée en km.
		
		Args:
			elements: Éléments orbitaux
			
		Returns:
			Altitude de l'apogée en km
		"""
		return elements.semimajorAxis * (1 + elements.eccentricity) - self.earthRadius