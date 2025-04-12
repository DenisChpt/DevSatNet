import pygame
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
from typing import Dict, List, Tuple, Any, Optional, Set
import math
import os
import sys

# Ajouter le répertoire parent au chemin d'importation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import get_logger
from utils.timer import Timer
from core.environment.marine_world import MarineWorld
from core.environment.terrain import Terrain
from core.environment.resources import FoodResource
from visualization.renderer import RenderMode


class EnvironmentRenderer:
	"""
	Classe responsable du rendu de l'environnement marin.
	Gère la représentation visuelle du terrain, de l'eau, des ressources
	et des effets visuels environnementaux.
	"""
	
	def __init__(self) -> None:
		"""
		Initialise le renderer d'environnement.
		"""
		self.logger = get_logger()
		self.timer = Timer("EnvironmentRenderer")
		self.world = None
		self.textures: Dict[str, int] = {}
		self.display_lists: Dict[str, int] = {}
		
		# Paramètres de rendu
		self.water_resolution = 32  # Résolution de la grille d'eau
		self.terrain_resolution = 64  # Résolution de la grille de terrain
		self.show_water = True
		self.show_terrain = True
		self.show_resources = True
		self.show_zones = True
		self.show_grid = False
		self.show_currents = False
		
		# Effet d'ondulation de l'eau
		self.wave_time = 0.0
		self.wave_speed = 1.0
		self.wave_height = 0.5
		
		# Paramètres visuels
		self.water_color = (0.2, 0.4, 0.8, 0.6)  # RGBA
		self.grid_color = (0.8, 0.8, 0.8, 0.3)
		self.terrain_colormap = [
			(0.8, 0.7, 0.5),    # Sable (peu profond)
			(0.6, 0.5, 0.4),    # Vase
			(0.5, 0.5, 0.5),    # Roche
			(0.9, 0.9, 0.6),    # Gravier
			(0.3, 0.5, 0.4)     # Algues
		]
		
		# Cache pour les ressources (évite de recalculer les mêmes objets)
		self.resource_cache = {}
		
		# Chargement des textures et initialisation des ressources
		self._init_resources()
	
	def _init_resources(self) -> None:
		"""
		Initialise les ressources nécessaires au rendu (textures, modèles, etc.).
		"""
		try:
			# Créer des textures pour les différents éléments environnementaux
			self._create_water_texture()
			self._create_terrain_textures()
			self._create_resource_textures()
			
			# Créer des listes d'affichage pour les primitives communes
			self._create_water_surface_display_list()
			self._create_terrain_display_list()
			self._create_grid_display_list()
			
			self.logger.info("Ressources de l'EnvironmentRenderer initialisées", module="visualization")
		except Exception as e:
			self.logger.error(f"Erreur lors de l'initialisation des ressources: {str(e)}", 
							  module="visualization", exc_info=True)
	
	def _create_water_texture(self) -> None:
		"""Crée une texture d'eau."""
		texture_id = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
		
		# Texture d'eau simple
		width, height = 256, 256
		texture_data = np.zeros((height, width, 4), dtype=np.uint8)
		
		for y in range(height):
			for x in range(width):
				# Motif de vagues
				value = int(128 + 64 * math.sin(x/16.0) * math.sin(y/16.0))
				# Couleur bleutée avec transparence
				texture_data[y, x] = [50, 100, value, 180]
		
		# Charger la texture
		gl.glTexImage2D(
			gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
			gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texture_data
		)
		
		# Configurer les paramètres de texture
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
		
		self.textures["water"] = texture_id
	
	def _create_terrain_textures(self) -> None:
		"""Crée les textures pour les différents types de terrain."""
		terrain_types = ["sand", "mud", "rock", "gravel", "coral"]
		
		for terrain_type in terrain_types:
			texture_id = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
			
			# Texture spécifique au type de terrain
			width, height = 128, 128
			texture_data = np.zeros((height, width, 3), dtype=np.uint8)
			
			if terrain_type == "sand":
				# Texture de sable: jaune clair avec léger bruit
				for y in range(height):
					for x in range(width):
						noise = int(15 * (np.random.random() - 0.5))
						texture_data[y, x] = [220 + noise, 200 + noise, 160 + noise]
			
			elif terrain_type == "mud":
				# Texture de vase: brune
				for y in range(height):
					for x in range(width):
						noise = int(10 * (np.random.random() - 0.5))
						texture_data[y, x] = [120 + noise, 100 + noise, 80 + noise]
			
			elif terrain_type == "rock":
				# Texture de roche: gris avec variation
				for y in range(height):
					for x in range(width):
						# Bruit de Perlin simplifié
						value = int(50 * (math.sin(x/10.0) * math.sin(y/8.0) + math.sin(x/5.0 + 2) * math.sin(y/7.0)))
						base = 150
						texture_data[y, x] = [base + value, base + value, base + value]
			
			elif terrain_type == "gravel":
				# Texture de gravier: petites pierres
				for y in range(height):
					for x in range(width):
						# Quadrillage irrégulier
						val = (int(x/4) % 2 + int(y/4) % 2 + np.random.randint(0, 2)) % 3
						base = 160 if val > 0 else 140
						texture_data[y, x] = [base, base, base]
			
			elif terrain_type == "coral":
				# Texture de corail: colorée
				for y in range(height):
					for x in range(width):
						# Combiner différentes fréquences
						r_val = int(200 + 55 * math.sin(x/20.0) * math.sin(y/20.0))
						g_val = int(100 + 50 * math.sin(x/15.0 + 1) * math.sin(y/15.0))
						b_val = int(100 + 155 * math.sin(x/10.0 + 2) * math.sin(y/30.0))
						texture_data[y, x] = [r_val, g_val, b_val]
			
			# Charger la texture
			gl.glTexImage2D(
				gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
				gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_data
			)
			
			# Configurer les paramètres de texture
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
			
			self.textures[terrain_type] = texture_id
	
	def _create_resource_textures(self) -> None:
		"""Crée les textures pour les différents types de ressources."""
		resource_types = ["algae", "plankton", "small_fish", "detritus"]
		
		for resource_type in resource_types:
			texture_id = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
			
			# Texture spécifique au type de ressource
			width, height = 64, 64
			texture_data = np.zeros((height, width, 4), dtype=np.uint8)  # RGBA
			
			if resource_type == "algae":
				# Texture d'algues: vert
				for y in range(height):
					for x in range(width):
						# Motif filamenteux
						value = int(200 * abs(math.sin(x/8.0 + y/10.0)))
						alpha = int(200 * abs(math.sin(x/5.0 + y/7.0)))
						texture_data[y, x] = [20, 150 + value % 100, 30, alpha]
			
			elif resource_type == "plankton":
				# Texture de plancton: petits points lumineux
				for y in range(height):
					for x in range(width):
						# Points aléatoires
						if np.random.random() < 0.1:
							brightness = np.random.randint(200, 256)
							texture_data[y, x] = [brightness, brightness, brightness, 200]
						else:
							texture_data[y, x] = [50, 100, 150, 100]
			
			elif resource_type == "small_fish":
				# Texture de petit poisson: écailles
				for y in range(height):
					for x in range(width):
						# Motif d'écailles
						dist = math.sin(x * 0.5) * 10 + math.sin(y * 0.5) * 10
						color_val = int(128 + dist) % 255
						texture_data[y, x] = [color_val, color_val, 200, 200]
			
			elif resource_type == "detritus":
				# Texture de détritus: particules brunes
				for y in range(height):
					for x in range(width):
						# Particules aléatoires
						if np.random.random() < 0.2:
							r = np.random.randint(100, 150)
							g = np.random.randint(80, 120)
							b = np.random.randint(40, 80)
							texture_data[y, x] = [r, g, b, 180]
						else:
							texture_data[y, x] = [0, 0, 0, 0]
			
			# Charger la texture
			gl.glTexImage2D(
				gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
				gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texture_data
			)
			
			# Configurer les paramètres de texture
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
			
			self.textures[resource_type] = texture_id
	
	def _create_water_surface_display_list(self) -> None:
		"""
		Crée une liste d'affichage pour la surface de l'eau.
		"""
		list_id = gl.glGenLists(1)
		
		gl.glNewList(list_id, gl.GL_COMPILE)
		
		# Dessiner une grille représentant la surface de l'eau
		gl.glBegin(gl.GL_QUADS)
		
		# Une grille simple pour la surface
		resolution = 20
		size = 1.0
		
		for z in range(resolution):
			for x in range(resolution):
				# Coordonnées des sommets
				x1 = x * size
				x2 = (x + 1) * size
				z1 = z * size
				z2 = (z + 1) * size
				
				# Coordonnées de texture
				tx1 = x / resolution
				tx2 = (x + 1) / resolution
				tz1 = z / resolution
				tz2 = (z + 1) / resolution
				
				# Normale dirigée vers le haut
				gl.glNormal3f(0.0, 1.0, 0.0)
				
				# Premier triangle
				gl.glTexCoord2f(tx1, tz1)
				gl.glVertex3f(x1, 0, z1)
				
				gl.glTexCoord2f(tx2, tz1)
				gl.glVertex3f(x2, 0, z1)
				
				gl.glTexCoord2f(tx2, tz2)
				gl.glVertex3f(x2, 0, z2)
				
				gl.glTexCoord2f(tx1, tz2)
				gl.glVertex3f(x1, 0, z2)
		
		gl.glEnd()
		
		gl.glEndList()
		
		self.display_lists["water_surface"] = list_id
	
	def _create_terrain_display_list(self) -> None:
		"""
		Crée une liste d'affichage pour le terrain sous-marin.
		"""
		# Cette liste sera générée dynamiquement lorsque le monde sera défini
		pass
	
	def _create_grid_display_list(self) -> None:
		"""
		Crée une liste d'affichage pour la grille de référence.
		"""
		list_id = gl.glGenLists(1)
		
		gl.glNewList(list_id, gl.GL_COMPILE)
		
		gl.glBegin(gl.GL_LINES)
		
		# Dessiner une grille simple
		size = 100.0
		step = 10.0
		
		# Lignes le long de l'axe X
		for i in range(int(-size), int(size) + 1, int(step)):
			gl.glVertex3f(i, 0, -size)
			gl.glVertex3f(i, 0, size)
		
		# Lignes le long de l'axe Z
		for i in range(int(-size), int(size) + 1, int(step)):
			gl.glVertex3f(-size, 0, i)
			gl.glVertex3f(size, 0, i)
		
		gl.glEnd()
		
		gl.glEndList()
		
		self.display_lists["grid"] = list_id
	
	def set_world(self, world: MarineWorld) -> None:
		"""
		Définit le monde marin à rendre.
		
		Args:
			world: Instance du monde marin
		"""
		self.world = world
		
		# Générer les display lists spécifiques au monde
		self._generate_terrain_display_list()
		
		self.logger.info("Monde défini pour l'EnvironmentRenderer", module="visualization")
	
	def _generate_terrain_display_list(self) -> None:
		"""
		Génère une liste d'affichage pour le terrain sous-marin spécifique au monde actuel.
		"""
		if not self.world or not self.world.terrain:
			return
			
		# Supprimer l'ancienne liste si elle existe
		if "terrain" in self.display_lists:
			gl.glDeleteLists(self.display_lists["terrain"], 1)
		
		# Créer une nouvelle liste
		list_id = gl.glGenLists(1)
		
		gl.glNewList(list_id, gl.GL_COMPILE)
		
		# Récupérer les dimensions du terrain
		terrain = self.world.terrain
		world_size = self.world.size
		resolution = terrain.resolution
		
		# Facteurs d'échelle
		scale_x = world_size[0] / resolution[0]
		scale_z = world_size[2] / resolution[2]
		
		# Dessiner le terrain en triangles
		gl.glBegin(gl.GL_TRIANGLES)
		
		for z in range(resolution[2] - 1):
			for x in range(resolution[0] - 1):
				# Obtenir les élévations aux quatre coins
				y00 = terrain.elevationMap[x, z]
				y10 = terrain.elevationMap[x + 1, z]
				y11 = terrain.elevationMap[x + 1, z + 1]
				y01 = terrain.elevationMap[x, z + 1]
				
				# Obtenir les types de terrain
				type00 = terrain.terrainTypeMap[x, z]
				type10 = terrain.terrainTypeMap[x + 1, z]
				type11 = terrain.terrainTypeMap[x + 1, z + 1]
				type01 = terrain.terrainTypeMap[x, z + 1]
				
				# Coordonnées des sommets
				x0 = x * scale_x
				x1 = (x + 1) * scale_x
				z0 = z * scale_z
				z1 = (z + 1) * scale_z
				
				# Coordonnées de texture
				tx0 = x / (resolution[0] - 1) * 5.0  # Répéter la texture
				tx1 = (x + 1) / (resolution[0] - 1) * 5.0
				tz0 = z / (resolution[2] - 1) * 5.0
				tz1 = (z + 1) / (resolution[2] - 1) * 5.0
				
				# Calculer les normales pour chaque face
				
				# Première face (00-10-11)
				v1 = np.array([x1 - x0, y10 - y00, 0])
				v2 = np.array([x1 - x0, y11 - y00, z1 - z0])
				normal1 = np.cross(v1, v2)
				normal1 = normal1 / np.linalg.norm(normal1)
				
				# Deuxième face (00-11-01)
				v1 = np.array([x1 - x0, y11 - y00, z1 - z0])
				v2 = np.array([0, y01 - y00, z1 - z0])
				normal2 = np.cross(v1, v2)
				normal2 = normal2 / np.linalg.norm(normal2)
				
				# Couleurs basées sur le type de terrain
				colors = [self.terrain_colormap[min(type00, len(self.terrain_colormap) - 1)],
						  self.terrain_colormap[min(type10, len(self.terrain_colormap) - 1)],
						  self.terrain_colormap[min(type11, len(self.terrain_colormap) - 1)],
						  self.terrain_colormap[min(type01, len(self.terrain_colormap) - 1)]]
				
				# Premier triangle
				gl.glNormal3f(normal1[0], normal1[1], normal1[2])
				
				gl.glColor3f(*colors[0])
				gl.glTexCoord2f(tx0, tz0)
				gl.glVertex3f(x0, y00, z0)
				
				gl.glColor3f(*colors[1])
				gl.glTexCoord2f(tx1, tz0)
				gl.glVertex3f(x1, y10, z0)
				
				gl.glColor3f(*colors[2])
				gl.glTexCoord2f(tx1, tz1)
				gl.glVertex3f(x1, y11, z1)
				
				# Deuxième triangle
				gl.glNormal3f(normal2[0], normal2[1], normal2[2])
				
				gl.glColor3f(*colors[0])
				gl.glTexCoord2f(tx0, tz0)
				gl.glVertex3f(x0, y00, z0)
				
				gl.glColor3f(*colors[2])
				gl.glTexCoord2f(tx1, tz1)
				gl.glVertex3f(x1, y11, z1)
				
				gl.glColor3f(*colors[3])
				gl.glTexCoord2f(tx0, tz1)
				gl.glVertex3f(x0, y01, z1)
		
		gl.glEnd()
		
		gl.glEndList()
		
		self.display_lists["terrain"] = list_id
	
	def update(self, delta_time: float) -> None:
		"""
		Met à jour les animations et effets visuels.
		
		Args:
			delta_time: Temps écoulé depuis la dernière mise à jour
		"""
		# Mettre à jour le temps pour l'animation des vagues
		self.wave_time += delta_time * self.wave_speed
		
		# Si nécessaire, mettre à jour d'autres effets d'animation
	
	def render(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu de l'environnement marin.
		
		Args:
			render_mode: Mode de rendu à utiliser
		"""
		if not self.world:
			return
			
		# Configurer le mode de rendu
		if render_mode == RenderMode.WIREFRAME:
			gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
			gl.glDisable(gl.GL_LIGHTING)
			gl.glDisable(gl.GL_TEXTURE_2D)
		elif render_mode == RenderMode.SOLID:
			gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
			gl.glDisable(gl.GL_LIGHTING)
			gl.glDisable(gl.GL_TEXTURE_2D)
		elif render_mode == RenderMode.TEXTURED:
			gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
			gl.glEnable(gl.GL_LIGHTING)
			gl.glEnable(gl.GL_TEXTURE_2D)
		else:  # SHADED
			gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
			gl.glEnable(gl.GL_LIGHTING)
			gl.glDisable(gl.GL_TEXTURE_2D)
		
		# Rendre le terrain si activé
		if self.show_terrain:
			self._render_terrain(render_mode)
			
		# Rendre l'eau si activée
		if self.show_water:
			self._render_water(render_mode)
		
		# Rendre les ressources si activées
		if self.show_resources:
			self._render_resources(render_mode)
		
		# Rendre les zones environnementales si activées
		if self.show_zones:
			self._render_zones(render_mode)
		
		# Rendre la grille de référence si activée
		if self.show_grid:
			self._render_grid()
			
		# Rendre les courants si activés
		if self.show_currents:
			self._render_currents()
	
	def _render_terrain(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu du terrain sous-marin.
		
		Args:
			render_mode: Mode de rendu à utiliser
		"""
		# Configurer le matériau pour le terrain
		if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.1, 0.1, 0.1, 1.0])
			gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 10.0)
			
			# Appliquer une texture si en mode texturé
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["sand"])
		
		# Appeler la liste d'affichage du terrain
		if "terrain" in self.display_lists:
			gl.glCallList(self.display_lists["terrain"])
		
		# Rendre les obstacles
		self._render_obstacles(render_mode)
		
		# Rendre les structures spéciales
		self._render_structures(render_mode)
	
	def _render_obstacles(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu des obstacles sur le terrain (rochers, formations, etc.).
		
		Args:
			render_mode: Mode de rendu à utiliser
		"""
		if not self.world or not self.world.terrain:
			return
			
		# Récupérer les obstacles du terrain
		obstacles = self.world.terrain.obstacles
		
		for obstacle in obstacles:
			# Récupérer les propriétés de l'obstacle
			position = obstacle["position"]
			size = obstacle["size"]
			obstacle_type = obstacle["type"]
			orientation = obstacle.get("orientation", 0.0)
			
			# Sélectionner la couleur et la texture en fonction du type
			if obstacle_type == "rock" or obstacle_type == "rock_formation":
				color = (0.6, 0.6, 0.6)  # Gris
				texture_name = "rock"
			elif obstacle_type == "coral_formation":
				color = (0.9, 0.6, 0.6)  # Rose corail
				texture_name = "coral"
			elif obstacle_type == "deep_sea_vent":
				color = (0.3, 0.3, 0.3)  # Gris foncé
				texture_name = "rock"
			else:
				color = (0.7, 0.7, 0.7)  # Gris par défaut
				texture_name = "rock"
			
			# Sauvegarder l'état
			gl.glPushMatrix()
			
			# Positionner l'obstacle
			gl.glTranslatef(position[0], position[1], position[2])
			
			# Appliquer la rotation
			gl.glRotatef(orientation * 180.0 / np.pi, 0, 1, 0)
			
			# Configurer le matériau
			if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [color[0], color[1], color[2], 1.0])
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
				gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 20.0)
				
				# Appliquer une texture si en mode texturé
				if render_mode == RenderMode.TEXTURED and texture_name in self.textures:
					gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[texture_name])
			else:
				gl.glColor3f(*color)
			
			# Dessiner l'obstacle en fonction de son type
			if obstacle_type == "rock":
				# Simple rocher (sphère déformée)
				self._render_rock(size)
			elif obstacle_type == "rock_formation":
				# Formation rocheuse (plusieurs rochers)
				self._render_rock_formation(size)
			elif obstacle_type == "coral_formation":
				# Formation de corail
				self._render_coral_formation(size)
			elif obstacle_type == "deep_sea_vent":
				# Cheminée hydrothermale
				self._render_deep_sea_vent(size)
			
			# Restaurer l'état
			gl.glPopMatrix()
	
	def _render_rock(self, size: float) -> None:
		"""
		Dessine un rocher simple.
		
		Args:
			size: Taille du rocher
		"""
		# Utiliser une sphère légèrement déformée
		quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(quad, gl.GL_TRUE)
		glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
		
		# Déformer légèrement la sphère pour un aspect plus naturel
		gl.glScalef(size * (0.8 + 0.4 * np.random.random()), 
				   size * (0.7 + 0.3 * np.random.random()), 
				   size * (0.9 + 0.2 * np.random.random()))
		
		glu.gluSphere(quad, 1.0, 16, 16)
		glu.gluDeleteQuadric(quad)
	
	def _render_rock_formation(self, size: float) -> None:
		"""
		Dessine une formation rocheuse composée de plusieurs rochers.
		
		Args:
			size: Taille globale de la formation
		"""
		# Nombre de rochers en fonction de la taille
		num_rocks = int(3 + size / 2)
		
		for i in range(num_rocks):
			# Position aléatoire autour du centre
			x = (np.random.random() - 0.5) * size * 0.8
			y = (np.random.random() * 0.5) * size * 0.4
			z = (np.random.random() - 0.5) * size * 0.8
			
			# Taille du rocher
			rock_size = size * (0.3 + 0.7 * np.random.random()) * 0.5
			
			gl.glPushMatrix()
			gl.glTranslatef(x, y, z)
			self._render_rock(rock_size)
			gl.glPopMatrix()
	
	def _render_coral_formation(self, size: float) -> None:
		"""
		Dessine une formation de corail.
		
		Args:
			size: Taille globale de la formation
		"""
		# Base de la formation (rocher)
		gl.glPushMatrix()
		gl.glScalef(size * 0.8, size * 0.3, size * 0.8)
		
		quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(quad, gl.GL_TRUE)
		glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
		glu.gluSphere(quad, 1.0, 16, 16)
		glu.gluDeleteQuadric(quad)
		gl.glPopMatrix()
		
		# Branches de corail
		num_branches = int(5 + size * 2)
		
		for i in range(num_branches):
			# Position sur la base
			angle = np.random.random() * 2 * np.pi
			radius = np.random.random() * size * 0.7
			x = np.cos(angle) * radius
			z = np.sin(angle) * radius
			
			# Taille et angle de la branche
			branch_height = size * (0.5 + 0.5 * np.random.random())
			branch_width = size * 0.1 * (0.5 + 0.5 * np.random.random())
			branch_angle = 30 * (np.random.random() - 0.5)
			
			gl.glPushMatrix()
			gl.glTranslatef(x, 0, z)
			gl.glRotatef(branch_angle, 1, 0, 1)
			
			# Dessiner une forme de branche de corail (cylindre avec sphère au bout)
			gl.glPushMatrix()
			gl.glRotatef(90, 1, 0, 0)  # Orienter le cylindre vers le haut
			
			# Cylindre pour la branche
			branch_quad = glu.gluNewQuadric()
			glu.gluQuadricTexture(branch_quad, gl.GL_TRUE)
			glu.gluQuadricNormals(branch_quad, glu.GLU_SMOOTH)
			glu.gluCylinder(branch_quad, branch_width, branch_width * 0.7, branch_height, 8, 4)
			glu.gluDeleteQuadric(branch_quad)
			gl.glPopMatrix()
			
			# Sphère au bout de la branche
			gl.glTranslatef(0, branch_height, 0)
			gl.glScalef(branch_width * 1.5, branch_width * 1.5, branch_width * 1.5)
			
			tip_quad = glu.gluNewQuadric()
			glu.gluQuadricTexture(tip_quad, gl.GL_TRUE)
			glu.gluQuadricNormals(tip_quad, glu.GLU_SMOOTH)
			glu.gluSphere(tip_quad, 1.0, 8, 8)
			glu.gluDeleteQuadric(tip_quad)
			
			gl.glPopMatrix()
	
	def _render_deep_sea_vent(self, size: float) -> None:
		"""
		Dessine une cheminée hydrothermale (deep sea vent).
		
		Args:
			size: Taille globale de la cheminée
		"""
		# Base large
		gl.glPushMatrix()
		gl.glRotatef(90, 1, 0, 0)  # Orienter vers le haut
		
		base_quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(base_quad, gl.GL_TRUE)
		glu.gluQuadricNormals(base_quad, glu.GLU_SMOOTH)
		glu.gluCylinder(base_quad, size * 0.8, size * 0.5, size * 0.4, 16, 4)
		glu.gluDeleteQuadric(base_quad)
		gl.glPopMatrix()
		
		# Partie centrale
		gl.glPushMatrix()
		gl.glTranslatef(0, size * 0.4, 0)
		gl.glRotatef(90, 1, 0, 0)
		
		middle_quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(middle_quad, gl.GL_TRUE)
		glu.gluQuadricNormals(middle_quad, glu.GLU_SMOOTH)
		glu.gluCylinder(middle_quad, size * 0.5, size * 0.3, size * 0.6, 16, 4)
		glu.gluDeleteQuadric(middle_quad)
		gl.glPopMatrix()
		
		# Cheminée supérieure
		gl.glPushMatrix()
		gl.glTranslatef(0, size * 1.0, 0)
		gl.glRotatef(90, 1, 0, 0)
		
		top_quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(top_quad, gl.GL_TRUE)
		glu.gluQuadricNormals(top_quad, glu.GLU_SMOOTH)
		glu.gluCylinder(top_quad, size * 0.3, size * 0.2, size * 0.4, 16, 4)
		glu.gluDeleteQuadric(top_quad)
		gl.glPopMatrix()
		
		# Ouverture de la cheminée
		gl.glPushMatrix()
		gl.glTranslatef(0, size * 1.4, 0)
		gl.glRotatef(90, 1, 0, 0)
		
		opening_quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(opening_quad, gl.GL_TRUE)
		glu.gluQuadricNormals(opening_quad, glu.GLU_SMOOTH)
		glu.gluDisk(opening_quad, 0, size * 0.2, 16, 4)
		glu.gluDeleteQuadric(opening_quad)
		gl.glPopMatrix()
	
	def _render_structures(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu des structures spéciales (récifs, épaves, etc.).
		
		Args:
			render_mode: Mode de rendu à utiliser
		"""
		if not self.world or not self.world.terrain:
			return
			
		# Récupérer les structures du terrain
		structures = self.world.terrain.structures
		
		for structure in structures:
			# Récupérer les propriétés de la structure
			position = structure["position"]
			size = structure["size"]
			structure_type = structure["type"]
			orientation = structure.get("orientation", 0.0)
			properties = structure.get("properties", {})
			
			# Sauvegarder l'état
			gl.glPushMatrix()
			
			# Positionner la structure
			gl.glTranslatef(position[0], position[1], position[2])
			
			# Appliquer la rotation
			gl.glRotatef(orientation * 180.0 / np.pi, 0, 1, 0)
			
			# Sélectionner la texture et la couleur en fonction du type
			if structure_type == "reef":
				texture_name = "coral"
				color = (0.9, 0.7, 0.7)
				self._render_reef(size, properties)
			elif structure_type == "shipwreck":
				texture_name = "rock"  # Texture provisoire
				color = (0.6, 0.5, 0.4)
				self._render_shipwreck(size, properties)
			elif structure_type == "underwater_cave":
				texture_name = "rock"
				color = (0.5, 0.5, 0.6)
				self._render_underwater_cave(size, properties)
			elif structure_type == "kelp_forest":
				texture_name = "algae"
				color = (0.3, 0.7, 0.4)
				self._render_kelp_forest(size, properties)
			elif structure_type == "abyss":
				texture_name = "rock"
				color = (0.2, 0.2, 0.3)
				self._render_abyss(size, properties)
			else:
				texture_name = "rock"
				color = (0.7, 0.7, 0.7)
			
			# Configurer le matériau
			if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [color[0], color[1], color[2], 1.0])
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
				gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 20.0)
				
				# Appliquer une texture si en mode texturé
				if render_mode == RenderMode.TEXTURED and texture_name in self.textures:
					gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[texture_name])
			else:
				gl.glColor3f(*color)
			
			# Restaurer l'état
			gl.glPopMatrix()
	
	def _render_reef(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Dessine un récif corallien.
		
		Args:
			size: Taille globale du récif
			properties: Propriétés spécifiques du récif
		"""
		# Extraire les propriétés
		coral_coverage = properties.get("coral_coverage", 0.7)
		biodiversity = properties.get("biodiversity", 0.8)
		
		# Base du récif (amas de roches)
		self._render_rock_formation(size)
		
		# Formations de corail
		num_corals = int(size * coral_coverage * 3)
		
		for i in range(num_corals):
			# Position aléatoire sur le récif
			angle = np.random.random() * 2 * np.pi
			radius = np.random.random() * size * 0.8
			x = np.cos(angle) * radius
			z = np.sin(angle) * radius
			y = size * 0.2 * np.random.random()
			
			# Taille variée en fonction de la biodiversité
			coral_size = size * 0.3 * (0.5 + 0.5 * biodiversity) * (0.5 + 0.5 * np.random.random())
			
			gl.glPushMatrix()
			gl.glTranslatef(x, y, z)
			self._render_coral_formation(coral_size)
			gl.glPopMatrix()
	
	def _render_shipwreck(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Dessine une épave de navire.
		
		Args:
			size: Taille globale de l'épave
			properties: Propriétés spécifiques de l'épave
		"""
		# Extraire les propriétés
		decay = properties.get("decay", 0.5)
		
		# Corps principal du navire
		gl.glPushMatrix()
		gl.glScalef(size * 0.3, size * 0.15, size)
		
		# Déformer la coque en fonction de l'état de dégradation
		if decay > 0.7:
			# Très dégradé - coque brisée
			gl.glRotatef(20 * (np.random.random() - 0.5), 0, 0, 1)
			gl.glTranslatef(0, -0.2 * decay, 0)
		
		# Dessiner la coque (boîte)
		gl.glBegin(gl.GL_QUADS)
		
		# Fond
		gl.glNormal3f(0, -1, 0)
		gl.glTexCoord2f(0, 0)
		gl.glVertex3f(-1, -1, -1)
		gl.glTexCoord2f(1, 0)
		gl.glVertex3f(1, -1, -1)
		gl.glTexCoord2f(1, 1)
		gl.glVertex3f(1, -1, 1)
		gl.glTexCoord2f(0, 1)
		gl.glVertex3f(-1, -1, 1)
		
		# Côtés
		gl.glNormal3f(-1, 0, 0)
		gl.glTexCoord2f(0, 0)
		gl.glVertex3f(-1, -1, -1)
		gl.glTexCoord2f(1, 0)
		gl.glVertex3f(-1, -1, 1)
		gl.glTexCoord2f(1, 1)
		gl.glVertex3f(-1, 1, 1)
		gl.glTexCoord2f(0, 1)
		gl.glVertex3f(-1, 1, -1)
		
		gl.glNormal3f(1, 0, 0)
		gl.glTexCoord2f(0, 0)
		gl.glVertex3f(1, -1, -1)
		gl.glTexCoord2f(1, 0)
		gl.glVertex3f(1, 1, -1)
		gl.glTexCoord2f(1, 1)
		gl.glVertex3f(1, 1, 1)
		gl.glTexCoord2f(0, 1)
		gl.glVertex3f(1, -1, 1)
		
		gl.glNormal3f(0, 0, -1)
		gl.glTexCoord2f(0, 0)
		gl.glVertex3f(-1, -1, -1)
		gl.glTexCoord2f(1, 0)
		gl.glVertex3f(-1, 1, -1)
		gl.glTexCoord2f(1, 1)
		gl.glVertex3f(1, 1, -1)
		gl.glTexCoord2f(0, 1)
		gl.glVertex3f(1, -1, -1)
		
		gl.glNormal3f(0, 0, 1)
		gl.glTexCoord2f(0, 0)
		gl.glVertex3f(-1, -1, 1)
		gl.glTexCoord2f(1, 0)
		gl.glVertex3f(1, -1, 1)
		gl.glTexCoord2f(1, 1)
		gl.glVertex3f(1, 1, 1)
		gl.glTexCoord2f(0, 1)
		gl.glVertex3f(-1, 1, 1)
		
		# Pont supérieur (en fonction de la dégradation)
		if decay < 0.8:
			gl.glNormal3f(0, 1, 0)
			gl.glTexCoord2f(0, 0)
			gl.glVertex3f(-1, 1, -1)
			gl.glTexCoord2f(1, 0)
			gl.glVertex3f(-1, 1, 1)
			gl.glTexCoord2f(1, 1)
			gl.glVertex3f(1, 1, 1)
			gl.glTexCoord2f(0, 1)
			gl.glVertex3f(1, 1, -1)
		
		gl.glEnd()
		gl.glPopMatrix()
		
		# Structure supérieure du navire (cabine, mât, etc.)
		if decay < 0.6:
			# Cabine
			gl.glPushMatrix()
			gl.glTranslatef(0, size * 0.15, -size * 0.2)
			gl.glScalef(size * 0.2, size * 0.1, size * 0.3)
			
			gl.glBegin(gl.GL_QUADS)
			# Faces de la cabine
			gl.glNormal3f(0, 0, -1)
			gl.glTexCoord2f(0, 0)
			gl.glVertex3f(-1, 0, -1)
			gl.glTexCoord2f(1, 0)
			gl.glVertex3f(1, 0, -1)
			gl.glTexCoord2f(1, 1)
			gl.glVertex3f(1, 1, -1)
			gl.glTexCoord2f(0, 1)
			gl.glVertex3f(-1, 1, -1)
			
			gl.glNormal3f(0, 0, 1)
			gl.glTexCoord2f(0, 0)
			gl.glVertex3f(-1, 0, 1)
			gl.glTexCoord2f(1, 0)
			gl.glVertex3f(1, 0, 1)
			gl.glTexCoord2f(1, 1)
			gl.glVertex3f(1, 1, 1)
			gl.glTexCoord2f(0, 1)
			gl.glVertex3f(-1, 1, 1)
			
			gl.glNormal3f(-1, 0, 0)
			gl.glTexCoord2f(0, 0)
			gl.glVertex3f(-1, 0, -1)
			gl.glTexCoord2f(1, 0)
			gl.glVertex3f(-1, 0, 1)
			gl.glTexCoord2f(1, 1)
			gl.glVertex3f(-1, 1, 1)
			gl.glTexCoord2f(0, 1)
			gl.glVertex3f(-1, 1, -1)
			
			gl.glNormal3f(1, 0, 0)
			gl.glTexCoord2f(0, 0)
			gl.glVertex3f(1, 0, -1)
			gl.glTexCoord2f(1, 0)
			gl.glVertex3f(1, 0, 1)
			gl.glTexCoord2f(1, 1)
			gl.glVertex3f(1, 1, 1)
			gl.glTexCoord2f(0, 1)
			gl.glVertex3f(1, 1, -1)
			
			gl.glNormal3f(0, 1, 0)
			gl.glTexCoord2f(0, 0)
			gl.glVertex3f(-1, 1, -1)
			gl.glTexCoord2f(1, 0)
			gl.glVertex3f(1, 1, -1)
			gl.glTexCoord2f(1, 1)
			gl.glVertex3f(1, 1, 1)
			gl.glTexCoord2f(0, 1)
			gl.glVertex3f(-1, 1, 1)
			gl.glEnd()
			
			gl.glPopMatrix()
		
		# Mât brisé
		if decay < 0.9:
			gl.glPushMatrix()
			
			# Angle du mât selon la dégradation
			mast_angle = 90 - 80 * decay
			gl.glTranslatef(0, size * 0.15, 0)
			gl.glRotatef(mast_angle, 1, 0, 0)
			
			# Dessiner le mât
			gl.glPushMatrix()
			gl.glRotatef(90, 1, 0, 0)
			
			mast_quad = glu.gluNewQuadric()
			glu.gluQuadricTexture(mast_quad, gl.GL_TRUE)
			glu.gluQuadricNormals(mast_quad, glu.GLU_SMOOTH)
			glu.gluCylinder(mast_quad, size * 0.03, size * 0.01, size * 0.7, 8, 4)
			glu.gluDeleteQuadric(mast_quad)
			
			gl.glPopMatrix()
			gl.glPopMatrix()
	
	def _render_underwater_cave(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Dessine une grotte sous-marine.
		
		Args:
			size: Taille globale de la grotte
			properties: Propriétés spécifiques de la grotte
		"""
		# Extraire les propriétés
		depth = properties.get("depth", 30)
		complexity = properties.get("complexity", 0.5)
		
		# Entrée de la grotte (arche)
		gl.glPushMatrix()
		
		# Base de l'arche
		gl.glPushMatrix()
		gl.glScalef(size, size * 0.7, size * 0.2)
		
		# Dessiner une forme d'arche
		gl.glBegin(gl.GL_TRIANGLE_FAN)
		
		# Centre de l'arche (point en haut au milieu)
		gl.glNormal3f(0, 0, 1)
		gl.glTexCoord2f(0.5, 1.0)
		gl.glVertex3f(0, 1, 0)
		
		# Points sur le bord de l'arche
		segments = 12
		for i in range(segments + 1):
			angle = np.pi * i / segments
			x = np.cos(angle)
			y = np.sin(angle)
			if angle > 0 and angle < np.pi:  # Seulement la moitié supérieure
				gl.glNormal3f(0, 0, 1)
				gl.glTexCoord2f(0.5 + 0.5 * x, 0.5 + 0.5 * y)
				gl.glVertex3f(x, y, 0)
		
		gl.glEnd()
		gl.glPopMatrix()
		
		# Profondeur de la grotte
		gl.glPushMatrix()
		gl.glTranslatef(0, 0, -depth)
		gl.glScalef(size * 1.5, size * 1.2, depth)
		
		# Dessiner le tunnel
		gl.glBegin(gl.GL_QUADS)
		
		# Côtés du tunnel
		steps = int(10 * complexity)
		for i in range(steps):
			t1 = i / steps
			t2 = (i + 1) / steps
			
			# Variation de taille pour un aspect plus naturel
			radius1 = 1.0 - 0.3 * t1 * np.sin(t1 * 5)
			radius2 = 1.0 - 0.3 * t2 * np.sin(t2 * 5)
			
			for j in range(12):
				angle1 = 2 * np.pi * j / 12
				angle2 = 2 * np.pi * (j + 1) / 12
				
				x1 = np.cos(angle1) * radius1
				y1 = np.sin(angle1) * radius1
				x2 = np.cos(angle2) * radius1
				y2 = np.sin(angle2) * radius1
				
				x3 = np.cos(angle2) * radius2
				y3 = np.sin(angle2) * radius2
				x4 = np.cos(angle1) * radius2
				y4 = np.sin(angle1) * radius2
				
				# Calcul de la normale
				nx = (x1 + x2 + x3 + x4) / 4
				ny = (y1 + y2 + y3 + y4) / 4
				norm = np.sqrt(nx*nx + ny*ny)
				if norm > 0:
					nx /= norm
					ny /= norm
				
				gl.glNormal3f(nx, ny, 0)
				gl.glTexCoord2f(t1 * 2, angle1 / (2 * np.pi))
				gl.glVertex3f(x1, y1, t1)
				
				gl.glNormal3f(nx, ny, 0)
				gl.glTexCoord2f(t1 * 2, angle2 / (2 * np.pi))
				gl.glVertex3f(x2, y2, t1)
				
				gl.glNormal3f(nx, ny, 0)
				gl.glTexCoord2f(t2 * 2, angle2 / (2 * np.pi))
				gl.glVertex3f(x3, y3, t2)
				
				gl.glNormal3f(nx, ny, 0)
				gl.glTexCoord2f(t2 * 2, angle1 / (2 * np.pi))
				gl.glVertex3f(x4, y4, t2)
		
		gl.glEnd()
		gl.glPopMatrix()
		
		gl.glPopMatrix()
	
	def _render_kelp_forest(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Dessine une forêt d'algues.
		
		Args:
			size: Taille globale de la forêt
			properties: Propriétés spécifiques de la forêt
		"""
		# Extraire les propriétés
		density = properties.get("density", 0.7)
		height = properties.get("height", 10)
		
		# Nombre d'algues
		num_kelp = int(size * density * 20)
		
		for i in range(num_kelp):
			# Position aléatoire
			angle = np.random.random() * 2 * np.pi
			radius = np.random.random() * size
			x = np.cos(angle) * radius
			z = np.sin(angle) * radius
			
			# Taille variée
			kelp_height = height * (0.7 + 0.6 * np.random.random())
			kelp_width = size * 0.05 * (0.7 + 0.6 * np.random.random())
			
			gl.glPushMatrix()
			gl.glTranslatef(x, 0, z)
			
			# Dessiner la tige principale (segments courbes)
			num_segments = int(kelp_height / 2)
			segment_height = kelp_height / num_segments
			
			# Points de contrôle pour la courbe
			control_points = []
			for j in range(num_segments + 1):
				if j == 0:
					offset_x = 0
					offset_z = 0
				else:
					# Offset aléatoire qui respecte une continuité
					offset_x = control_points[-1][0] + 0.5 * np.random.random() - 0.25
					offset_z = control_points[-1][2] + 0.5 * np.random.random() - 0.25
				
				control_points.append((offset_x, j * segment_height, offset_z))
			
			# Dessiner les segments
			for j in range(num_segments):
				gl.glPushMatrix()
				
				# Position et orientation du segment
				p1 = control_points[j]
				p2 = control_points[j + 1]
				
				direction = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
				length = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
				
				# Amincir vers le haut
				thickness_factor = 1.0 - 0.7 * (j / num_segments)
				
				gl.glTranslatef(p1[0], p1[1], p1[2])
				
				# Orienter vers le point suivant
				if length > 0:
					# Angle autour de l'axe Y
					angle_y = np.degrees(np.arctan2(direction[0], direction[2]))
					gl.glRotatef(angle_y, 0, 1, 0)
					
					# Angle par rapport à l'axe vertical
					dir_xz = np.sqrt(direction[0]**2 + direction[2]**2)
					angle_x = np.degrees(np.arctan2(dir_xz, direction[1])) - 90
					gl.glRotatef(angle_x, 1, 0, 0)
				
				# Dessiner le segment
				quad = glu.gluNewQuadric()
				glu.gluQuadricTexture(quad, gl.GL_TRUE)
				glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
				glu.gluCylinder(quad, kelp_width * thickness_factor, kelp_width * thickness_factor * 0.8, length, 8, 1)
				glu.gluDeleteQuadric(quad)
				
				gl.glPopMatrix()
			
			# Dessiner les feuilles
			for j in range(1, num_segments, 2):
				p = control_points[j]
				
				gl.glPushMatrix()
				gl.glTranslatef(p[0], p[1], p[2])
				
				# Rotation aléatoire pour la feuille
				gl.glRotatef(np.random.random() * 360, 0, 1, 0)
				
				# Taille réduite vers le haut
				leaf_size = kelp_width * 10 * (1.0 - 0.5 * (j / num_segments))
				
				# Dessiner une feuille (plan texturé)
				gl.glBegin(gl.GL_TRIANGLE_FAN)
				
				gl.glNormal3f(0, 0, 1)
				gl.glTexCoord2f(0.5, 0.5)
				gl.glVertex3f(0, 0, 0)
				
				points = 8
				for k in range(points + 1):
					angle = 2 * np.pi * k / points
					x = np.cos(angle) * leaf_size
					y = np.sin(angle) * leaf_size * 2  # Feuilles allongées
					
					gl.glTexCoord2f(0.5 + 0.5 * np.cos(angle), 0.5 + 0.5 * np.sin(angle))
					gl.glVertex3f(x, y, 0)
				
				gl.glEnd()
				
				gl.glPopMatrix()
			
			gl.glPopMatrix()
	
	def _render_abyss(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Dessine un abîme océanique.
		
		Args:
			size: Taille globale de l'abîme
			properties: Propriétés spécifiques de l'abîme
		"""
		# Extraire les propriétés
		thermal_activity = properties.get("thermal_activity", 0.5)
		
		# Dessiner le gouffre principal
		gl.glPushMatrix()
		gl.glRotatef(90, 1, 0, 0)  # Orienter vers le bas
		
		# Cône pour l'abîme
		quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(quad, gl.GL_TRUE)
		glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
		glu.gluCylinder(quad, size * 1.2, size * 0.5, size * 2, 16, 4)
		glu.gluDeleteQuadric(quad)
		
		# Bas de l'abîme (disque)
		gl.glTranslatef(0, 0, size * 2)
		glu.gluDisk(quad, 0, size * 0.5, 16, 4)
		glu.gluDeleteQuadric(quad)
		
		gl.glPopMatrix()
		
		# Activité thermale si présente
		if thermal_activity > 0.2:
			# Cheminées hydrothermales autour de l'abîme
			num_vents = int(thermal_activity * 5) + 1
			
			for i in range(num_vents):
				angle = 2 * np.pi * i / num_vents
				distance = size * (0.7 + 0.3 * np.random.random())
				x = np.cos(angle) * distance
				z = np.sin(angle) * distance
				
				vent_size = size * 0.15 * (0.7 + 0.6 * np.random.random())
				
				gl.glPushMatrix()
				gl.glTranslatef(x, 0, z)
				self._render_deep_sea_vent(vent_size)
				gl.glPopMatrix()
	
	def _render_water(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu de la surface et du volume d'eau.
		
		Args:
			render_mode: Mode de rendu à utiliser
		"""
		if not self.world:
			return
			
		# Activer la transparence pour l'eau
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		
		# Configurer le matériau pour l'eau
		if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, 
						  [self.water_color[0] * 0.3, self.water_color[1] * 0.3, 
						   self.water_color[2] * 0.3, self.water_color[3]])
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, 
						  [self.water_color[0], self.water_color[1], 
						   self.water_color[2], self.water_color[3]])
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 0.5])
			gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, 100.0)
			
			# Appliquer une texture si en mode texturé
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["water"])
		else:
			gl.glColor4f(self.water_color[0], self.water_color[1], 
					   self.water_color[2], self.water_color[3])
		
		# Dessiner la surface de l'eau
		world_size = self.world.size
		
		gl.glPushMatrix()
		
		# Appliquer une ondulation à la surface
		if render_mode != RenderMode.WIREFRAME:
			gl.glEnable(gl.GL_BLEND)
			gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		
		# Rendre la surface de l'eau
		gl.glPushMatrix()
		gl.glScalef(world_size[0] / 20, 1.0, world_size[2] / 20)
		gl.glCallList(self.display_lists["water_surface"])
		gl.glPopMatrix()
		
		gl.glPopMatrix()
		
		# Restaurer l'état
		gl.glDisable(gl.GL_BLEND)
	
	def _render_resources(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu des ressources alimentaires.
		
		Args:
			render_mode: Mode de rendu à utiliser
		"""
		if not self.world:
			return
			
		# Activer la transparence pour certaines ressources
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		
		# Récupérer les ressources du monde
		resources = self.world.foodResources
		
		for i, resource in enumerate(resources):
			# Ne pas rendre les ressources consommées
			if resource.isConsumed:
				continue
				
			# Récupérer les propriétés de la ressource
			position = resource.position
			size = resource.size
			resource_type = resource.foodType
			
			# Sélectionner la couleur et la texture en fonction du type
			if resource_type == "algae":
				color = (0.2, 0.8, 0.3, 0.7)  # Vert
				texture_name = "algae"
			elif resource_type == "plankton":
				color = (0.6, 0.8, 1.0, 0.5)  # Bleu clair
				texture_name = "plankton"
			elif resource_type == "small_fish":
				color = (0.7, 0.7, 0.9, 0.9)  # Bleu-gris
				texture_name = "small_fish"
			elif resource_type == "detritus":
				color = (0.6, 0.5, 0.3, 0.8)  # Brun
				texture_name = "detritus"
			else:
				color = (0.7, 0.7, 0.7, 0.7)  # Gris par défaut
				texture_name = "plankton"
			
			# Sauvegarder l'état
			gl.glPushMatrix()
			
			# Positionner la ressource
			gl.glTranslatef(position[0], position[1], position[2])
			
			# Animation légère pour certains types
			if resource_type == "plankton" or resource_type == "small_fish":
				gl.glRotatef(self.wave_time * 20 % 360, 0, 1, 0)
			
			# Configurer le matériau
			if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, 
							 [color[0], color[1], color[2], color[3]])
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
				gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 50.0)
				
				# Appliquer une texture si en mode texturé
				if render_mode == RenderMode.TEXTURED and texture_name in self.textures:
					gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[texture_name])
			else:
				gl.glColor4f(*color)
			
			# Dessiner la ressource en fonction de son type
			if resource_type == "algae":
				self._render_algae(size)
			elif resource_type == "plankton":
				self._render_plankton(size)
			elif resource_type == "small_fish":
				self._render_small_fish(size)
			elif resource_type == "detritus":
				self._render_detritus(size)
			else:
				# Par défaut, une simple sphère
				quad = glu.gluNewQuadric()
				glu.gluQuadricTexture(quad, gl.GL_TRUE)
				glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
				glu.gluSphere(quad, size, 8, 8)
				glu.gluDeleteQuadric(quad)
			
			# Restaurer l'état
			gl.glPopMatrix()
		
		# Restaurer l'état de transparence
		gl.glDisable(gl.GL_BLEND)
	
	def _render_algae(self, size: float) -> None:
		"""
		Dessine une algue.
		
		Args:
			size: Taille de l'algue
		"""
		# Tronc principal
		gl.glPushMatrix()
		gl.glRotatef(90, 1, 0, 0)  # Orienter vers le haut
		
		stem_quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(stem_quad, gl.GL_TRUE)
		glu.gluQuadricNormals(stem_quad, glu.GLU_SMOOTH)
		glu.gluCylinder(stem_quad, size * 0.1, size * 0.05, size, 8, 1)
		glu.gluDeleteQuadric(stem_quad)
		gl.glPopMatrix()
		
		# Feuilles
		num_leaves = int(5 + size * 3)
		
		for i in range(num_leaves):
			height = (i / num_leaves) * size
			angle = i * 137.5  # Angle d'or pour une distribution naturelle
			
			gl.glPushMatrix()
			gl.glTranslatef(0, height, 0)
			gl.glRotatef(angle, 0, 1, 0)
			gl.glRotatef(30, 1, 0, 0)  # Incliner la feuille
			
			# Taille de la feuille
			leaf_size = size * 0.8 * (0.5 + 0.5 * (i / num_leaves))
			
			# Dessiner une feuille simple
			gl.glBegin(gl.GL_TRIANGLE_FAN)
			
			gl.glNormal3f(0, 0, 1)
			gl.glTexCoord2f(0.5, 0)
			gl.glVertex3f(0, 0, 0)
			
			points = 8
			for j in range(points + 1):
				t = j / points
				angle_leaf = np.pi * t
				x = np.cos(angle_leaf) * leaf_size * 0.2
				y = np.sin(angle_leaf) * leaf_size
				
				gl.glTexCoord2f(t, 1)
				gl.glVertex3f(x, y, 0)
			
			gl.glEnd()
			
			gl.glPopMatrix()
	
	def _render_plankton(self, size: float) -> None:
		"""
		Dessine du plancton.
		
		Args:
			size: Taille du plancton
		"""
		# Nuage de particules
		num_particles = int(10 + size * 20)
		
		gl.glPointSize(2.0)
		gl.glBegin(gl.GL_POINTS)
		
		for i in range(num_particles):
			# Position aléatoire dans une sphère
			theta = np.random.random() * 2 * np.pi
			phi = np.random.random() * np.pi
			radius = size * np.random.random()
			
			x = radius * np.sin(phi) * np.cos(theta)
			y = radius * np.sin(phi) * np.sin(theta)
			z = radius * np.cos(phi)
			
			gl.glVertex3f(x, y, z)
		
		gl.glEnd()
		
		# Centre plus dense
		gl.glPointSize(3.0)
		gl.glBegin(gl.GL_POINTS)
		
		num_center = int(num_particles / 3)
		for i in range(num_center):
			# Position aléatoire dans une sphère plus petite
			theta = np.random.random() * 2 * np.pi
			phi = np.random.random() * np.pi
			radius = size * 0.5 * np.random.random()
			
			x = radius * np.sin(phi) * np.cos(theta)
			y = radius * np.sin(phi) * np.sin(theta)
			z = radius * np.cos(phi)
			
			gl.glVertex3f(x, y, z)
		
		gl.glEnd()
	
	def _render_small_fish(self, size: float) -> None:
		"""
		Dessine un petit poisson.
		
		Args:
			size: Taille du poisson
		"""
		# Corps du poisson (sphéroïde)
		gl.glPushMatrix()
		gl.glScalef(size, size * 0.5, size * 0.2)
		
		quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(quad, gl.GL_TRUE)
		glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
		glu.gluSphere(quad, 1.0, 12, 8)
		glu.gluDeleteQuadric(quad)
		gl.glPopMatrix()
		
		# Queue
		gl.glPushMatrix()
		gl.glTranslatef(-size * 1.0, 0, 0)
		gl.glRotatef(90, 0, 1, 0)
		
		# Animation de battement de queue
		tail_angle = 20 * np.sin(self.wave_time * 5)
		gl.glRotatef(tail_angle, 0, 0, 1)
		
		gl.glBegin(gl.GL_TRIANGLES)
		gl.glNormal3f(0, 0, 1)
		gl.glTexCoord2f(0, 0)
		gl.glVertex3f(0, 0, 0)
		
		gl.glTexCoord2f(1, 0)
		gl.glVertex3f(size * 0.7, size * 0.5, 0)
		
		gl.glTexCoord2f(1, 1)
		gl.glVertex3f(size * 0.7, -size * 0.5, 0)
		gl.glEnd()
		
		gl.glPopMatrix()
		
		# Nageoire dorsale
		gl.glPushMatrix()
		gl.glTranslatef(0, size * 0.5, 0)
		gl.glRotatef(90, 1, 0, 0)
		
		gl.glBegin(gl.GL_TRIANGLES)
		gl.glNormal3f(0, 0, 1)
		gl.glTexCoord2f(0, 0)
		gl.glVertex3f(0, 0, 0)
		
		gl.glTexCoord2f(1, 0)
		gl.glVertex3f(size * 0.3, 0, 0)
		
		gl.glTexCoord2f(0.5, 1)
		gl.glVertex3f(0, size * 0.4, 0)
		gl.glEnd()
		
		gl.glPopMatrix()
	
	def _render_detritus(self, size: float) -> None:
		"""
		Dessine des détritus.
		
		Args:
			size: Taille des détritus
		"""
		# Nuage de particules irrégulières
		num_particles = int(5 + size * 10)
		
		for i in range(num_particles):
			# Position aléatoire dans un disque
			angle = np.random.random() * 2 * np.pi
			radius = size * np.random.random()
			
			x = radius * np.cos(angle)
			y = np.random.random() * size * 0.5
			z = radius * np.sin(angle)
			
			# Taille aléatoire pour la particule
			particle_size = size * 0.2 * (0.5 + 0.5 * np.random.random())
			
			gl.glPushMatrix()
			gl.glTranslatef(x, y, z)
			gl.glRotatef(np.random.random() * 360, 0, 1, 0)
			gl.glRotatef(np.random.random() * 360, 1, 0, 0)
			
			# Dessiner une forme irrégulière
			gl.glBegin(gl.GL_TRIANGLE_FAN)
			
			gl.glNormal3f(0, 1, 0)
			gl.glTexCoord2f(0.5, 0.5)
			gl.glVertex3f(0, 0, 0)
			
			points = 5
			for j in range(points + 1):
				angle_p = 2 * np.pi * j / points
				radius_p = particle_size * (0.7 + 0.3 * np.sin(angle_p * 3))
				
				x_p = radius_p * np.cos(angle_p)
				z_p = radius_p * np.sin(angle_p)
				
				gl.glTexCoord2f(0.5 + 0.5 * np.cos(angle_p), 0.5 + 0.5 * np.sin(angle_p))
				gl.glVertex3f(x_p, 0, z_p)
			
			gl.glEnd()
			
			gl.glPopMatrix()
	
	def _render_zones(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu des zones environnementales.
		
		Args:
			render_mode: Mode de rendu à utiliser
		"""
		if not self.world:
			return
			
		# Activer la transparence
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		
		# Récupérer les zones du monde
		zones = self.world.zones
		
		for zone in zones:
			# Récupérer les limites de la zone
			bounds = zone.bounds
			name = zone.name
			
			# Calculer les dimensions
			min_x, max_x = bounds[0]
			min_y, max_y = bounds[1]
			min_z, max_z = bounds[2]
			
			width = max_x - min_x
			height = max_y - min_y
			depth = max_z - min_z
			
			# Sélectionner la couleur en fonction du type de zone
			if "Surface" in name:
				color = (0.2, 0.6, 0.9, 0.1)  # Bleu clair
			elif "Deep" in name:
				color = (0.1, 0.2, 0.5, 0.1)  # Bleu foncé
			elif "Reef" in name:
				color = (0.8, 0.8, 0.2, 0.1)  # Jaune
			elif "Current" in name:
				color = (0.2, 0.8, 0.8, 0.1)  # Turquoise
			else:
				color = (0.5, 0.5, 0.5, 0.1)  # Gris par défaut
			
			# Sauvegarder l'état
			gl.glPushMatrix()
			
			# Positionner la zone
			gl.glTranslatef(min_x, min_y, min_z)
			
			# Configurer le matériau
			if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
				gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE, 
							 [color[0], color[1], color[2], color[3]])
				gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, [0.0, 0.0, 0.0, 0.0])
				gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, 0.0)
			else:
				gl.glColor4f(*color)
			
			# Dessiner un cube transparent pour représenter la zone
			gl.glBegin(gl.GL_QUADS)
			
			# Face avant
			gl.glNormal3f(0, 0, 1)
			gl.glVertex3f(0, 0, depth)
			gl.glVertex3f(width, 0, depth)
			gl.glVertex3f(width, height, depth)
			gl.glVertex3f(0, height, depth)
			
			# Face arrière
			gl.glNormal3f(0, 0, -1)
			gl.glVertex3f(0, 0, 0)
			gl.glVertex3f(0, height, 0)
			gl.glVertex3f(width, height, 0)
			gl.glVertex3f(width, 0, 0)
			
			# Face gauche
			gl.glNormal3f(-1, 0, 0)
			gl.glVertex3f(0, 0, 0)
			gl.glVertex3f(0, 0, depth)
			gl.glVertex3f(0, height, depth)
			gl.glVertex3f(0, height, 0)
			
			# Face droite
			gl.glNormal3f(1, 0, 0)
			gl.glVertex3f(width, 0, 0)
			gl.glVertex3f(width, height, 0)
			gl.glVertex3f(width, height, depth)
			gl.glVertex3f(width, 0, depth)
			
			# Face supérieure
			gl.glNormal3f(0, 1, 0)
			gl.glVertex3f(0, height, 0)
			gl.glVertex3f(0, height, depth)
			gl.glVertex3f(width, height, depth)
			gl.glVertex3f(width, height, 0)
			
			# Face inférieure
			gl.glNormal3f(0, -1, 0)
			gl.glVertex3f(0, 0, 0)
			gl.glVertex3f(width, 0, 0)
			gl.glVertex3f(width, 0, depth)
			gl.glVertex3f(0, 0, depth)
			
			gl.glEnd()
			
			gl.glPopMatrix()
		
		# Restaurer l'état
		gl.glDisable(gl.GL_BLEND)
	
	def _render_grid(self) -> None:
		"""
		Effectue le rendu de la grille de référence.
		"""
		# Sauvegarder l'état
		gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
		
		# Désactiver l'éclairage et la profondeur pour la grille
		gl.glDisable(gl.GL_LIGHTING)
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		
		# Couleur de la grille
		gl.glColor4f(*self.grid_color)
		
		# Positionner la grille
		gl.glPushMatrix()
		
		# Appeler la liste d'affichage
		gl.glCallList(self.display_lists["grid"])
		
		gl.glPopMatrix()
		
		# Restaurer l'état
		gl.glPopAttrib()
	
	def _render_currents(self) -> None:
		"""
		Effectue le rendu des courants marins.
		"""
		if not self.world:
			return
			
		# Sauvegarder l'état
		gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
		
		# Désactiver l'éclairage pour les flèches
		gl.glDisable(gl.GL_LIGHTING)
		
		# Couleur des courants
		gl.glColor3f(0.2, 0.6, 0.8)
		
		# Grille d'échantillonnage des courants
		grid_size = 10
		step_x = self.world.size[0] / grid_size
		step_y = self.world.size[1] / grid_size
		step_z = self.world.size[2] / grid_size
		
		for x in range(grid_size):
			for y in range(grid_size):
				for z in range(grid_size):
					# Position dans le monde
					pos_x = x * step_x
					pos_y = y * step_y
					pos_z = z * step_z
					
					position = np.array([pos_x, pos_y, pos_z])
					
					# Obtenir le courant à cette position
					current = self.world.waterProperties.getCurrentAt(position)
					
					# Ne dessiner que si le courant est significatif
					if np.linalg.norm(current) > 0.05:
						# Normaliser et mettre à l'échelle pour l'affichage
						magnitude = np.linalg.norm(current)
						normalized = current / magnitude
						
						# Longueur de la flèche proportionnelle à la magnitude
						arrow_length = min(step_x, step_y, step_z) * 0.8 * magnitude * 2
						
						# Dessiner une flèche simple
						gl.glPushMatrix()
						gl.glTranslatef(pos_x, pos_y, pos_z)
						
						# Orienter vers la direction du courant
						if abs(normalized[2]) < 0.999:
							# Angle entre la direction par défaut (axe z) et la direction cible
							angle = np.degrees(np.arccos(normalized[2]))
							
							# Axe de rotation (perpendiculaire aux deux directions)
							rotation_axis = np.cross([0, 0, 1], normalized)
							rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
							
							gl.glRotatef(angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
						elif normalized[2] < 0:
							# Si la direction est exactement opposée à z, tourner de 180° autour de x
							gl.glRotatef(180, 1, 0, 0)
						
						# Dessiner le corps de la flèche
						gl.glBegin(gl.GL_LINES)
						gl.glVertex3f(0, 0, 0)
						gl.glVertex3f(0, 0, arrow_length)
						gl.glEnd()
						
						# Dessiner la pointe de la flèche
