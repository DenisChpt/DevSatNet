import pygame
import numpy as np
import OpenGL.GL as gl
import OpenGL.GLU as glu
from typing import Dict, List, Tuple, Any, Optional, Set
import math
import os
import sys
import time

# Ajouter le répertoire parent au chemin d'importation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import get_logger
from core.environment.marine_world import MarineWorld
from core.environment.terrain import Terrain
from core.environment.resources import FoodResource
from core.environment.zones import EnvironmentalZone
from visualization.renderer import RenderMode


class EnvironmentRenderer:
	"""
	Classe responsable du rendu de l'environnement marin.
	Gère le rendu du terrain, de l'eau, des courants, des ressources et des zones environnementales.
	"""

	def __init__(self) -> None:
		"""
		Initialise le renderer d'environnement.
		"""
		self.logger = get_logger()
		self.world: Optional[MarineWorld] = None
		self.textures: Dict[str, int] = {}
		self.display_lists: Dict[str, int] = {}

		# Paramètres du rendu
		self.water_detail = 32          # Niveau de détail pour le maillage d'eau
		self.terrain_detail = 64        # Niveau de détail pour le terrain
		self.show_water = True
		self.show_terrain = True
		self.show_resources = True
		self.show_zones = True
		self.show_grid = False
		self.water_transparency = 0.7   # Transparence de l'eau (0-1)
		self.water_wave_amplitude = 0.2  # Amplitude des vagues
		self.water_wave_frequency = 0.5  # Fréquence des vagues

		# Limites du rendu (pour l'optimisation)
		self.render_distance = 500.0

		# Chargement des textures et initialisation des ressources
		self._init_resources()

	def _init_resources(self) -> None:
		"""
		Initialise les ressources nécessaires au rendu (textures, modèles, etc.).
		"""
		try:
			self._create_water_texture()
			self._create_terrain_textures()
			self._create_resource_textures()
			self._create_zone_textures()

			self._create_water_display_list()
			self._create_terrain_display_list()
			self._create_grid_display_list()

			self.logger.info("Ressources du EnvironmentRenderer initialisées", module="visualization")
		except Exception as e:
			self.logger.error(f"Erreur lors de l'initialisation des ressources: {str(e)}", 
								module="visualization", exc_info=True)

	def _create_water_texture(self) -> None:
		"""Crée une texture d'eau pour la surface de l'océan."""
		texture_id = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

		width, height = 256, 256
		texture_data = np.zeros((height, width, 4), dtype=np.uint8)

		for y in range(height):
			for x in range(width):
				nx = x / width
				ny = y / height
				wave1 = math.sin(nx * 10 + ny * 8) * 0.5 + 0.5
				wave2 = math.sin(nx * 5 - ny * 12 + 0.2) * 0.5 + 0.5
				detail = math.sin(nx * 30 + ny * 30) * 0.1 + 0.9
				blend = (wave1 * 0.6 + wave2 * 0.4) * detail
				blue = int(150 + blend * 40)
				green = int(150 + blend * 20)
				alpha = 180
				texture_data[y, x] = [50, green, blue, alpha]

		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
						gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texture_data)

		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

		self.textures["water"] = texture_id

	def _create_terrain_textures(self) -> None:
		"""Crée des textures pour les différents types de terrain sous-marin."""
		terrain_types = {
			"sand": (220, 210, 170),
			"mud": (130, 110, 90),
			"rock": (120, 120, 130),
			"coral": (240, 150, 150),
			"gravel": (180, 170, 160)
		}

		for terrain_type, base_color in terrain_types.items():
			texture_id = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

			width, height = 256, 256
			texture_data = np.zeros((height, width, 3), dtype=np.uint8)

			for y in range(height):
				for x in range(width):
					if terrain_type == "sand":
						noise = np.random.randint(-10, 11)
						wave = math.sin(x / 20 + y / 20) * 5
					elif terrain_type == "mud":
						noise = np.random.randint(-5, 6)
						if np.random.random() < 0.01:
							noise = np.random.randint(-20, -10)
					elif terrain_type == "rock":
						noise = np.random.randint(-20, 21)
						if (x + y) % 40 < 2:
							noise = -30
					elif terrain_type == "coral":
						noise = np.random.randint(-15, 16)
						if np.random.random() < 0.05:
							r, g, b = base_color
							r = min(255, r + np.random.randint(0, 40))
							texture_data[y, x] = [r, g // 2, b // 2]
							continue
					else:  # gravel
						noise = np.random.randint(-25, 26)

					r = max(0, min(255, base_color[0] + noise))
					g = max(0, min(255, base_color[1] + noise))
					b = max(0, min(255, base_color[2] + noise))
					texture_data[y, x] = [r, g, b]

			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
							gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_data)

			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

			self.textures[f"terrain_{terrain_type}"] = texture_id

	def _create_resource_textures(self) -> None:
		"""Crée des textures pour les différentes ressources (nourriture, etc.)."""
		resource_types = {
			"algae": (50, 180, 50),
			"plankton": (180, 200, 100),
			"small_fish": (180, 180, 220),
			"detritus": (120, 100, 60)
		}

		for resource_type, base_color in resource_types.items():
			texture_id = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

			width, height = 64, 64
			texture_data = np.zeros((height, width, 4), dtype=np.uint8)

			for y in range(height):
				for x in range(width):
					dx, dy = x - width // 2, y - height // 2
					dist = math.sqrt(dx * dx + dy * dy) / (width // 2)
					alpha = int(255 * max(0, 1 - dist * 1.2))

					if resource_type == "algae":
						wave = math.sin(y / 5 + x / 10) * 0.5 + 0.5
						r = int(base_color[0] * wave)
						g = int(base_color[1] * (0.8 + wave * 0.2))
						b = int(base_color[2] * wave)
					elif resource_type == "plankton":
						if np.random.random() < 0.1:
							brightness = np.random.random() * 0.5 + 0.5
							r = int(base_color[0] * brightness)
							g = int(base_color[1] * brightness)
							b = int(base_color[2] * brightness)
						else:
							r, g, b = base_color
					elif resource_type == "small_fish":
						fish_shape = (abs(dx) < width // 4) and (abs(dy) < height // 6)
						if fish_shape:
							r, g, b = base_color
							alpha = 255
						else:
							r, g, b = 0, 0, 0
							alpha = 0
					else:  # detritus
						noise = np.random.randint(-30, 30)
						r = max(0, min(255, base_color[0] + noise))
						g = max(0, min(255, base_color[1] + noise))
						b = max(0, min(255, base_color[2] + noise))

					texture_data[y, x] = [r, g, b, alpha]

			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
							gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texture_data)

			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

			self.textures[f"resource_{resource_type}"] = texture_id

	def _create_zone_textures(self) -> None:
		"""Crée des textures pour les différentes zones environnementales."""
		texture_id = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)

		width, height = 64, 64
		texture_data = np.zeros((height, width, 4), dtype=np.uint8)

		for y in range(height):
			for x in range(width):
				is_border = (x < 2 or x > width - 3 or y < 2 or y > height - 3 or
							 abs(x - width // 2) < 1 or abs(y - height // 2) < 1)
				if is_border:
					texture_data[y, x] = [255, 255, 255, 180]
				else:
					texture_data[y, x] = [100, 100, 100, 30]

		gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
						gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, texture_data)

		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)

		self.textures["zone_boundary"] = texture_id

	def _create_water_display_list(self) -> None:
		"""Crée une liste d'affichage pour la surface de l'eau."""
		list_id = gl.glGenLists(1)
		gl.glNewList(list_id, gl.GL_COMPILE)
		gl.glBegin(gl.GL_QUADS)

		size = 1.0
		subdivs = self.water_detail
		step = size / subdivs
		half_size = size / 2

		for i in range(subdivs):
			for j in range(subdivs):
				x0 = -half_size + i * step
				z0 = -half_size + j * step
				x1 = x0 + step
				z1 = z0 + step
				tx0 = i / subdivs * 5
				tz0 = j / subdivs * 5
				tx1 = (i + 1) / subdivs * 5
				tz1 = (j + 1) / subdivs * 5

				gl.glNormal3f(0.0, 1.0, 0.0)
				gl.glTexCoord2f(tx0, tz0)
				gl.glVertex3f(x0, 0.0, z0)
				gl.glTexCoord2f(tx1, tz0)
				gl.glVertex3f(x1, 0.0, z0)
				gl.glTexCoord2f(tx1, tz1)
				gl.glVertex3f(x1, 0.0, z1)
				gl.glTexCoord2f(tx0, tz1)
				gl.glVertex3f(x0, 0.0, z1)

		gl.glEnd()
		gl.glEndList()

		self.display_lists["water_surface"] = list_id

	def _create_terrain_display_list(self) -> None:
		"""Crée une liste d'affichage pour le terrain de base."""
		list_id = gl.glGenLists(1)
		gl.glNewList(list_id, gl.GL_COMPILE)
		gl.glBegin(gl.GL_QUADS)

		gl.glNormal3f(0.0, 1.0, 0.0)
		gl.glTexCoord2f(0.0, 0.0)
		gl.glVertex3f(0.0, 0.0, 0.0)
		gl.glTexCoord2f(1.0, 0.0)
		gl.glVertex3f(1.0, 0.0, 0.0)
		gl.glTexCoord2f(1.0, 1.0)
		gl.glVertex3f(1.0, 0.0, 1.0)
		gl.glTexCoord2f(0.0, 1.0)
		gl.glVertex3f(0.0, 0.0, 1.0)

		gl.glEnd()
		gl.glEndList()

		self.display_lists["terrain_tile"] = list_id

	def _create_grid_display_list(self) -> None:
		"""Crée une liste d'affichage pour la grille de référence."""
		list_id = gl.glGenLists(1)
		gl.glNewList(list_id, gl.GL_COMPILE)
		gl.glBegin(gl.GL_LINES)

		grid_size = 100.0
		grid_step = 10.0

		for i in range(int(-grid_size), int(grid_size) + 1, int(grid_step)):
			gl.glVertex3f(i, 0, -grid_size)
			gl.glVertex3f(i, 0, grid_size)

		for i in range(int(-grid_size), int(grid_size) + 1, int(grid_step)):
			gl.glVertex3f(-grid_size, 0, i)
			gl.glVertex3f(grid_size, 0, i)

		gl.glEnd()
		gl.glEndList()

		self.display_lists["reference_grid"] = list_id

	def set_world(self, world: MarineWorld) -> None:
		"""
		Définit le monde marin à rendre.
		
		Args:
			world: Instance du monde marin.
		"""
		self.world = world

	def render(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu de l'environnement marin.
		
		Args:
			render_mode: Mode de rendu à utiliser.
		"""
		if not self.world:
			return

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

		if self.show_terrain:
			self._render_terrain(render_mode)
		if self.show_water:
			self._render_water(render_mode)
		if self.show_resources:
			self._render_resources(render_mode)
		if self.show_zones:
			self._render_zones(render_mode)
		if self.show_grid:
			self._render_grid()

	def _render_terrain(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu du terrain sous-marin.
		
		Args:
			render_mode: Mode de rendu à utiliser.
		"""
		terrain = self.world.terrain
		world_size = self.world.size

		render_detail = min(self.terrain_detail, min(terrain.resolution[0], terrain.resolution[2]))
		step_x = terrain.resolution[0] / render_detail
		step_z = terrain.resolution[2] / render_detail

		tile_size_x = world_size[0] / render_detail
		tile_size_z = world_size[2] / render_detail

		gl.glPushMatrix()

		for i in range(render_detail):
			for j in range(render_detail):
				terrain_i = min(int(i * step_x), terrain.resolution[0] - 1)
				terrain_j = min(int(j * step_z), terrain.resolution[2] - 1)
				height = terrain.elevationMap[terrain_i, terrain_j]
				terrain_type = terrain.terrainTypeMap[terrain_i, terrain_j]

				heights: List[float] = []
				for di, dj in [(0, 0), (1, 0), (1, 1), (0, 1)]:
					ti = min(terrain_i + di, terrain.resolution[0] - 1)
					tj = min(terrain_j + dj, terrain.resolution[2] - 1)
					heights.append(terrain.elevationMap[ti, tj])

				pos_x = i * tile_size_x
				pos_z = j * tile_size_z

				gl.glPushMatrix()
				gl.glTranslatef(pos_x, height, pos_z)
				gl.glScalef(tile_size_x, 1.0, tile_size_z)

				if render_mode in (RenderMode.SHADED, RenderMode.TEXTURED):
					if terrain_type == 0:  # Sable
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.6, 1.0])
						if render_mode == RenderMode.TEXTURED:
							gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.get("terrain_sand", 0))
					elif terrain_type == 1:  # Vase
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.5, 0.4, 0.3, 1.0])
						if render_mode == RenderMode.TEXTURED:
							gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.get("terrain_mud", 0))
					elif terrain_type == 2:  # Rocher
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
						if render_mode == RenderMode.TEXTURED:
							gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.get("terrain_rock", 0))
					elif terrain_type == 3:  # Corail
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.9, 0.5, 0.5, 1.0])
						if render_mode == RenderMode.TEXTURED:
							gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.get("terrain_coral", 0))
					else:  # Gravier
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.7, 0.65, 0.6, 1.0])
						if render_mode == RenderMode.TEXTURED:
							gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.get("terrain_gravel", 0))
				else:
					if terrain_type == 0:
						gl.glColor3f(0.8, 0.8, 0.6)
					elif terrain_type == 1:
						gl.glColor3f(0.5, 0.4, 0.3)
					elif terrain_type == 2:
						gl.glColor3f(0.5, 0.5, 0.5)
					elif terrain_type == 3:
						gl.glColor3f(0.9, 0.5, 0.5)
					else:
						gl.glColor3f(0.7, 0.65, 0.6)

				self._render_terrain_tile(heights)
				gl.glPopMatrix()

		# Rendu des caractéristiques (obstacles, structures, etc.)
		self._render_terrain_features(render_mode)
		gl.glPopMatrix()

	def _render_terrain_tile(self, heights: List[float]) -> None:
		"""
		Rend un carreau de terrain avec une élévation donnée.
		
		Args:
			heights: Liste des hauteurs aux 4 coins.
		"""
		if "terrain_tile" in self.display_lists:
			gl.glCallList(self.display_lists["terrain_tile"])
		else:
			gl.glBegin(gl.GL_QUADS)
			gl.glNormal3f(0.0, 1.0, 0.0)
			gl.glTexCoord2f(0.0, 0.0); gl.glVertex3f(0.0, 0.0, 0.0)
			gl.glTexCoord2f(1.0, 0.0); gl.glVertex3f(1.0, 0.0, 0.0)
			gl.glTexCoord2f(1.0, 1.0); gl.glVertex3f(1.0, 0.0, 1.0)
			gl.glTexCoord2f(0.0, 1.0); gl.glVertex3f(0.0, 0.0, 1.0)
			gl.glEnd()

	def _render_terrain_features(self, render_mode: RenderMode) -> None:
		"""
		Rend les obstacles et structures spéciales sur le terrain.
		
		Args:
			render_mode: Mode de rendu à utiliser.
		"""
		if not self.world or not hasattr(self.world.terrain, 'obstacles'):
			return

		terrain = self.world.terrain

		# Obstacles (rochers, coraux, évents, etc.)
		for obstacle in terrain.obstacles:
			obstacle_type = obstacle.get("type", "")
			position = obstacle.get("position", (0, 0, 0))
			size = obstacle.get("size", 1.0)
			orientation = obstacle.get("orientation", 0.0)

			gl.glPushMatrix()
			gl.glTranslatef(position[0], position[1], position[2])
			gl.glRotatef(math.degrees(orientation), 0, 1, 0)
			gl.glScalef(size, size, size)

			if render_mode in (RenderMode.SHADED, RenderMode.TEXTURED):
				if obstacle_type == "coral_formation":
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.9, 0.5, 0.5, 1.0])
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.5, 0.3, 0.3, 1.0])
					gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 10.0)
				elif obstacle_type in ("rock", "rock_formation"):
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.3, 0.3, 0.3, 1.0])
					gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 5.0)
				elif obstacle_type == "deep_sea_vent":
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.3, 0.3, 0.3, 1.0])
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.8, 0.3, 0.1, 1.0])
					gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 30.0)
				else:
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.6, 0.6, 0.6, 1.0])
			else:
				if obstacle_type == "coral_formation":
					gl.glColor3f(0.9, 0.5, 0.5)
				elif obstacle_type in ("rock", "rock_formation"):
					gl.glColor3f(0.5, 0.5, 0.5)
				elif obstacle_type == "deep_sea_vent":
					gl.glColor3f(0.3, 0.3, 0.3)
				else:
					gl.glColor3f(0.6, 0.6, 0.6)

			if obstacle_type == "coral_formation":
				self._render_coral(size)
			elif obstacle_type == "rock":
				self._render_rock(size)
			elif obstacle_type == "rock_formation":
				self._render_rock_formation(size)
			elif obstacle_type == "deep_sea_vent":
				self._render_deep_sea_vent(size)
			else:
				quad = glu.gluNewQuadric()
				glu.gluSphere(quad, 0.5, 12, 12)
				glu.gluDeleteQuadric(quad)
			gl.glPopMatrix()

		if hasattr(terrain, 'structures'):
			for structure in terrain.structures:
				structure_type = structure.get("type", "")
				position = structure.get("position", (0, 0, 0))
				size = structure.get("size", 1.0)
				orientation = structure.get("orientation", 0.0)
				properties = structure.get("properties", {})

				gl.glPushMatrix()
				gl.glTranslatef(position[0], position[1], position[2])
				gl.glRotatef(math.degrees(orientation), 0, 1, 0)
				gl.glScalef(size, size, size)

				if render_mode in (RenderMode.SHADED, RenderMode.TEXTURED):
					if structure_type == "reef":
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.8, 0.4, 0.4, 1.0])
					elif structure_type == "shipwreck":
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.6, 0.5, 0.4, 1.0])
					elif structure_type == "underwater_cave":
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.3, 0.3, 0.4, 1.0])
					elif structure_type == "kelp_forest":
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.2, 0.7, 0.3, 1.0])
					elif structure_type == "abyss":
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.1, 0.1, 0.2, 1.0])
					else:
						gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
				else:
					if structure_type == "reef":
						gl.glColor3f(0.8, 0.4, 0.4)
					elif structure_type == "shipwreck":
						gl.glColor3f(0.6, 0.5, 0.4)
					elif structure_type == "underwater_cave":
						gl.glColor3f(0.3, 0.3, 0.4)
					elif structure_type == "kelp_forest":
						gl.glColor3f(0.2, 0.7, 0.3)
					elif structure_type == "abyss":
						gl.glColor3f(0.1, 0.1, 0.2)
					else:
						gl.glColor3f(0.5, 0.5, 0.5)

				if structure_type == "reef":
					self._render_reef(size, properties)
				elif structure_type == "shipwreck":
					self._render_shipwreck(size, properties)
				elif structure_type == "underwater_cave":
					self._render_underwater_cave(size, properties)
				elif structure_type == "kelp_forest":
					self._render_kelp_forest(size, properties)
				elif structure_type == "abyss":
					self._render_abyss(size, properties)
				else:
					quad = glu.gluNewQuadric()
					glu.gluSphere(quad, 0.5, 12, 12)
					glu.gluDeleteQuadric(quad)
				gl.glPopMatrix()

	def _render_coral(self, size: float) -> None:
		"""
		Rend une formation de corail.
		
		Args:
			size: Taille de la formation.
		"""
		num_branches = 5 + int(size * 3)
		max_height = 0.8 * size

		for i in range(num_branches):
			angle = 2 * math.pi * i / num_branches
			offset_x = 0.3 * math.cos(angle)
			offset_z = 0.3 * math.sin(angle)
			branch_height = 0.3 + np.random.random() * max_height
			branch_width = 0.05 + np.random.random() * 0.1

			gl.glPushMatrix()
			gl.glTranslatef(offset_x, 0, offset_z)

			quad = glu.gluNewQuadric()
			glu.gluCylinder(quad, branch_width, branch_width * 0.7, branch_height, 8, 3)

			gl.glTranslatef(0, 0, branch_height)
			glu.gluDisk(quad, 0, branch_width * 1.5, 8, 1)
			glu.gluDeleteQuadric(quad)
			gl.glPopMatrix()

	def _render_rock(self, size: float) -> None:
		"""
		Rend un rocher.
		
		Args:
			size: Taille du rocher.
		"""
		gl.glPushMatrix()
		gl.glScalef(1.0, 0.7, 0.9)
		quad = glu.gluNewQuadric()
		glu.gluSphere(quad, 0.5, 12, 12)
		glu.gluDeleteQuadric(quad)
		gl.glPopMatrix()

	def _render_rock_formation(self, size: float) -> None:
		"""
		Rend une formation rocheuse.
		
		Args:
			size: Taille de la formation.
		"""
		num_rocks = 3 + int(size * 2)
		for i in range(num_rocks):
			offset_x = (np.random.random() - 0.5) * 0.6
			offset_z = (np.random.random() - 0.5) * 0.6
			rock_size = 0.2 + np.random.random() * 0.3
			height = np.random.random() * 0.2

			gl.glPushMatrix()
			gl.glTranslatef(offset_x, height, offset_z)
			gl.glScalef(rock_size, rock_size, rock_size)
			gl.glScalef(np.random.random() * 0.4 + 0.8,
						np.random.random() * 0.4 + 0.6,
						np.random.random() * 0.4 + 0.8)
			quad = glu.gluNewQuadric()
			glu.gluSphere(quad, 0.5, 10, 10)
			glu.gluDeleteQuadric(quad)
			gl.glPopMatrix()

	def _render_deep_sea_vent(self, size: float) -> None:
		"""
		Rend un évent hydrothermal des fonds marins.
		
		Args:
			size: Taille de l'évent.
		"""
		gl.glPushMatrix()
		quad = glu.gluNewQuadric()
		glu.gluCylinder(quad, 0.6, 0.2, 0.8, 12, 3)
		gl.glTranslatef(0, 0, 0.8)
		glu.gluDisk(quad, 0, 0.2, 12, 1)
		glu.gluDeleteQuadric(quad)

		if gl.glGetIntegerv(gl.GL_POLYGON_MODE)[1] != gl.GL_LINE:
			gl.glEnable(gl.GL_BLEND)
			gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
			gl.glColor4f(0.7, 0.7, 0.7, 0.3)
			for i in range(5):
				vent_height = 0.2 * (i + 1)
				vent_width = 0.1 + 0.05 * i
				gl.glBegin(gl.GL_TRIANGLE_FAN)
				gl.glVertex3f(0, 0, vent_height)
				for j in range(13):
					angle = j * 2 * math.pi / 12
					gl.glVertex3f(vent_width * math.cos(angle), vent_width * math.sin(angle), vent_height)
				gl.glEnd()
			gl.glDisable(gl.GL_BLEND)
		gl.glPopMatrix()

	def _render_reef(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Rend un récif corallien.
		
		Args:
			size: Taille du récif.
			properties: Propriétés spécifiques du récif.
		"""
		coral_coverage = properties.get("coral_coverage", 0.7)
		biodiversity = properties.get("biodiversity", 0.8)
		width = size
		length = size
		height = size * 0.3

		gl.glPushMatrix()
		gl.glScalef(width, height, length)
		gl.glBegin(gl.GL_TRIANGLE_FAN)
		gl.glNormal3f(0, 1, 0)
		gl.glVertex3f(0, 0.5, 0)
		for i in range(17):
			angle = i * math.pi / 8
			x = math.cos(angle)
			z = math.sin(angle)
			y = 0.3 + 0.2 * math.sin(angle * 3)
			gl.glVertex3f(0.5 * x, y, 0.5 * z)
		gl.glEnd()
		gl.glPopMatrix()

		num_corals = int(20 * coral_coverage * size)
		coral_types = int(5 * biodiversity)
		for i in range(num_corals):
			angle = np.random.random() * 2 * math.pi
			radius = np.random.random() * 0.4 * size
			coral_x = radius * math.cos(angle)
			coral_z = radius * math.sin(angle)
			coral_size = 0.1 + np.random.random() * 0.2 * size

			gl.glPushMatrix()
			gl.glTranslatef(coral_x, 0, coral_z)
			gl.glScalef(coral_size, coral_size, coral_size)
			coral_type = i % coral_types
			if coral_type == 0:
				gl.glColor3f(0.9, 0.5, 0.5)
			elif coral_type == 1:
				gl.glColor3f(0.9, 0.7, 0.4)
			elif coral_type == 2:
				gl.glColor3f(0.8, 0.8, 0.4)
			elif coral_type == 3:
				gl.glColor3f(0.4, 0.7, 0.9)
			else:
				gl.glColor3f(0.5, 0.9, 0.5)
			if coral_type < 2:
				self._render_coral(coral_size)
			elif coral_type < 4:
				quad = glu.gluNewQuadric()
				glu.gluCylinder(quad, 0.1, 0.5, 0.3, 8, 2)
				gl.glTranslatef(0, 0, 0.3)
				glu.gluDisk(quad, 0, 0.5, 8, 1)
				glu.gluDeleteQuadric(quad)
			else:
				self._render_branching_coral(coral_size)
			gl.glPopMatrix()

	def _render_branching_coral(self, size: float) -> None:
		"""
		Rend un corail branchu.
		
		Args:
			size: Taille du corail.
		"""
		num_branches = 4 + int(size * 5)
		quad = glu.gluNewQuadric()
		glu.gluCylinder(quad, 0.1, 0.05, 0.5, 6, 2)
		for i in range(num_branches):
			angle = 2 * math.pi * i / num_branches
			branch_height = 0.2 + 0.3 * np.random.random()
			gl.glPushMatrix()
			gl.glTranslatef(0, 0, 0.2 + 0.3 * np.random.random())
			gl.glRotatef(30 + 20 * np.random.random(), math.cos(angle), math.sin(angle), 0)
			glu.gluCylinder(quad, 0.03, 0.01, branch_height, 4, 1)
			gl.glPopMatrix()
		glu.gluDeleteQuadric(quad)

	def _render_shipwreck(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Rend une épave de navire.
		
		Args:
			size: Taille de l'épave.
			properties: Propriétés spécifiques de l'épave.
		"""
		age = properties.get("age", 50)
		decay = properties.get("decay", 0.5)
		length = size
		width = size * 0.3
		height = size * 0.2
		decay_factor = min(1.0, age / 100.0) * decay

		gl.glPushMatrix()
		gl.glRotatef(20 * np.random.random(), 1, 0, 0)
		gl.glRotatef(15 * np.random.random(), 0, 0, 1)
		gl.glScalef(width, height, length)

		gl.glBegin(gl.GL_QUADS)
		gl.glNormal3f(-1, 0, 0)
		gl.glVertex3f(-0.5, 0, -0.5)
		gl.glVertex3f(-0.5, 0, 0.5)
		gl.glVertex3f(-0.5, 0.5, 0.4)
		gl.glVertex3f(-0.5, 0.5, -0.4)

		gl.glNormal3f(1, 0, 0)
		gl.glVertex3f(0.5, 0, -0.5)
		gl.glVertex3f(0.5, 0.5, -0.4)
		gl.glVertex3f(0.5, 0.5, 0.4)
		gl.glVertex3f(0.5, 0, 0.5)

		gl.glNormal3f(0, 0, -1)
		gl.glVertex3f(-0.5, 0, -0.5)
		gl.glVertex3f(-0.5, 0.5, -0.4)
		gl.glVertex3f(0.5, 0.5, -0.4)
		gl.glVertex3f(0.5, 0, -0.5)

		gl.glNormal3f(0, 0, 1)
		gl.glVertex3f(-0.5, 0, 0.5)
		gl.glVertex3f(0.5, 0, 0.5)
		gl.glVertex3f(0.5, 0.5, 0.4)
		gl.glVertex3f(-0.5, 0.5, 0.4)

		gl.glNormal3f(0, 1, 0)
		gl.glVertex3f(-0.5, 0.5, -0.4)
		gl.glVertex3f(-0.5, 0.5, 0.4)
		gl.glVertex3f(0.5, 0.5, 0.4)
		gl.glVertex3f(0.5, 0.5, -0.4)
		gl.glEnd()

		if decay_factor < 0.7:
			gl.glPushMatrix()
			gl.glTranslatef(0, 0.5, 0)
			gl.glScalef(0.6, 0.3, 0.4)
			gl.glBegin(gl.GL_QUADS)
			gl.glNormal3f(-1, 0, 0)
			gl.glVertex3f(-0.5, 0, -0.5)
			gl.glVertex3f(-0.5, 0, 0.5)
			gl.glVertex3f(-0.5, 1, 0.5)
			gl.glVertex3f(-0.5, 1, -0.5)
			gl.glNormal3f(1, 0, 0)
			gl.glVertex3f(0.5, 0, -0.5)
			gl.glVertex3f(0.5, 1, -0.5)
			gl.glVertex3f(0.5, 1, 0.5)
			gl.glVertex3f(0.5, 0, 0.5)
			gl.glNormal3f(0, 0, -1)
			gl.glVertex3f(-0.5, 0, -0.5)
			gl.glVertex3f(-0.5, 1, -0.5)
			gl.glVertex3f(0.5, 1, -0.5)
			gl.glVertex3f(0.5, 0, -0.5)
			gl.glNormal3f(0, 0, 1)
			gl.glVertex3f(-0.5, 0, 0.5)
			gl.glVertex3f(0.5, 0, 0.5)
			gl.glVertex3f(0.5, 1, 0.5)
			gl.glVertex3f(-0.5, 1, 0.5)
			gl.glNormal3f(0, 1, 0)
			gl.glVertex3f(-0.5, 1, -0.5)
			gl.glVertex3f(-0.5, 1, 0.5)
			gl.glVertex3f(0.5, 1, 0.5)
			gl.glVertex3f(0.5, 1, -0.5)
			gl.glEnd()
			gl.glPopMatrix()

		if decay_factor > 0.3:
			num_holes = int(5 * decay_factor)
			for _ in range(num_holes):
				hole_x = (np.random.random() - 0.5) * 0.8
				hole_z = (np.random.random() - 0.5) * 0.8
				hole_size = 0.05 + np.random.random() * 0.1
				gl.glPushMatrix()
				gl.glTranslatef(hole_x, 0.25, hole_z)
				gl.glColor3f(0.2, 0.2, 0.2)
				quad = glu.gluNewQuadric()
				glu.gluDisk(quad, 0, hole_size, 8, 1)
				glu.gluDeleteQuadric(quad)
				gl.glPopMatrix()
		gl.glPopMatrix()

	def _render_underwater_cave(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Rend une grotte sous-marine.
		
		Args:
			size: Taille de la grotte.
			properties: Propriétés spécifiques de la grotte.
		"""
		cave_depth = properties.get("depth", 20) * 0.05
		complexity = properties.get("complexity", 0.5)

		gl.glPushMatrix()
		entrance_width = size * 0.5
		entrance_height = size * 0.3
		gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.2, 0.2, 0.25, 1.0])
		gl.glBegin(gl.GL_TRIANGLE_FAN)
		gl.glVertex3f(0, 0, -0.2)
		num_segments = 12 + int(8 * complexity)
		for i in range(num_segments + 1):
			angle = 2 * math.pi * i / num_segments
			radius_var = 0.3 * math.sin(angle * 3) * complexity
			x = entrance_width * math.cos(angle) * (0.8 + radius_var)
			y = entrance_height * math.sin(angle) * (0.8 + radius_var)
			gl.glVertex3f(x, y, 0)
		gl.glEnd()

		for depth in np.linspace(0, cave_depth, 5):
			scale = 1.0 - depth * 0.15
			gl.glPushMatrix()
			gl.glTranslatef(0, 0, -depth)
			gl.glScalef(scale, scale, 1.0)
			gl.glBegin(gl.GL_LINE_LOOP)
			for i in range(num_segments):
				angle = 2 * math.pi * i / num_segments
				radius_var = 0.3 * math.sin(angle * 3 + depth * 2) * complexity
				x = entrance_width * math.cos(angle) * (0.8 + radius_var)
				y = entrance_height * math.sin(angle) * (0.8 + radius_var)
				gl.glVertex3f(x, y, 0)
			gl.glEnd()
			gl.glPopMatrix()
		gl.glPopMatrix()

	def _render_kelp_forest(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Rend une forêt de kelp (algues géantes).
		
		Args:
			size: Taille de la forêt de kelp.
			properties: Propriétés spécifiques de la forêt.
		"""
		density = properties.get("density", 0.7)
		kelp_height = properties.get("height", 10) * 0.1
		num_kelp = int(20 * density * size)

		for i in range(num_kelp):
			pos_x = (np.random.random() - 0.5) * size
			pos_z = (np.random.random() - 0.5) * size
			height = kelp_height * (0.7 + 0.6 * np.random.random())
			width = 0.05 + 0.03 * np.random.random()

			gl.glPushMatrix()
			gl.glTranslatef(pos_x, 0, pos_z)

			quad = glu.gluNewQuadric()
			glu.gluCylinder(quad, width, width * 0.7, height, 8, 3)
			glu.gluDeleteQuadric(quad)

			num_leaves = 3 + int(height * 2)
			for j in range(num_leaves):
				leaf_height = height * j / num_leaves
				leaf_size = 0.1 + 0.1 * j / num_leaves
				leaf_angle = j * 137.5
				gl.glPushMatrix()
				gl.glTranslatef(0, 0, leaf_height)
				gl.glRotatef(leaf_angle, 0, 0, 1)
				gl.glBegin(gl.GL_TRIANGLES)
				gl.glNormal3f(0, 0, 1)
				gl.glVertex3f(0, 0, 0)
				gl.glVertex3f(leaf_size, 0, 0)
				gl.glVertex3f(leaf_size * 0.5, leaf_size * 2, 0)
				gl.glEnd()
				gl.glPopMatrix()
			gl.glPopMatrix()

	def _render_abyss(self, size: float, properties: Dict[str, Any]) -> None:
		"""
		Rend une fosse abyssale.
		
		Args:
			size: Taille de la fosse.
			properties: Propriétés spécifiques de la fosse.
		"""
		pressure = properties.get("pressure", 200)
		thermal_activity = properties.get("thermal_activity", 0.5)
		gl.glPushMatrix()
		depth = size * 0.2 * (pressure / 100)
		gl.glBegin(gl.GL_TRIANGLE_FAN)
		gl.glVertex3f(0, -depth, 0)
		num_segments = 16
		for i in range(num_segments + 1):
			angle = 2 * math.pi * i / num_segments
			x = size * 0.5 * math.cos(angle)
			z = size * 0.5 * math.sin(angle)
			gl.glVertex3f(x, 0, z)
		gl.glEnd()

		if thermal_activity > 0.2:
			num_vents = int(5 * thermal_activity)
			for i in range(num_vents):
				vent_radius = np.random.random() * 0.3 * size
				vent_angle = np.random.random() * 2 * math.pi
				vent_x = vent_radius * math.cos(vent_angle)
				vent_z = vent_radius * math.sin(vent_angle)
				vent_size = 0.1 + np.random.random() * 0.1
				gl.glPushMatrix()
				gl.glTranslatef(vent_x, -depth + 0.05, vent_z)
				gl.glScalef(vent_size, vent_size, vent_size)
				current_color = [0.0, 0.0, 0.0, 0.0]
				gl.glGetFloatv(gl.GL_CURRENT_COLOR, current_color)
				gl.glColor3f(0.7, 0.3, 0.1)
				self._render_deep_sea_vent(vent_size)
				gl.glColor3f(current_color[0], current_color[1], current_color[2])
				gl.glPopMatrix()
		gl.glPopMatrix()

	def _render_water(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu de la surface et du volume d'eau.
		
		Args:
			render_mode: Mode de rendu à utiliser.
		"""
		if not self.world:
			return

		world_size = self.world.size
		gl.glPushMatrix()

		if render_mode in (RenderMode.SHADED, RenderMode.TEXTURED):
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT, [0.1, 0.3, 0.5, self.water_transparency])
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_DIFFUSE, [0.2, 0.4, 0.6, self.water_transparency])
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_SPECULAR, [0.8, 0.9, 1.0, 1.0])
			gl.glMaterialf(gl.GL_FRONT_AND_BACK, gl.GL_SHININESS, 50.0)
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.get("water", 0))
		else:
			gl.glColor4f(0.2, 0.4, 0.6, self.water_transparency)

		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

		gl.glTranslatef(world_size[0] / 2, 0, world_size[2] / 2)
		gl.glScalef(world_size[0], 1.0, world_size[2])

		if render_mode != RenderMode.WIREFRAME:
			current_time = self.world.currentTime if self.world.currentTime else time.time()
			gl.glMatrixMode(gl.GL_TEXTURE)
			gl.glPushMatrix()
			gl.glTranslatef(0.5 * math.sin(current_time * self.water_wave_frequency),
						   0.5 * math.cos(current_time * self.water_wave_frequency * 0.7),
						   0)
			gl.glMatrixMode(gl.GL_MODELVIEW)

		if "water_surface" in self.display_lists:
			gl.glCallList(self.display_lists["water_surface"])
		else:
			self._render_water_surface(self.water_detail)

		if render_mode != RenderMode.WIREFRAME:
			gl.glMatrixMode(gl.GL_TEXTURE)
			gl.glPopMatrix()
			gl.glMatrixMode(gl.GL_MODELVIEW)

		gl.glDisable(gl.GL_BLEND)
		gl.glPopMatrix()

		if self.water_transparency < 1.0 and render_mode != RenderMode.WIREFRAME:
			self._render_water_volume(render_mode)

	def _render_water_surface(self, detail: int) -> None:
		"""
		Rend la surface de l'eau avec le niveau de détail spécifié.
		
		Args:
			detail: Niveau de détail.
		"""
		gl.glBegin(gl.GL_QUADS)
		size = 1.0
		subdivs = detail
		step = size / subdivs
		half_size = size / 2
		for i in range(subdivs):
			for j in range(subdivs):
				x0 = -half_size + i * step
				z0 = -half_size + j * step
				x1 = x0 + step
				z1 = z0 + step
				tx0 = i / subdivs * 5
				tz0 = j / subdivs * 5
				tx1 = (i + 1) / subdivs * 5
				tz1 = (j + 1) / subdivs * 5
				gl.glNormal3f(0.0, 1.0, 0.0)
				gl.glTexCoord2f(tx0, tz0); gl.glVertex3f(x0, 0.0, z0)
				gl.glTexCoord2f(tx1, tz0); gl.glVertex3f(x1, 0.0, z0)
				gl.glTexCoord2f(tx1, tz1); gl.glVertex3f(x1, 0.0, z1)
				gl.glTexCoord2f(tx0, tz1); gl.glVertex3f(x0, 0.0, z1)
		gl.glEnd()

	def _render_water_volume(self, render_mode: RenderMode) -> None:
		"""
		Rend un volume d'eau pour simuler la profondeur par effet de brouillard.
		
		Args:
			render_mode: Mode de rendu à utiliser.
		"""
		if not self.world:
			return
		world_size = self.world.size
		fog_color = [0.1, 0.2, 0.4, 0.02]
		gl.glEnable(gl.GL_FOG)
		gl.glFogi(gl.GL_FOG_MODE, gl.GL_EXP2)
		gl.glFogfv(gl.GL_FOG_COLOR, fog_color)
		gl.glFogf(gl.GL_FOG_DENSITY, 0.01)
		gl.glHint(gl.GL_FOG_HINT, gl.GL_NICEST)
		gl.glDisable(gl.GL_FOG)

	def _render_resources(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu des ressources (nourriture, etc.).
		
		Args:
			render_mode: Mode de rendu à utiliser.
		"""
		if not self.world:
			return

		for food in self.world.foodResources:
			if food.isConsumed:
				continue
			position = food.position
			size = food.size
			food_type = food.foodType

			gl.glPushMatrix()
			gl.glTranslatef(position[0], position[1], position[2])
			if render_mode in (RenderMode.SHADED, RenderMode.TEXTURED):
				if food_type == "algae":
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.2, 0.7, 0.2, 1.0])
				elif food_type == "plankton":
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.7, 0.8, 0.4, 1.0])
				elif food_type == "small_fish":
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.7, 0.7, 0.9, 1.0])
				else:
					gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.5, 0.4, 0.3, 1.0])
				if render_mode == RenderMode.TEXTURED:
					texture_name = f"resource_{food_type}"
					if texture_name in self.textures:
						gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures[texture_name])
			else:
				if food_type == "algae":
					gl.glColor3f(0.2, 0.7, 0.2)
				elif food_type == "plankton":
					gl.glColor3f(0.7, 0.8, 0.4)
				elif food_type == "small_fish":
					gl.glColor3f(0.7, 0.7, 0.9)
				else:
					gl.glColor3f(0.5, 0.4, 0.3)

			if food_type == "algae":
				self._render_algae(size)
			elif food_type == "plankton":
				self._render_plankton(size)
			elif food_type == "small_fish":
				self._render_small_fish(size)
			else:
				self._render_detritus(size)

			gl.glPopMatrix()

	def _render_algae(self, size: float) -> None:
		"""
		Rend une algue.
		
		Args:
			size: Taille de l'algue.
		"""
		num_leaves = 3 + int(size * 5)
		for i in range(num_leaves):
			angle = i * 137.5
			height = size * (0.5 + 0.5 * np.random.random())
			gl.glPushMatrix()
			gl.glRotatef(angle, 0, 0, 1)
			gl.glRotatef(70, 1, 0, 0)
			gl.glBegin(gl.GL_TRIANGLE_FAN)
			gl.glVertex3f(0, 0, 0)
			leaf_width = 0.2 * size
			for j in range(13):
				angle_j = j * 2 * math.pi / 12
				x = leaf_width * math.sin(angle_j)
				y = height * math.cos(angle_j)
				gl.glVertex3f(x, y, 0)
			gl.glEnd()
			gl.glPopMatrix()

	def _render_plankton(self, size: float) -> None:
		"""
		Rend du plancton.
		
		Args:
			size: Taille du plancton.
		"""
		gl.glPointSize(2.0)
		gl.glBegin(gl.GL_POINTS)
		num_particles = 20 + int(size * 50)
		for _ in range(num_particles):
			theta = np.random.random() * 2 * math.pi
			phi = np.random.random() * math.pi
			radius = size * np.random.random()
			x = radius * math.sin(phi) * math.cos(theta)
			y = radius * math.sin(phi) * math.sin(theta)
			z = radius * math.cos(phi)
			gl.glVertex3f(x, y, z)
		gl.glEnd()
		gl.glPointSize(1.0)

	def _render_small_fish(self, size: float) -> None:
		"""
		Rend un petit poisson.
		
		Args:
			size: Taille du poisson.
		"""
		gl.glPushMatrix()
		quad = glu.gluNewQuadric()
		gl.glScalef(1.0, 0.5, 0.2)
		glu.gluSphere(quad, size * 0.5, 8, 8)
		gl.glPushMatrix()
		gl.glTranslatef(-size * 0.5, 0, 0)
		gl.glRotatef(90, 0, 1, 0)
		gl.glScalef(0.2, 0.4, 1.0)
		glu.gluCylinder(quad, size * 0.5, 0, size * 0.4, 8, 1)
		gl.glPopMatrix()
		glu.gluDeleteQuadric(quad)
		gl.glPopMatrix()

	def _render_detritus(self, size: float) -> None:
		"""
		Rend des détritus organiques.
		
		Args:
			size: Taille des détritus.
		"""
		num_pieces = 3 + int(size * 5)
		for i in range(num_pieces):
			piece_size = size * (0.3 + 0.7 * np.random.random())
			gl.glPushMatrix()
			offset_x = (np.random.random() - 0.5) * size
			offset_y = (np.random.random() - 0.5) * size * 0.5
			offset_z = (np.random.random() - 0.5) * size
			gl.glTranslatef(offset_x, offset_y, offset_z)
			gl.glRotatef(np.random.random() * 360, 1, 0, 0)
			gl.glRotatef(np.random.random() * 360, 0, 1, 0)
			gl.glRotatef(np.random.random() * 360, 0, 0, 1)
			gl.glScalef(piece_size, piece_size * 0.2, piece_size * 0.6)
			quad = glu.gluNewQuadric()
			glu.gluSphere(quad, 0.5, 6, 4)
			glu.gluDeleteQuadric(quad)
			gl.glPopMatrix()

	def _render_zones(self, render_mode: RenderMode) -> None:
		"""
		Effectue le rendu des zones environnementales.
		
		Args:
			render_mode: Mode de rendu à utiliser.
		"""
		if not self.world or not hasattr(self.world, 'zones'):
			return

		for zone in self.world.zones:
			if render_mode in (RenderMode.WIREFRAME, RenderMode.TEXTURED):
				self._render_zone_boundary(zone, render_mode)
			if (render_mode in (RenderMode.SOLID, RenderMode.SHADED)) and render_mode != RenderMode.WIREFRAME:
				self._render_zone_volume(zone, render_mode)
			self._render_zone_features(zone, render_mode)

	def _render_zone_boundary(self, zone: EnvironmentalZone, render_mode: RenderMode) -> None:
		"""
		Rend les limites d'une zone environnementale.
		
		Args:
			zone: Zone à rendre.
			render_mode: Mode de rendu à utiliser.
		"""
		bounds = zone.bounds
		min_x, max_x = bounds[0]
		min_y, max_y = bounds[1]
		min_z, max_z = bounds[2]

		if zone.name == "Surface Zone":
			color = (0.4, 0.6, 0.9, 0.3)
		elif zone.name == "Intermediate Zone":
			color = (0.2, 0.4, 0.7, 0.3)
		elif zone.name == "Deep Zone":
			color = (0.1, 0.2, 0.5, 0.3)
		elif zone.name == "Reef Zone":
			color = (0.8, 0.5, 0.5, 0.3)
		elif zone.name == "Current Zone":
			color = (0.5, 0.7, 0.9, 0.3)
		else:
			color = (0.5, 0.5, 0.5, 0.3)

		if render_mode in (RenderMode.SHADED, RenderMode.TEXTURED):
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE, [*color])
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures.get("zone_boundary", 0))
		else:
			gl.glColor4f(*color)

		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		gl.glBegin(gl.GL_LINES)
		# Arêtes horizontales inférieures
		gl.glVertex3f(min_x, min_y, min_z); gl.glVertex3f(max_x, min_y, min_z)
		gl.glVertex3f(max_x, min_y, min_z); gl.glVertex3f(max_x, min_y, max_z)
		gl.glVertex3f(max_x, min_y, max_z); gl.glVertex3f(min_x, min_y, max_z)
		gl.glVertex3f(min_x, min_y, max_z); gl.glVertex3f(min_x, min_y, min_z)
		# Arêtes horizontales supérieures
		gl.glVertex3f(min_x, max_y, min_z); gl.glVertex3f(max_x, max_y, min_z)
		gl.glVertex3f(max_x, max_y, min_z); gl.glVertex3f(max_x, max_y, max_z)
		gl.glVertex3f(max_x, max_y, max_z); gl.glVertex3f(min_x, max_y, max_z)
		gl.glVertex3f(min_x, max_y, max_z); gl.glVertex3f(min_x, max_y, min_z)
		# Arêtes verticales
		gl.glVertex3f(min_x, min_y, min_z); gl.glVertex3f(min_x, max_y, min_z)
		gl.glVertex3f(max_x, min_y, min_z); gl.glVertex3f(max_x, max_y, min_z)
		gl.glVertex3f(max_x, min_y, max_z); gl.glVertex3f(max_x, max_y, max_z)
		gl.glVertex3f(min_x, min_y, max_z); gl.glVertex3f(min_x, max_y, max_z)
		gl.glEnd()
		gl.glDisable(gl.GL_BLEND)

	def _render_zone_volume(self, zone: EnvironmentalZone, render_mode: RenderMode) -> None:
		"""
		Rend le volume d'une zone environnementale avec transparence.
		
		Args:
			zone: Zone à rendre.
			render_mode: Mode de rendu à utiliser.
		"""
		bounds = zone.bounds
		min_x, max_x = bounds[0]
		min_y, max_y = bounds[1]
		min_z, max_z = bounds[2]

		if zone.name == "Surface Zone":
			color = (0.4, 0.6, 0.9, 0.05)
		elif zone.name == "Intermediate Zone":
			color = (0.2, 0.4, 0.7, 0.05)
		elif zone.name == "Deep Zone":
			color = (0.1, 0.2, 0.5, 0.05)
		elif zone.name == "Reef Zone":
			color = (0.8, 0.5, 0.5, 0.05)
		elif zone.name == "Current Zone":
			color = (0.5, 0.7, 0.9, 0.05)
		else:
			color = (0.5, 0.5, 0.5, 0.05)

		if render_mode == RenderMode.SHADED:
			gl.glMaterialfv(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE, [*color])
		else:
			gl.glColor4f(*color)

		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		gl.glBegin(gl.GL_QUADS)
		# Face avant
		gl.glVertex3f(min_x, min_y, min_z)
		gl.glVertex3f(max_x, min_y, min_z)
		gl.glVertex3f(max_x, max_y, min_z)
		gl.glVertex3f(min_x, max_y, min_z)
		# Face arrière
		gl.glVertex3f(min_x, min_y, max_z)
		gl.glVertex3f(min_x, max_y, max_z)
		gl.glVertex3f(max_x, max_y, max_z)
		gl.glVertex3f(max_x, min_y, max_z)
		# Face gauche
		gl.glVertex3f(min_x, min_y, min_z)
		gl.glVertex3f(min_x, max_y, min_z)
		gl.glVertex3f(min_x, max_y, max_z)
		gl.glVertex3f(min_x, min_y, max_z)
		# Face droite
		gl.glVertex3f(max_x, min_y, min_z)
		gl.glVertex3f(max_x, max_y, min_z)
		gl.glVertex3f(max_x, max_y, max_z)
		gl.glVertex3f(max_x, min_y, max_z)
		gl.glEnd()
		gl.glDisable(gl.GL_BLEND)

	def _render_zone_features(self, zone: EnvironmentalZone, render_mode: RenderMode) -> None:
		"""
		Rend les caractéristiques spécifiques d'une zone environnementale.
		
		Args:
			zone: Zone à rendre.
			render_mode: Mode de rendu à utiliser.
		"""
		if zone.name == "Current Zone":
			self._render_current_indicators(zone, render_mode)
		elif zone.name == "Reef Zone":
			self._render_reef_decorations(zone, render_mode)

	def _render_current_indicators(self, zone: EnvironmentalZone, render_mode: RenderMode) -> None:
		"""
		Rend des indicateurs de courant pour une zone.
		
		Args:
			zone: Zone à rendre.
			render_mode: Mode de rendu à utiliser.
		"""
		bounds = zone.bounds
		min_x, max_x = bounds[0]
		min_y, max_y = bounds[1]
		min_z, max_z = bounds[2]
		center_x = (min_x + max_x) / 2
		center_y = (min_y + max_y) / 2
		center_z = (min_z + max_z) / 2
		width = max_x - min_x
		height = max_y - min_y
		depth = max_z - min_z
		arrow_scale = min(width, depth) * 0.05
		num_arrows_x = max(2, int(width / 20))
		num_arrows_y = max(2, int(height / 20))
		num_arrows_z = max(2, int(depth / 20))
		current_dir = zone.currentDirection
		current_mag = zone.currentMagnitude

		for i in range(num_arrows_x):
			for j in range(num_arrows_y):
				for k in range(num_arrows_z):
					x = min_x + width * (i + 0.5) / num_arrows_x
					y = min_y + height * (j + 0.5) / num_arrows_y
					z = min_z + depth * (k + 0.5) / num_arrows_z
					gl.glColor4f(0.0, 0.7, 0.9, 0.7)
					gl.glPushMatrix()
					gl.glTranslatef(x, y, z)
					yaw = math.atan2(current_dir[0], current_dir[2]) * 180 / math.pi
					horizontal_length = math.sqrt(current_dir[0]**2 + current_dir[2]**2)
					pitch = math.atan2(current_dir[1], horizontal_length) * 180 / math.pi
					gl.glRotatef(-yaw, 0, 1, 0)
					gl.glRotatef(pitch, 1, 0, 0)
					self._render_arrow(arrow_scale * current_mag, render_mode)
					gl.glPopMatrix()

	def _render_arrow(self, length: float, render_mode: RenderMode) -> None:
		"""
		Rend une flèche simple.
		
		Args:
			length: Longueur de la flèche.
			render_mode: Mode de rendu à utiliser.
		"""
		shaft_radius = length * 0.05
		head_radius = shaft_radius * 2.5
		shaft_length = length * 0.7
		head_length = length * 0.3
		quad = glu.gluNewQuadric()
		glu.gluCylinder(quad, shaft_radius, shaft_radius, shaft_length, 8, 1)
		gl.glPushMatrix()
		gl.glTranslatef(0, 0, shaft_length)
		glu.gluCylinder(quad, head_radius, 0, head_length, 8, 1)
		gl.glPopMatrix()
		glu.gluDeleteQuadric(quad)

	def _render_reef_decorations(self, zone: EnvironmentalZone, render_mode: RenderMode) -> None:
		"""
		Rend des éléments décoratifs de récif dans une zone.
		
		Args:
			zone: Zone à rendre.
			render_mode: Mode de rendu à utiliser.
		"""
		bounds = zone.bounds
		min_x, max_x = bounds[0]
		min_y, max_y = bounds[1]
		min_z, max_z = bounds[2]
		width = max_x - min_x
		depth = max_z - min_z
		num_decorations = int(width * depth / 1000)
		properties = {"coral_coverage": 0.8, "biodiversity": 0.9}

		for _ in range(num_decorations):
			x = min_x + width * np.random.random()
			z = min_z + depth * np.random.random()
			size = 0.5 + np.random.random() * 2.0
			gl.glPushMatrix()
			gl.glTranslatef(x, min_y, z)
			if np.random.random() < 0.3:
				self._render_reef(size, properties)
			else:
				self._render_coral(size)
			gl.glPopMatrix()

	def _render_grid(self) -> None:
		"""
		Effectue le rendu d'une grille de référence.
		"""
		if "reference_grid" in self.display_lists:
			gl.glColor3f(0.5, 0.5, 0.5)
			gl.glCallList(self.display_lists["reference_grid"])
		else:
			gl.glColor3f(0.5, 0.5, 0.5)
			gl.glBegin(gl.GL_LINES)
			grid_size = 100.0
			grid_step = 10.0
			for i in range(int(-grid_size), int(grid_size) + 1, int(grid_step)):
				gl.glVertex3f(i, 0, -grid_size)
				gl.glVertex3f(i, 0, grid_size)
			for i in range(int(-grid_size), int(grid_size) + 1, int(grid_step)):
				gl.glVertex3f(-grid_size, 0, i)
				gl.glVertex3f(grid_size, 0, i)
			gl.glEnd()

	def set_render_options(self,
						   show_water: bool = True,
						   show_terrain: bool = True,
						   show_resources: bool = True,
						   show_zones: bool = True,
						   show_grid: bool = False,
						   water_transparency: float = 0.7,
						   water_wave_amplitude: float = 0.2,
						   water_wave_frequency: float = 0.5,
						   terrain_detail: int = 64,
						   water_detail: int = 32) -> None:
		"""
		Configure les options de rendu de l'environnement.
		
		Args:
			show_water: Afficher l'eau.
			show_terrain: Afficher le terrain.
			show_resources: Afficher les ressources.
			show_zones: Afficher les zones environnementales.
			show_grid: Afficher la grille de référence.
			water_transparency: Transparence de l'eau (0-1).
			water_wave_amplitude: Amplitude des vagues.
			water_wave_frequency: Fréquence des vagues.
			terrain_detail: Niveau de détail du terrain.
			water_detail: Niveau de détail de l'eau.
		"""
		self.show_water = show_water
		self.show_terrain = show_terrain
		self.show_resources = show_resources
		self.show_zones = show_zones
		self.show_grid = show_grid
		self.water_transparency = water_transparency
		self.water_wave_amplitude = water_wave_amplitude
		self.water_wave_frequency = water_wave_frequency
		self.terrain_detail = terrain_detail
		self.water_detail = water_detail

	def cleanup(self) -> None:
		"""
		Nettoie les ressources OpenGL (textures, listes d'affichage).
		"""
		try:
			if self.textures:
				texture_ids = list(self.textures.values())
				gl.glDeleteTextures(texture_ids)
				self.textures = {}
			if self.display_lists:
				for list_id in self.display_lists.values():
					gl.glDeleteLists(list_id, 1)
				self.display_lists = {}
			self.logger.info("Ressources du EnvironmentRenderer nettoyées", module="visualization")
		except Exception as e:
			self.logger.error(f"Erreur lors du nettoyage des ressources: {str(e)}", 
							  module="visualization", exc_info=True)
