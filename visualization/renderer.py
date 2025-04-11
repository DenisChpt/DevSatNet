import pygame
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import threading
import os
import sys
from enum import Enum

# Ajouter le répertoire parent au chemin d'importation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.environment.marine_world import MarineWorld
from utils.logger import get_logger
from utils.timer import Timer

# Importer les renderers spécifiques
from visualization.creature_renderer import CreatureRenderer
from visualization.environment_renderer import EnvironmentRenderer


class RenderMode(Enum):
	"""Modes de rendu disponibles pour la visualisation."""
	WIREFRAME = 0    # Affichage en fil de fer (contours)
	SOLID = 1        # Affichage solide (surfaces pleines)
	TEXTURED = 2     # Affichage avec textures
	SHADED = 3       # Affichage ombré (éclairage)


class ViewMode(Enum):
	"""Modes de vue pour la caméra."""
	PERSPECTIVE = 0   # Vue en perspective
	ORTHOGRAPHIC = 1  # Vue orthographique
	TOP_DOWN = 2      # Vue de dessus
	SIDE = 3          # Vue latérale
	FOLLOW = 4        # Suivre une créature


class Renderer:
	"""
	Classe principale de rendu pour la simulation.
	Gère l'initialisation de la fenêtre, les entrées utilisateur et le rendu
	des différents éléments de la simulation.
	"""
	
	def __init__(
		self,
		width: int = 1280,
		height: int = 720,
		title: str = "Marine Evolution Simulation",
		render_mode: RenderMode = RenderMode.SHADED,
		view_mode: ViewMode = ViewMode.PERSPECTIVE,
		fps_limit: int = 60,
		background_color: Tuple[int, int, int] = (10, 30, 70),
		show_debug_info: bool = True,
		antialiasing: bool = True,
		msaa_samples: int = 4
	) -> None:
		"""
		Initialise le renderer.
		
		Args:
			width: Largeur de la fenêtre en pixels
			height: Hauteur de la fenêtre en pixels
			title: Titre de la fenêtre
			render_mode: Mode de rendu initial
			view_mode: Mode de vue initial
			fps_limit: Limite de FPS
			background_color: Couleur d'arrière-plan (RGB)
			show_debug_info: Afficher les informations de débogage
			antialiasing: Activer l'anticrénelage
			msaa_samples: Nombre d'échantillons MSAA si l'anticrénelage est activé
		"""
		self.logger = get_logger()
		self.timer = Timer("Renderer")
		
		# Dimensions de la fenêtre
		self.width = width
		self.height = height
		self.title = title
		self.render_mode = render_mode
		self.view_mode = view_mode
		self.fps_limit = fps_limit
		self.background_color = background_color
		self.show_debug_info = show_debug_info
		self.antialiasing = antialiasing
		self.msaa_samples = msaa_samples
		
		# État du renderer
		self.running = False
		self.paused = False
		self.frame_count = 0
		self.current_fps = 0
		self.simulation_speed = 1.0  # Multiplicateur de vitesse
		
		# Caméra
		self.camera_position = np.array([0.0, 10.0, 30.0], dtype=np.float32)
		self.camera_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
		self.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
		self.camera_zoom = 1.0
		self.camera_rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # pitch, yaw, roll
		self.followed_entity_id = None
		
		# Focus et sélection
		self.selected_entity_id = None
		self.highlighted_entities = []
		
		# Entrées
		self.mouse_position = (0, 0)
		self.mouse_pressed = [False, False, False]  # Gauche, Milieu, Droit
		self.keys_pressed = {}
		
		# Renderers spécifiques
		self.creature_renderer = None
		self.environment_renderer = None
		
		# Éléments UI
		self.ui_elements = {}
		self.font = None
		self.font_size = 16
		
		# Horloge pour limiter les FPS
		self.clock = None
		
		# Verrou pour la synchronisation du rendu
		self.render_lock = threading.Lock()
		
		# Initialisation de PyGame
		self._initialize_pygame()
	
	def _initialize_pygame(self) -> None:
		"""Initialise PyGame et crée la fenêtre de rendu."""
		self.logger.info("Initialisation de PyGame", module="visualization")
		pygame.init()
		
		# Créer la fenêtre
		pygame.display.set_caption(self.title)
		
		# Configuration des flags OpenGL
		flags = pygame.DOUBLEBUF | pygame.OPENGL
		if self.antialiasing:
			pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
			pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, self.msaa_samples)
		
		self.screen = pygame.display.set_mode((self.width, self.height), flags)
		
		# Initialiser l'horloge
		self.clock = pygame.time.Clock()
		
		# Initialiser les renderers spécifiques
		self.creature_renderer = CreatureRenderer()
		self.environment_renderer = EnvironmentRenderer()
		
		# Initialiser les polices
		pygame.font.init()
		self.font = pygame.font.SysFont("Arial", self.font_size)
		
		# Initialiser OpenGL
		self._initialize_opengl()
		
		self.logger.info(f"Fenêtre de rendu créée: {self.width}x{self.height}", module="visualization")
		
	def _initialize_opengl(self) -> None:
		"""Initialise OpenGL et configure les paramètres de rendu."""
		try:
			import OpenGL.GL as gl
			import OpenGL.GLU as glu
			
			# Configurer la couleur d'arrière-plan
			gl.glClearColor(
				self.background_color[0] / 255.0,
				self.background_color[1] / 255.0,
				self.background_color[2] / 255.0,
				1.0
			)
			
			# Activer la profondeur
			gl.glEnable(gl.GL_DEPTH_TEST)
			gl.glDepthFunc(gl.GL_LEQUAL)
			
			# Activer l'anticrénelage si demandé
			if self.antialiasing:
				gl.glEnable(gl.GL_MULTISAMPLE)
				
			# Configurer l'éclairage
			gl.glEnable(gl.GL_LIGHTING)
			gl.glEnable(gl.GL_LIGHT0)
			gl.glEnable(gl.GL_COLOR_MATERIAL)
			gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
			
			# Position de la lumière
			light_position = [100.0, 100.0, 100.0, 1.0]
			gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, light_position)
			
			# Configurer le mélange pour la transparence
			gl.glEnable(gl.GL_BLEND)
			gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
			
			# Configurer la matrice de projection
			gl.glMatrixMode(gl.GL_PROJECTION)
			gl.glLoadIdentity()
			glu.gluPerspective(45, self.width / self.height, 0.1, 1000.0)
			
			# Configurer la matrice de modèle-vue
			gl.glMatrixMode(gl.GL_MODELVIEW)
			gl.glLoadIdentity()
			
			self.logger.info("OpenGL initialisé avec succès", module="visualization")
		except ImportError:
			self.logger.error("OpenGL n'est pas disponible, le rendu 3D sera limité", module="visualization")
	
	def set_world(self, world: MarineWorld) -> None:
		"""
		Définit le monde marin à rendre.
		
		Args:
			world: Instance du monde marin
		"""
		self.world = world
		self.environment_renderer.set_world(world)
		self.creature_renderer.set_world(world)
		
		# Ajuster la caméra pour voir l'ensemble du monde
		world_size = world.size
		self.camera_position = np.array([world_size[0] / 2, world_size[1] / 2, world_size[2] * 1.5])
		self.camera_target = np.array([world_size[0] / 2, world_size[1] / 2, 0])
	
	def start_rendering(self) -> None:
		"""Démarre la boucle de rendu."""
		self.running = True
		self.timer.start()
		
		self.logger.info("Démarrage de la boucle de rendu", module="visualization")
		
		# Boucle principale
		while self.running:
			# Mesurer le temps de rendu
			with self.timer.section("frame"):
				# Traiter les événements
				self._process_events()
				
				# Mettre à jour la caméra
				self._update_camera()
				
				# Effectuer le rendu
				self._render_frame()
				
				# Limiter le FPS
				self.current_fps = self.clock.get_fps()
				self.clock.tick(self.fps_limit)
				
				# Incrementer le compteur de frames
				self.frame_count += 1
		
		# Nettoyage
		self._cleanup()
	
	def stop_rendering(self) -> None:
		"""Arrête la boucle de rendu."""
		self.running = False
	
	def _process_events(self) -> None:
		"""Traite les événements utilisateur."""
		for event in pygame.event.get():
			# Fermeture de la fenêtre
			if event.type == pygame.QUIT:
				self.running = False
			
			# Événements du clavier
			elif event.type == pygame.KEYDOWN:
				self.keys_pressed[event.key] = True
				self._handle_key_press(event.key)
			
			elif event.type == pygame.KEYUP:
				self.keys_pressed[event.key] = False
			
			# Événements de la souris
			elif event.type == pygame.MOUSEMOTION:
				self.mouse_position = event.pos
				self._handle_mouse_motion(event)
			
			elif event.type == pygame.MOUSEBUTTONDOWN:
				if event.button <= 3:
					self.mouse_pressed[event.button - 1] = True
				self._handle_mouse_button_down(event)
			
			elif event.type == pygame.MOUSEBUTTONUP:
				if event.button <= 3:
					self.mouse_pressed[event.button - 1] = False
				self._handle_mouse_button_up(event)
			
			# Événements de la molette
			elif event.type == pygame.MOUSEWHEEL:
				self._handle_mouse_wheel(event)
	
	def _handle_key_press(self, key: int) -> None:
		"""
		Gère les touches du clavier.
		
		Args:
			key: Code de la touche pressée
		"""
		# Touche Échap: quitter
		if key == pygame.K_ESCAPE:
			self.running = False
		
		# Touche P: pause/reprise
		elif key == pygame.K_p:
			self.paused = not self.paused
			self.logger.info(f"Simulation {'en pause' if self.paused else 'reprise'}", module="visualization")
		
		# Touches de contrôle de la vitesse
		elif key == pygame.K_PLUS or key == pygame.K_KP_PLUS:
			self.simulation_speed = min(8.0, self.simulation_speed * 2.0)
			self.logger.info(f"Vitesse de simulation: {self.simulation_speed}x", module="visualization")
		
		elif key == pygame.K_MINUS or key == pygame.K_KP_MINUS:
			self.simulation_speed = max(0.25, self.simulation_speed / 2.0)
			self.logger.info(f"Vitesse de simulation: {self.simulation_speed}x", module="visualization")
		
		# Touche F: activer/désactiver les infos de débogage
		elif key == pygame.K_f:
			self.show_debug_info = not self.show_debug_info
		
		# Touches de mode de rendu
		elif key == pygame.K_1:
			self.render_mode = RenderMode.WIREFRAME
		elif key == pygame.K_2:
			self.render_mode = RenderMode.SOLID
		elif key == pygame.K_3:
			self.render_mode = RenderMode.TEXTURED
		elif key == pygame.K_4:
			self.render_mode = RenderMode.SHADED
		
		# Touches de mode de vue
		elif key == pygame.K_F1:
			self.view_mode = ViewMode.PERSPECTIVE
		elif key == pygame.K_F2:
			self.view_mode = ViewMode.ORTHOGRAPHIC
		elif key == pygame.K_F3:
			self.view_mode = ViewMode.TOP_DOWN
		elif key == pygame.K_F4:
			self.view_mode = ViewMode.SIDE
		elif key == pygame.K_F5:
			if self.selected_entity_id:
				self.view_mode = ViewMode.FOLLOW
				self.followed_entity_id = self.selected_entity_id
			else:
				self.logger.warning("Aucune entité sélectionnée à suivre", module="visualization")
	
	def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
		"""
		Gère les mouvements de la souris.
		
		Args:
			event: Événement de mouvement de la souris
		"""
		# Rotation de la caméra avec le bouton gauche de la souris
		if self.mouse_pressed[0]:
			dx, dy = event.rel
			sensitivity = 0.1
			
			self.camera_rotation[1] += dx * sensitivity  # Yaw
			self.camera_rotation[0] += dy * sensitivity  # Pitch
			
			# Limiter le pitch pour éviter le retournement
			self.camera_rotation[0] = np.clip(self.camera_rotation[0], -89.0, 89.0)
	
	def _handle_mouse_button_down(self, event: pygame.event.Event) -> None:
		"""
		Gère les clics de souris.
		
		Args:
			event: Événement de clic de souris
		"""
		# Clic gauche: sélectionner une entité
		if event.button == 1:
			entity_id = self._pick_entity_at_cursor()
			if entity_id:
				self.selected_entity_id = entity_id
				self.logger.info(f"Entité sélectionnée: {entity_id}", module="visualization")
			else:
				self.selected_entity_id = None
	
	def _handle_mouse_button_up(self, event: pygame.event.Event) -> None:
		"""
		Gère les relâchements de boutons de souris.
		
		Args:
			event: Événement de relâchement de bouton de souris
		"""
		pass
	
	def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
		"""
		Gère les événements de la molette de la souris.
		
		Args:
			event: Événement de la molette de la souris
		"""
		# Zoom de la caméra
		self.camera_zoom *= 1.1 ** event.y
		self.camera_zoom = np.clip(self.camera_zoom, 0.1, 10.0)
	
	def _update_camera(self) -> None:
		"""Met à jour la position et l'orientation de la caméra."""
		# Si en mode suivi, positionner la caméra derrière l'entité suivie
		if self.view_mode == ViewMode.FOLLOW and self.followed_entity_id:
			entity = self._get_entity_by_id(self.followed_entity_id)
			if entity:
				entity_position = np.array(entity.position)
				entity_orientation = np.array(entity.orientation)
				
				# Calculer la position de la caméra derrière l'entité
				# (utiliser l'orientation de l'entité pour déterminer "derrière")
				camera_offset = np.array([
					-np.sin(entity_orientation[1]) * 10.0,
					5.0,
					-np.cos(entity_orientation[1]) * 10.0
				])
				
				self.camera_position = entity_position + camera_offset
				self.camera_target = entity_position
		
		# Appliquer la rotation de la caméra dans les autres modes de vue
		elif self.view_mode != ViewMode.TOP_DOWN and self.view_mode != ViewMode.SIDE:
			# Convertir les angles d'Euler en une position sur une sphère
			pitch_rad = np.radians(self.camera_rotation[0])
			yaw_rad = np.radians(self.camera_rotation[1])
			
			distance = np.linalg.norm(self.camera_position - self.camera_target) * self.camera_zoom
			
			# Calculer la nouvelle position de la caméra
			x = np.sin(yaw_rad) * np.cos(pitch_rad)
			y = np.sin(pitch_rad)
			z = np.cos(yaw_rad) * np.cos(pitch_rad)
			
			self.camera_position = self.camera_target + np.array([x, y, z]) * distance
		
		# Mettre à jour la vue orthographique
		if self.view_mode == ViewMode.ORTHOGRAPHIC:
			pass  # Sera géré dans le rendu
		
		# Vue de dessus
		elif self.view_mode == ViewMode.TOP_DOWN:
			if self.world:
				center = np.array([self.world.size[0] / 2, 0, self.world.size[2] / 2])
				self.camera_position = np.array([center[0], self.world.size[1] * 2, center[2]])
				self.camera_target = center
		
		# Vue latérale
		elif self.view_mode == ViewMode.SIDE:
			if self.world:
				center = np.array([self.world.size[0] / 2, self.world.size[1] / 2, 0])
				self.camera_position = np.array([center[0], center[1], self.world.size[2] * 2])
				self.camera_target = center
	
	def _render_frame(self) -> None:
		"""Effectue le rendu d'une image."""
		try:
			import OpenGL.GL as gl
			import OpenGL.GLU as glu
			
			# Effacer l'écran et le buffer de profondeur
			gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
			
			# Configurer la caméra
			gl.glMatrixMode(gl.GL_PROJECTION)
			gl.glLoadIdentity()
			
			if self.view_mode == ViewMode.ORTHOGRAPHIC:
				# Calculer une projection orthographique adaptée aux dimensions du monde
				size = 20.0 * self.camera_zoom
				aspect = self.width / self.height
				gl.glOrtho(-size * aspect, size * aspect, -size, size, 0.1, 1000.0)
			else:
				# Projection perspective
				glu.gluPerspective(45 / self.camera_zoom, self.width / self.height, 0.1, 1000.0)
			
			gl.glMatrixMode(gl.GL_MODELVIEW)
			gl.glLoadIdentity()
			
			# Positionner la caméra
			glu.gluLookAt(
				self.camera_position[0], self.camera_position[1], self.camera_position[2],
				self.camera_target[0], self.camera_target[1], self.camera_target[2],
				self.camera_up[0], self.camera_up[1], self.camera_up[2]
			)
			
			# Mode de rendu
			if self.render_mode == RenderMode.WIREFRAME:
				gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
			else:
				gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
			
			# Désactiver l'éclairage pour les modes simples
			if self.render_mode == RenderMode.WIREFRAME or self.render_mode == RenderMode.SOLID:
				gl.glDisable(gl.GL_LIGHTING)
			else:
				gl.glEnable(gl.GL_LIGHTING)
			
			# Activer les textures si nécessaire
			if self.render_mode == RenderMode.TEXTURED:
				gl.glEnable(gl.GL_TEXTURE_2D)
			else:
				gl.glDisable(gl.GL_TEXTURE_2D)
			
			# Rendu de l'environnement
			with self.timer.section("environment_render"):
				if self.environment_renderer and hasattr(self, 'world'):
					self.environment_renderer.render(self.render_mode)
			
			# Rendu des créatures
			with self.timer.section("creatures_render"):
				if self.creature_renderer and hasattr(self, 'world'):
					self.creature_renderer.render(self.render_mode, self.selected_entity_id, self.highlighted_entities)
			
			# Rendu de l'interface utilisateur
			with self.timer.section("ui_render"):
				self._render_ui()
			
			# Échanger les tampons pour afficher l'image rendue
			pygame.display.flip()
			
		except Exception as e:
			self.logger.error(f"Erreur lors du rendu: {str(e)}", module="visualization", exc_info=True)
	
	def _render_ui(self) -> None:
		"""Effectue le rendu des éléments de l'interface utilisateur."""
		try:
			import OpenGL.GL as gl
			
			# Désactiver le test de profondeur pour l'UI
			gl.glDisable(gl.GL_DEPTH_TEST)
			gl.glDisable(gl.GL_LIGHTING)
			
			# Passer en mode 2D
			gl.glMatrixMode(gl.GL_PROJECTION)
			gl.glPushMatrix()
			gl.glLoadIdentity()
			gl.glOrtho(0, self.width, self.height, 0, -1, 1)
			
			gl.glMatrixMode(gl.GL_MODELVIEW)
			gl.glPushMatrix()
			gl.glLoadIdentity()
			
			# Rendre les informations de débogage si activé
			if self.show_debug_info:
				self._render_debug_info()
			
			# Rendre les détails de l'entité sélectionnée
			if self.selected_entity_id:
				self._render_entity_details()
			
			# Rendre les autres éléments d'UI
			for ui_element_id, ui_element in self.ui_elements.items():
				if ui_element.get("visible", True):
					self._render_ui_element(ui_element)
			
			# Revenir au mode 3D
			gl.glMatrixMode(gl.GL_PROJECTION)
			gl.glPopMatrix()
			
			gl.glMatrixMode(gl.GL_MODELVIEW)
			gl.glPopMatrix()
			
			# Réactiver le test de profondeur
			gl.glEnable(gl.GL_DEPTH_TEST)
			if self.render_mode == RenderMode.SHADED or self.render_mode == RenderMode.TEXTURED:
				gl.glEnable(gl.GL_LIGHTING)
				
		except Exception as e:
			self.logger.error(f"Erreur lors du rendu de l'UI: {str(e)}", module="visualization", exc_info=True)
	
	def _render_debug_info(self) -> None:
		"""Affiche les informations de débogage."""
		# Créer une surface de texte pour les informations de débogage
		debug_lines = [
			f"FPS: {int(self.current_fps)}",
			f"Frame: {self.frame_count}",
			f"Render Mode: {self.render_mode.name}",
			f"View Mode: {self.view_mode.name}",
			f"Speed: {self.simulation_speed:.2f}x",
			f"Camera Pos: ({self.camera_position[0]:.1f}, {self.camera_position[1]:.1f}, {self.camera_position[2]:.1f})",
			f"Selected: {self.selected_entity_id if self.selected_entity_id else 'None'}",
		]
		
		# Ajouter des informations sur le monde si disponible
		if hasattr(self, 'world'):
			debug_lines.extend([
				f"Creatures: {len(self.world.creatures)}",
				f"Time: {self.world.currentTime:.1f}",
				f"Food Resources: {len(self.world.foodResources)}"
			])
		
		# Ajouter des informations sur le rendu
		render_times = self.timer.get_section_stats("frame")
		if render_times and "average" in render_times:
			debug_lines.append(f"Frame Time: {render_times['average']*1000:.1f} ms")
		
		render_times = self.timer.get_section_stats("environment_render")
		if render_times and "average" in render_times:
			debug_lines.append(f"Environment: {render_times['average']*1000:.1f} ms")
		
		render_times = self.timer.get_section_stats("creatures_render")
		if render_times and "average" in render_times:
			debug_lines.append(f"Creatures: {render_times['average']*1000:.1f} ms")
		
		# Afficher les lignes de texte
		y_offset = 10
		for line in debug_lines:
			self._render_text(line, 10, y_offset, (255, 255, 255))
			y_offset += 20
	
	def _render_entity_details(self) -> None:
		"""Affiche les détails de l'entité sélectionnée."""
		entity = self._get_entity_by_id(self.selected_entity_id)
		if not entity:
			return
			
		# Créer un panneau pour les informations de l'entité
		panel_width = 300
		panel_height = 400
		panel_x = self.width - panel_width - 10
		panel_y = 10
		
		# Dessiner le fond du panneau
		self._render_ui_panel(panel_x, panel_y, panel_width, panel_height, (0, 0, 0, 180))
		
		# Titre du panneau
		title = f"Entity Details: {entity.id[:8]}..."
		self._render_text(title, panel_x + 10, panel_y + 10, (255, 255, 255))
		
		# Afficher les propriétés de l'entité
		y_offset = panel_y + 40
		line_height = 20
		
		# Propriétés générales
		properties = [
			f"Type: {entity.__class__.__name__}",
			f"Species: {getattr(entity, 'speciesId', 'N/A')}",
			f"Alive: {getattr(entity, 'isAlive', True)}",
			f"Position: ({entity.position[0]:.1f}, {entity.position[1]:.1f}, {entity.position[2]:.1f})",
			f"Velocity: {np.linalg.norm(entity.velocity):.2f}",
			f"Energy: {getattr(entity, 'energy', 0):.1f}/{getattr(entity, 'maxEnergy', 0):.1f}",
			f"Age: {getattr(entity, 'age', 0):.1f}",
			f"Fitness: {getattr(entity, 'fitness', 0):.2f}"
		]
		
		for prop in properties:
			self._render_text(prop, panel_x + 10, y_offset, (220, 220, 220))
			y_offset += line_height
		
		# Séparateur
		y_offset += 10
		self._render_line(panel_x + 10, y_offset, panel_x + panel_width - 10, y_offset, (150, 150, 150))
		y_offset += 10
		
		# Morphologie
		if hasattr(entity, 'joints') and hasattr(entity, 'limbs') and hasattr(entity, 'muscles'):
			morphology = [
				f"Joints: {len(entity.joints)}",
				f"Limbs: {len(entity.limbs)}",
				f"Muscles: {len(entity.muscles)}",
				f"Sensors: {len(getattr(entity, 'sensors', []))}",
				f"Size: {getattr(entity, 'size', 1.0):.2f}"
			]
			
			for prop in morphology:
				self._render_text(prop, panel_x + 10, y_offset, (220, 220, 220))
				y_offset += line_height
	
	def _render_ui_element(self, element: Dict[str, Any]) -> None:
		"""
		Rend un élément d'interface utilisateur.
		
		Args:
			element: Dictionnaire décrivant l'élément
		"""
		element_type = element.get("type", "")
		
		if element_type == "panel":
			self._render_ui_panel(
				element.get("x", 0),
				element.get("y", 0),
				element.get("width", 100),
				element.get("height", 100),
				element.get("color", (0, 0, 0, 180))
			)
		
		elif element_type == "text":
			self._render_text(
				element.get("text", ""),
				element.get("x", 0),
				element.get("y", 0),
				element.get("color", (255, 255, 255))
			)
		
		elif element_type == "button":
			self._render_ui_button(
				element.get("text", ""),
				element.get("x", 0),
				element.get("y", 0),
				element.get("width", 100),
				element.get("height", 30),
				element.get("color", (50, 50, 200, 200)),
				element.get("hover", False)
			)
		
		elif element_type == "line":
			self._render_line(
				element.get("x1", 0),
				element.get("y1", 0),
				element.get("x2", 100),
				element.get("y2", 0),
				element.get("color", (255, 255, 255))
			)
	
	def _render_ui_panel(self, x: int, y: int, width: int, height: int, color: Tuple[int, int, int, int]) -> None:
		"""
		Rend un panneau d'interface utilisateur.
		
		Args:
			x, y: Position du panneau
			width, height: Dimensions du panneau
			color: Couleur et transparence (RGBA)
		"""
		try:
			import OpenGL.GL as gl
			
			r, g, b, a = color[0]/255.0, color[1]/255.0, color[2]/255.0, color[3]/255.0 if len(color) > 3 else 1.0
			
			gl.glColor4f(r, g, b, a)
			gl.glBegin(gl.GL_QUADS)
			gl.glVertex2f(x, y)
			gl.glVertex2f(x + width, y)
			gl.glVertex2f(x + width, y + height)
			gl.glVertex2f(x, y + height)
			gl.glEnd()
			
			# Bordure
			gl.glColor4f(r + 0.2, g + 0.2, b + 0.2, a)
			gl.glBegin(gl.GL_LINE_LOOP)
			gl.glVertex2f(x, y)
			gl.glVertex2f(x + width, y)
			gl.glVertex2f(x + width, y + height)
			gl.glVertex2f(x, y + height)
			gl.glEnd()
			
		except Exception as e:
			self.logger.error(f"Erreur lors du rendu du panneau: {str(e)}", module="visualization", exc_info=True)
	
	def _render_ui_button(self, text: str, x: int, y: int, width: int, height: int, 
						color: Tuple[int, int, int, int], hover: bool = False) -> None:
		"""
		Rend un bouton d'interface utilisateur.
		
		Args:
			text: Texte du bouton
			x, y: Position du bouton
			width, height: Dimensions du bouton
			color: Couleur et transparence (RGBA)
			hover: Si le curseur survole le bouton
		"""
		# Dessiner le fond du bouton
		if hover:
			# Éclaircir la couleur en cas de survol
			hover_color = tuple(min(255, c + 30) for c in color[:3]) + (color[3],) if len(color) > 3 else color
			self._render_ui_panel(x, y, width, height, hover_color)
		else:
			self._render_ui_panel(x, y, width, height, color)
		
		# Centrer le texte
		text_surface = self.font.render(text, True, (255, 255, 255))
		text_width, text_height = text_surface.get_size()
		text_x = x + (width - text_width) // 2
		text_y = y + (height - text_height) // 2
		
		# Dessiner le texte
		self._render_text(text, text_x, text_y, (255, 255, 255))
	
	def _render_text(self, text: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
		"""
		Rend du texte à l'écran.
		
		Args:
			text: Texte à afficher
			x, y: Position du texte
			color: Couleur du texte (RGB)
		"""
		try:
			# Créer une surface de texte
			text_surface = self.font.render(text, True, color)
			text_data = pygame.image.tostring(text_surface, "RGBA", True)
			text_width, text_height = text_surface.get_size()
			
			import OpenGL.GL as gl
			
			# Sauvegarder l'état actuel
			gl.glPushAttrib(gl.GL_ALL_ATTRIB_BITS)
			
			# Désactiver certaines fonctionnalités pour le texte
			gl.glDisable(gl.GL_LIGHTING)
			gl.glDisable(gl.GL_DEPTH_TEST)
			gl.glEnable(gl.GL_BLEND)
			gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
			
			# Créer une texture pour le texte
			texture_id = gl.glGenTextures(1)
			gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
			gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
			gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, text_width, text_height, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, text_data)
			
			# Dessiner le texte
			gl.glEnable(gl.GL_TEXTURE_2D)
			gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
			gl.glColor3f(1.0, 1.0, 1.0)
			
			gl.glBegin(gl.GL_QUADS)
			gl.glTexCoord2f(0, 0); gl.glVertex2f(x, y)
			gl.glTexCoord2f(1, 0); gl.glVertex2f(x + text_width, y)
			gl.glTexCoord2f(1, 1); gl.glVertex2f(x + text_width, y + text_height)
			gl.glTexCoord2f(0, 1); gl.glVertex2f(x, y + text_height)
			gl.glEnd()
			
			# Nettoyer
			gl.glDeleteTextures([texture_id])
			
			# Restaurer l'état précédent
			gl.glPopAttrib()
			
		except Exception as e:
			self.logger.error(f"Erreur lors du rendu du texte: {str(e)}", module="visualization", exc_info=True)
	
	def _render_line(self, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
		"""
		Dessine une ligne à l'écran.
		
		Args:
			x1, y1: Point de départ
			x2, y2: Point d'arrivée
			color: Couleur de la ligne (RGB)
		"""
		try:
			import OpenGL.GL as gl
			
			r, g, b = color[0]/255.0, color[1]/255.0, color[2]/255.0
			
			gl.glColor3f(r, g, b)
			gl.glBegin(gl.GL_LINES)
			gl.glVertex2f(x1, y1)
			gl.glVertex2f(x2, y2)
			gl.glEnd()
			
		except Exception as e:
			self.logger.error(f"Erreur lors du rendu de la ligne: {str(e)}", module="visualization", exc_info=True)
	
	def _pick_entity_at_cursor(self) -> Optional[str]:
		"""
		Sélectionne une entité à la position du curseur.
		
		Returns:
			ID de l'entité sélectionnée, ou None si aucune entité n'est sous le curseur
		"""
		# Utiliser le picking OpenGL pour sélectionner une entité
		try:
			import OpenGL.GL as gl
			import OpenGL.GLU as glu
			
			# Obtenir la position du curseur
			x, y = self.mouse_position
			y = self.height - y  # Inverser la coordonnée y (OpenGL origine en bas à gauche)
			
			# Lire le buffer de profondeur à la position du curseur
			depth = gl.glReadPixels(x, y, 1, 1, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT)
			
			# Si la profondeur est 1.0, il n'y a rien sous le curseur
			if depth[0][0] == 1.0:
				return None
			
			# Obtenir les matrices de projection et de modèle-vue
			modelview_matrix = gl.glGetDoublev(gl.GL_MODELVIEW_MATRIX)
			projection_matrix = gl.glGetDoublev(gl.GL_PROJECTION_MATRIX)
			viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)
			
			# Convertir les coordonnées d'écran en coordonnées 3D
			world_pos = glu.gluUnProject(x, y, depth[0][0], 
										 modelview_matrix, projection_matrix, viewport)
			
			# Trouver l'entité la plus proche
			if hasattr(self, 'world') and self.world:
				closest_entity = None
				closest_distance = float('inf')
				
				for entity_id, entity in self.world.creatures.items():
					if hasattr(entity, 'position'):
						distance = np.linalg.norm(np.array(entity.position) - np.array(world_pos))
						if distance < closest_distance and distance < 5.0:  # Seuil de distance
							closest_distance = distance
							closest_entity = entity_id
				
				return closest_entity
			
		except Exception as e:
			self.logger.error(f"Erreur lors du picking: {str(e)}", module="visualization", exc_info=True)
			
		return None
	
	def _get_entity_by_id(self, entity_id: str) -> Any:
		"""
		Récupère une entité par son ID.
		
		Args:
			entity_id: ID de l'entité
			
		Returns:
			L'entité correspondante, ou None si non trouvée
		"""
		if hasattr(self, 'world') and self.world and entity_id in self.world.creatures:
			return self.world.creatures[entity_id]
		return None
	
	def add_ui_element(self, element_id: str, element_data: Dict[str, Any]) -> None:
		"""
		Ajoute un élément à l'interface utilisateur.
		
		Args:
			element_id: Identifiant unique de l'élément
			element_data: Données de l'élément (type, position, etc.)
		"""
		self.ui_elements[element_id] = element_data
	
	def remove_ui_element(self, element_id: str) -> None:
		"""
		Supprime un élément de l'interface utilisateur.
		
		Args:
			element_id: Identifiant de l'élément à supprimer
		"""
		if element_id in self.ui_elements:
			del self.ui_elements[element_id]
	
	def update_ui_element(self, element_id: str, update_data: Dict[str, Any]) -> None:
		"""
		Met à jour un élément de l'interface utilisateur.
		
		Args:
			element_id: Identifiant de l'élément à mettre à jour
			update_data: Nouvelles données à appliquer
		"""
		if element_id in self.ui_elements:
			self.ui_elements[element_id].update(update_data)
	
	def set_highlight_entities(self, entity_ids: List[str]) -> None:
		"""
		Définit les entités à mettre en évidence.
		
		Args:
			entity_ids: Liste des IDs d'entités à mettre en évidence
		"""
		self.highlighted_entities = entity_ids
	
	def take_screenshot(self, filename: Optional[str] = None) -> bool:
		"""
		Prend une capture d'écran de la fenêtre.
		
		Args:
			filename: Nom du fichier de sortie (optionnel)
			
		Returns:
			True si la capture a réussi, False sinon
		"""
		try:
			if filename is None:
				timestamp = time.strftime("%Y%m%d_%H%M%S")
				filename = f"screenshot_{timestamp}.png"
			
			# Assurez-vous que le répertoire existe
			directory = os.path.dirname(filename)
			if directory and not os.path.exists(directory):
				os.makedirs(directory)
			
			# Prendre la capture d'écran
			pygame.image.save(self.screen, filename)
			
			self.logger.info(f"Capture d'écran enregistrée: {filename}", module="visualization")
			return True
			
		except Exception as e:
			self.logger.error(f"Erreur lors de la capture d'écran: {str(e)}", module="visualization", exc_info=True)
			return False
	
	def _cleanup(self) -> None:
		"""Nettoie les ressources avant de fermer la fenêtre."""
		self.logger.info("Nettoyage des ressources de rendu", module="visualization")
		
		# Arrêter le timer
		self.timer.stop()
		
		# Nettoyer les renderers spécifiques
		if self.creature_renderer:
			self.creature_renderer.cleanup()
		
		if self.environment_renderer:
			self.environment_renderer.cleanup()
		
		# Fermer PyGame
		pygame.quit()