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
from core.entities.creature import Creature
from core.entities.joint import Joint
from core.entities.limb import Limb
from core.entities.muscle import Muscle
from core.entities.sensor import Sensor
from visualization.renderer import RenderMode


class CreatureRenderer:
	"""
	Classe responsable du rendu des créatures marines.
	Gère la représentation visuelle des créatures avec leurs articulations,
	membres, muscles et capteurs.
	"""
	
	def __init__(self) -> None:
		"""
		Initialise le renderer de créatures.
		"""
		self.logger = get_logger()
		self.world = None
		self.creature_meshes: Dict[str, Dict[str, Any]] = {}
		self.textures: Dict[str, int] = {}
		self.display_lists: Dict[str, int] = {}
		
		# Couleurs prédéfinies pour différentes espèces
		self.species_colors: Dict[str, Tuple[float, float, float]] = {
			"default": (0.5, 0.5, 0.8)
		}
		
		# Paramètres du rendu
		self.joint_detail = 8  # Niveau de détail pour les sphères
		self.show_joints = True
		self.show_limbs = True
		self.show_muscles = True
		self.show_sensors = True
		self.highlight_selected = True
		self.x_ray_view = False
		
		# Chargement des textures et initialisation des ressources
		self._init_resources()
	
	def _init_resources(self) -> None:
		"""
		Initialise les ressources nécessaires au rendu (textures, modèles, etc.).
		"""
		try:
			# Créer des textures simples pour les différents types de tissus
			self._create_skin_texture()
			self._create_muscle_texture()
			self._create_bone_texture()
			self._create_sensor_texture()
			
			# Créer des listes d'affichage pour les primitives communes
			self._create_sphere_display_list()
			self._create_cylinder_display_list()
			
			self.logger.info("Ressources du CreatureRenderer initialisées", module="visualization")
		except Exception as e:
			self.logger.error(f"Erreur lors de l'initialisation des ressources: {str(e)}", 
							module="visualization", exc_info=True)
	
	def _create_skin_texture(self) -> None:
		"""Crée une texture de peau pour les créatures."""
		texture_id = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
		
		# Texture de peau simple (motif d'écailles)
		width, height = 64, 64
		texture_data = np.zeros((height, width, 3), dtype=np.uint8)
		
		for y in range(height):
			for x in range(width):
				# Créer un motif d'écailles
				dist = math.sin(x * 0.5) * 10 + math.sin(y * 0.5) * 10
				color_val = int(128 + dist) % 255
				texture_data[y, x] = [color_val, color_val, color_val + 40]
		
		# Charger la texture
		gl.glTexImage2D(
			gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
			gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_data
		)
		
		# Configurer les paramètres de texture
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		
		self.textures["skin"] = texture_id
	
	def _create_muscle_texture(self) -> None:
		"""Crée une texture de muscle pour les créatures."""
		texture_id = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
		
		# Texture de muscle simple (motif strié)
		width, height = 64, 64
		texture_data = np.zeros((height, width, 3), dtype=np.uint8)
		
		for y in range(height):
			for x in range(width):
				# Créer un motif strié
				stripe = (math.sin(y * 1.0) > 0.7) * 60
				texture_data[y, x] = [180 + stripe, 20 + stripe, 20 + stripe]
		
		# Charger la texture
		gl.glTexImage2D(
			gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
			gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_data
		)
		
		# Configurer les paramètres de texture
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		
		self.textures["muscle"] = texture_id
	
	def _create_bone_texture(self) -> None:
		"""Crée une texture d'os pour les articulations."""
		texture_id = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
		
		# Texture d'os simple (blanc cassé avec taches)
		width, height = 64, 64
		texture_data = np.zeros((height, width, 3), dtype=np.uint8)
		
		for y in range(height):
			for x in range(width):
				# Base blanc cassé
				base_color = 230
				# Ajouter quelques taches aléatoires
				noise = int(10 * (math.sin(x * 0.3) * math.sin(y * 0.3)))
				texture_data[y, x] = [base_color + noise, base_color - 5 + noise, base_color - 15 + noise]
		
		# Charger la texture
		gl.glTexImage2D(
			gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
			gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_data
		)
		
		# Configurer les paramètres de texture
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		
		self.textures["bone"] = texture_id
	
	def _create_sensor_texture(self) -> None:
		"""Crée une texture pour les capteurs."""
		texture_id = gl.glGenTextures(1)
		gl.glBindTexture(gl.GL_TEXTURE_2D, texture_id)
		
		# Texture de capteur (motif nervuré)
		width, height = 64, 64
		texture_data = np.zeros((height, width, 3), dtype=np.uint8)
		
		for y in range(height):
			for x in range(width):
				# Motif nervuré/électrique
				pattern = int(20 * math.sin((x + y) * 0.4) * math.sin(x * 0.5))
				r = 100 + pattern
				g = 180 + pattern
				b = 220 + pattern
				texture_data[y, x] = [min(255, r), min(255, g), min(255, b)]
		
		# Charger la texture
		gl.glTexImage2D(
			gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0,
			gl.GL_RGB, gl.GL_UNSIGNED_BYTE, texture_data
		)
		
		# Configurer les paramètres de texture
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
		gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
		
		self.textures["sensor"] = texture_id
	
	def _create_sphere_display_list(self) -> None:
		"""Crée une liste d'affichage pour une sphère."""
		list_id = gl.glGenLists(1)
		
		gl.glNewList(list_id, gl.GL_COMPILE)
		
		# Dessiner une sphère
		quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(quad, gl.GL_TRUE)
		glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
		glu.gluSphere(quad, 1.0, self.joint_detail, self.joint_detail)
		glu.gluDeleteQuadric(quad)
		
		gl.glEndList()
		
		self.display_lists["sphere"] = list_id
	
	def _create_cylinder_display_list(self) -> None:
		"""Crée une liste d'affichage pour un cylindre."""
		list_id = gl.glGenLists(1)
		
		gl.glNewList(list_id, gl.GL_COMPILE)
		
		# Dessiner un cylindre
		quad = glu.gluNewQuadric()
		glu.gluQuadricTexture(quad, gl.GL_TRUE)
		glu.gluQuadricNormals(quad, glu.GLU_SMOOTH)
		glu.gluCylinder(quad, 1.0, 1.0, 1.0, self.joint_detail, 4)
		glu.gluDeleteQuadric(quad)
		
		gl.glEndList()
		
		self.display_lists["cylinder"] = list_id
	
	def set_world(self, world) -> None:
		"""
		Définit le monde marin à rendre.
		
		Args:
			world: Instance du monde marin
		"""
		self.world = world
		
		# Initialiser les maillages pour chaque créature
		self._init_creature_meshes()
	
	def _init_creature_meshes(self) -> None:
		"""
		Initialise les maillages pour toutes les créatures actuelles.
		"""
		if not self.world:
			return
			
		# Nettoyer les maillages existants
		self.creature_meshes = {}
		
		# Pour chaque créature dans le monde
		for creature_id, creature in self.world.creatures.items():
			try:
				# Générer le maillage de la créature
				self._generate_creature_mesh(creature)
			except Exception as e:
				self.logger.error(f"Erreur lors de la génération du maillage pour la créature {creature_id}: {str(e)}", 
								module="visualization", exc_info=True)
	
	def _generate_creature_mesh(self, creature: Creature) -> None:
		"""
		Génère le maillage pour une créature spécifique.
		
		Args:
			creature: Créature pour laquelle générer un maillage
		"""
		if creature.id in self.creature_meshes:
			return
			
		# Obtenir la couleur de l'espèce
		if creature.speciesId in self.species_colors:
			base_color = self.species_colors[creature.speciesId]
		else:
			# Générer une couleur aléatoire basée sur l'ID de l'espèce
			seed = hash(creature.speciesId) % 1000
			np.random.seed(seed)
			base_color = (
				0.3 + 0.6 * np.random.random(),
				0.3 + 0.6 * np.random.random(),
				0.3 + 0.6 * np.random.random()
			)
			# Mémoriser cette couleur pour l'espèce
			self.species_colors[creature.speciesId] = base_color
		
		# Structure pour stocker le maillage de la créature
		mesh = {
			"base_color": base_color,
			"joints": {},     # Articulations
			"limbs": {},      # Membres
			"muscles": {},    # Muscles
			"sensors": {}     # Capteurs
		}
		
		# Traiter les articulations
		for joint in creature.joints:
			mesh["joints"][joint.id] = {
				"position": joint.position,
				"radius": 0.2 * creature.size,  # Taille basée sur la taille de la créature
				"color": (base_color[0] * 1.2, base_color[1] * 1.2, base_color[2] * 1.2)
			}
		
		# Traiter les membres
		for limb in creature.limbs:
			# Obtenir les positions des articulations connectées
			start_joint_pos = None
			end_joint_pos = None
			
			for joint in creature.joints:
				if joint.id == limb.startJointId:
					start_joint_pos = joint.position
				if limb.endJointId and joint.id == limb.endJointId:
					end_joint_pos = joint.position
			
			if start_joint_pos is None:
				continue
				
			if end_joint_pos is None:
				# Si c'est un membre terminal (sans articulation d'arrivée)
				# Créer un point d'extrémité basé sur la direction et la longueur
				direction = np.array([0.0, 0.0, 1.0])  # Direction par défaut
				end_joint_pos = start_joint_pos + direction * limb.length
			
			# Calculer la direction et la longueur du membre
			direction = end_joint_pos - start_joint_pos
			length = np.linalg.norm(direction)
			
			mesh["limbs"][limb.id] = {
				"start_pos": start_joint_pos,
				"end_pos": end_joint_pos,
				"direction": direction / max(length, 1e-6),  # Normaliser
				"length": length,
				"width": limb.width * creature.size,
				"shape": limb.shape,
				"color": (
					base_color[0] * 0.8,
					base_color[1] * 0.8,
					base_color[2] * 0.8
				) if hasattr(limb, 'color') else base_color
			}
		
		# Traiter les muscles
		for muscle in creature.muscles:
			# Trouver les articulations connectées par ce muscle
			joint_positions = []
			
			for joint_id in muscle.jointIds:
				for joint in creature.joints:
					if joint.id == joint_id:
						joint_positions.append(joint.position)
						break
			
			if len(joint_positions) < 2:
				continue
				
			# Calculer le chemin du muscle entre les articulations
			path = []
			for i in range(len(joint_positions) - 1):
				start_pos = joint_positions[i]
				end_pos = joint_positions[i + 1]
				
				# Générer des points intermédiaires pour un rendu lisse
				for t in np.linspace(0, 1, 5):
					pos = start_pos * (1 - t) + end_pos * t
					path.append(pos)
			
			# Calculer le niveau d'activation du muscle (tension)
			activation = muscle.activation if hasattr(muscle, 'activation') else 0.0
			
			mesh["muscles"][muscle.id] = {
				"path": path,
				"thickness": 0.15 * creature.size * (1.0 + 0.3 * abs(activation)),
				"activation": activation,
				"color": (
					0.8 + 0.2 * abs(activation),
					0.2 - 0.2 * abs(activation),
					0.2 - 0.2 * abs(activation)
				)
			}
		
		# Traiter les capteurs
		for sensor in creature.sensors:
			# Positionner le capteur
			mesh["sensors"][sensor.id] = {
				"position": sensor.position,
				"direction": sensor.direction,
				"radius": 0.15 * creature.size,
				"type": sensor.sensorType,
				"color": self._get_sensor_color(sensor.sensorType)
			}
		
		# Stocker le maillage généré
		self.creature_meshes[creature.id] = mesh
	
	def _get_sensor_color(self, sensor_type: str) -> Tuple[float, float, float]:
		"""
		Retourne la couleur à utiliser pour un type de capteur donné.
		
		Args:
			sensor_type: Type de capteur
			
		Returns:
			Couleur RGB
		"""
		colors = {
			"vision": (0.2, 0.8, 0.8),
			"pressure": (0.8, 0.2, 0.8),
			"temperature": (0.8, 0.5, 0.2),
			"chemical": (0.3, 0.8, 0.3),
			"electromagnetic": (0.8, 0.8, 0.2),
			"proximity": (0.5, 0.5, 0.8)
		}
		
		return colors.get(sensor_type, (0.6, 0.6, 0.6))
	
	def render(self, render_mode: RenderMode, selected_entity_id: Optional[str] = None, 
			 highlighted_entities: Optional[List[str]] = None) -> None:
		"""
		Effectue le rendu de toutes les créatures.
		
		Args:
			render_mode: Mode de rendu à utiliser
			selected_entity_id: ID de l'entité sélectionnée (pour la mise en évidence)
			highlighted_entities: Liste des IDs d'entités à mettre en évidence
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
		
		# Si le mode X-ray est activé, configurer le rendu en transparence
		if self.x_ray_view:
			gl.glEnable(gl.GL_BLEND)
			gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
			gl.glDepthMask(gl.GL_FALSE)
		else:
			gl.glDepthMask(gl.GL_TRUE)
		
		# Liste des entités à mettre en évidence
		if highlighted_entities is None:
			highlighted_entities = []
		
		# Mettre à jour les maillages des nouvelles créatures
		for creature_id, creature in self.world.creatures.items():
			if creature_id not in self.creature_meshes:
				self._generate_creature_mesh(creature)
		
		# Rendre chaque créature
		for creature_id, creature in self.world.creatures.items():
			# Vérifier si la créature est vivante
			if not creature.isAlive:
				continue
				
			# Vérifier si le maillage existe
			if creature_id not in self.creature_meshes:
				continue
				
			# Mettre à jour la position et l'orientation du maillage
			self._update_creature_mesh(creature)
			
			# Calcul du niveau de mise en évidence
			highlight_level = 0.0
			if selected_entity_id and creature_id == selected_entity_id:
				highlight_level = 1.0
			elif creature_id in highlighted_entities:
				highlight_level = 0.6
			
			# Rendu de la créature
			self._render_creature(creature_id, render_mode, highlight_level)
	
	def _update_creature_mesh(self, creature: Creature) -> None:
		"""
		Met à jour le maillage d'une créature en fonction de sa position et orientation.
		
		Args:
			creature: Créature à mettre à jour
		"""
		if creature.id not in self.creature_meshes:
			return
			
		mesh = self.creature_meshes[creature.id]
		
		# Matrice de transformation pour la position et l'orientation de la créature
		gl.glPushMatrix()
		
		# Appliquer la position globale
		gl.glTranslatef(creature.position[0], creature.position[1], creature.position[2])
		
		# Appliquer l'orientation (angles d'Euler)
		gl.glRotatef(np.degrees(creature.orientation[0]), 1, 0, 0)  # Pitch
		gl.glRotatef(np.degrees(creature.orientation[1]), 0, 1, 0)  # Yaw
		gl.glRotatef(np.degrees(creature.orientation[2]), 0, 0, 1)  # Roll
		
		# Si des muscles sont activés, mettre à jour leurs effets sur les articulations
		if hasattr(creature, 'muscles') and creature.muscles:
			for muscle in creature.muscles:
				if hasattr(muscle, 'activation') and abs(muscle.activation) > 0.01:
					# Mettre à jour la position des articulations en fonction de l'activation musculaire
					self._apply_muscle_effect(creature, muscle, mesh)
		
		gl.glPopMatrix()
	
	def _apply_muscle_effect(self, creature: Creature, muscle: Muscle, mesh: Dict[str, Any]) -> None:
		"""
		Applique l'effet d'un muscle activé sur la position des articulations.
		
		Args:
			creature: Créature concernée
			muscle: Muscle activé
			mesh: Maillage de la créature
		"""
		# Simplification: le muscle tire les articulations l'une vers l'autre proportionnellement à son activation
		if len(muscle.jointIds) < 2 or muscle.activation == 0:
			return
			
		# Récupérer les articulations connectées
		joint_positions = []
		joint_ids = []
		
		for joint_id in muscle.jointIds:
			for joint in creature.joints:
				if joint.id == joint_id:
					joint_positions.append(joint.position)
					joint_ids.append(joint_id)
					break
		
		if len(joint_positions) < 2:
			return
			
		# Force de contraction
		contraction_force = abs(muscle.activation) * 0.1
		
		# Pour chaque paire d'articulations connectées
		for i in range(len(joint_ids) - 1):
			pos1 = joint_positions[i]
			pos2 = joint_positions[i + 1]
			
			# Direction de la contraction
			direction = pos2 - pos1
			distance = np.linalg.norm(direction)
			
			if distance > 1e-6:
				# Normaliser la direction
				direction = direction / distance
				
				# Déplacement en fonction de la contraction
				displacement = direction * contraction_force * distance
				
				# Mettre à jour les positions des articulations dans le maillage
				if joint_ids[i] in mesh["joints"]:
					joint_mesh = mesh["joints"][joint_ids[i]]
					if muscle.activation > 0:
						joint_mesh["position"] = pos1 + displacement
					else:
						joint_mesh["position"] = pos1 - displacement
				
				if joint_ids[i + 1] in mesh["joints"]:
					joint_mesh = mesh["joints"][joint_ids[i + 1]]
					if muscle.activation > 0:
						joint_mesh["position"] = pos2 - displacement
					else:
						joint_mesh["position"] = pos2 + displacement
	
	def _render_creature(self, creature_id: str, render_mode: RenderMode, highlight_level: float) -> None:
		"""
		Effectue le rendu d'une créature spécifique.
		
		Args:
			creature_id: ID de la créature à rendre
			render_mode: Mode de rendu à utiliser
			highlight_level: Niveau de mise en évidence (0.0 à 1.0)
		"""
		if creature_id not in self.creature_meshes or not self.world:
			return
			
		creature = self.world.creatures.get(creature_id)
		if not creature:
			return
			
		mesh = self.creature_meshes[creature_id]
		
		# Sauvegarder l'état
		gl.glPushMatrix()
		
		# Appliquer la position globale
		gl.glTranslatef(creature.position[0], creature.position[1], creature.position[2])
		
		# Appliquer l'orientation (angles d'Euler)
		gl.glRotatef(np.degrees(creature.orientation[0]), 1, 0, 0)  # Pitch
		gl.glRotatef(np.degrees(creature.orientation[1]), 0, 1, 0)  # Yaw
		gl.glRotatef(np.degrees(creature.orientation[2]), 0, 0, 1)  # Roll
		
		# Calcul de la couleur avec mise en évidence
		base_color = mesh["base_color"]
		if highlight_level > 0:
			# Éclaircir la couleur de base pour la mise en évidence
			highlight_color = tuple(min(1.0, c * (1.0 + highlight_level * 0.5)) for c in base_color)
		else:
			highlight_color = base_color
		
		# Rendre les membres
		if self.show_limbs:
			for limb_id, limb_data in mesh["limbs"].items():
				self._render_limb(limb_data, render_mode, highlight_color, self.x_ray_view)
		
		# Rendre les articulations
		if self.show_joints:
			for joint_id, joint_data in mesh["joints"].items():
				self._render_joint(joint_data, render_mode, highlight_color, self.x_ray_view)
		
		# Rendre les muscles
		if self.show_muscles:
			for muscle_id, muscle_data in mesh["muscles"].items():
				self._render_muscle(muscle_data, render_mode, highlight_color, self.x_ray_view)
		
		# Rendre les capteurs
		if self.show_sensors:
			for sensor_id, sensor_data in mesh["sensors"].items():
				self._render_sensor(sensor_data, render_mode, highlight_color, self.x_ray_view)
		
		# Restaurer l'état
		gl.glPopMatrix()
	
	def _render_joint(self, joint_data: Dict[str, Any], render_mode: RenderMode, 
					highlight_color: Tuple[float, float, float], x_ray: bool) -> None:
		"""
		Effectue le rendu d'une articulation.
		
		Args:
			joint_data: Données de l'articulation
			render_mode: Mode de rendu à utiliser
			highlight_color: Couleur de mise en évidence
			x_ray: Si True, rendu en mode transparence
		"""
		position = joint_data["position"]
		radius = joint_data["radius"]
		color = joint_data["color"]
		
		# Appliquer la mise en évidence
		render_color = tuple(c * h for c, h in zip(color, highlight_color))
		
		# Sauvegarder l'état
		gl.glPushMatrix()
		
		# Positionner l'articulation
		gl.glTranslatef(position[0], position[1], position[2])
		gl.glScalef(radius, radius, radius)
		
		# Configurer le matériau
		if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, 
						 [render_color[0], render_color[1], render_color[2], 
						 0.7 if x_ray else 1.0])
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
			gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 50.0)
			
			# Appliquer une texture si nécessaire
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["bone"])
		else:
			gl.glColor4f(render_color[0], render_color[1], render_color[2], 
					   0.7 if x_ray else 1.0)
		
		# Dessiner la sphère
		gl.glCallList(self.display_lists["sphere"])
		
		# Restaurer l'état
		gl.glPopMatrix()
	
	def _render_limb(self, limb_data: Dict[str, Any], render_mode: RenderMode, 
				   highlight_color: Tuple[float, float, float], x_ray: bool) -> None:
		"""
		Effectue le rendu d'un membre.
		
		Args:
			limb_data: Données du membre
			render_mode: Mode de rendu à utiliser
			highlight_color: Couleur de mise en évidence
			x_ray: Si True, rendu en mode transparence
		"""
		start_pos = limb_data["start_pos"]
		end_pos = limb_data["end_pos"]
		width = limb_data["width"]
		shape = limb_data["shape"]
		color = limb_data["color"]
		
		# Vecteur du membre
		direction = end_pos - start_pos
		length = np.linalg.norm(direction)
		
		if length < 1e-6:
			return
			
		# Normaliser la direction
		direction = direction / length
		
		# Appliquer la mise en évidence
		render_color = tuple(c * h for c, h in zip(color, highlight_color))
		
		# Sauvegarder l'état
		gl.glPushMatrix()
		
		# Positionner le membre
		gl.glTranslatef(start_pos[0], start_pos[1], start_pos[2])
		
		# Orienter le membre vers la direction cible
		if abs(direction[2]) < 0.999:  # Éviter la singularité quand z = 1
			# Angle entre la direction par défaut du cylindre (axe z) et la direction cible
			angle = math.degrees(math.acos(direction[2]))
			
			# Axe de rotation (perpendiculaire aux deux directions)
			rotation_axis = np.cross([0, 0, 1], direction)
			rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
			
			gl.glRotatef(angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
		elif direction[2] < 0:
			# Si la direction est exactement opposée à z, tourner de 180° autour de x
			gl.glRotatef(180, 1, 0, 0)
		
		# Configurer le matériau
		if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, 
						 [render_color[0], render_color[1], render_color[2], 
						 0.7 if x_ray else 1.0])
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.5, 0.5, 0.5, 1.0])
			gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 30.0)
			
			# Appliquer une texture si nécessaire
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["skin"])
		else:
			gl.glColor4f(render_color[0], render_color[1], render_color[2], 
					   0.7 if x_ray else 1.0)
		
		# Choisir le rendu en fonction de la forme du membre
		if shape == "cylinder" or shape == "tentacle":
			# Dessiner un cylindre
			gl.glScalef(width, width, length)
			gl.glCallList(self.display_lists["cylinder"])
		elif shape == "fin" or shape == "paddle":
			# Dessiner une nageoire plate
			gl.glBegin(gl.GL_TRIANGLE_FAN)
			
			# Centre de la nageoire
			gl.glTexCoord2f(0.5, 0.5)
			gl.glNormal3f(0, 1, 0)
			gl.glVertex3f(0, 0, length / 2)
			
			# Points sur le bord de la nageoire
			segments = 12
			for i in range(segments + 1):
				angle = 2 * math.pi * i / segments
				x = width * math.cos(angle)
				y = 0.1 * width  # Hauteur de la nageoire
				z = length * (0.5 + 0.5 * math.sin(angle))
				
				gl.glTexCoord2f(0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle))
				gl.glNormal3f(0, 1, 0)
				gl.glVertex3f(x, y, z)
			
			gl.glEnd()
		else:
			# Par défaut, utiliser un cylindre
			gl.glScalef(width, width, length)
			gl.glCallList(self.display_lists["cylinder"])
		
		# Restaurer l'état
		gl.glPopMatrix()
	
	def _render_muscle(self, muscle_data: Dict[str, Any], render_mode: RenderMode, 
					 highlight_color: Tuple[float, float, float], x_ray: bool) -> None:
		"""
		Effectue le rendu d'un muscle.
		
		Args:
			muscle_data: Données du muscle
			render_mode: Mode de rendu à utiliser
			highlight_color: Couleur de mise en évidence
			x_ray: Si True, rendu en mode transparence
		"""
		path = muscle_data["path"]
		thickness = muscle_data["thickness"]
		color = muscle_data["color"]
		activation = muscle_data["activation"]
		
		if len(path) < 2:
			return
			
		# Appliquer la mise en évidence
		render_color = tuple(c * h for c, h in zip(color, highlight_color))
		
		# Configurer le matériau
		if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, 
						 [render_color[0], render_color[1], render_color[2], 
						 0.7 if x_ray else 1.0])
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [0.7, 0.3, 0.3, 1.0])
			gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 40.0)
			
			# Appliquer une texture si nécessaire
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["muscle"])
		else:
			gl.glColor4f(render_color[0], render_color[1], render_color[2], 
					   0.7 if x_ray else 1.0)
		
		# Dessiner le muscle en segments
		for i in range(len(path) - 1):
			start = path[i]
			end = path[i + 1]
			
			# Vecteur du segment
			direction = end - start
			length = np.linalg.norm(direction)
			
			if length < 1e-6:
				continue
				
			# Normaliser la direction
			direction = direction / length
			
			# Sauvegarder l'état
			gl.glPushMatrix()
			
			# Positionner le segment
			gl.glTranslatef(start[0], start[1], start[2])
			
			# Orienter le segment vers la direction cible
			if abs(direction[2]) < 0.999:  # Éviter la singularité quand z = 1
				angle = math.degrees(math.acos(direction[2]))
				rotation_axis = np.cross([0, 0, 1], direction)
				rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
				gl.glRotatef(angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
			elif direction[2] < 0:
				gl.glRotatef(180, 1, 0, 0)
			
			# Dessiner un segment de cylindre
			# Varier l'épaisseur selon l'activation (contracté au milieu, détendu aux extrémités)
			segment_pos = i / (len(path) - 2)  # Position relative dans le muscle
			segment_thickness = thickness * (1.0 + 0.2 * abs(activation) * math.sin(segment_pos * math.pi))
			
			gl.glScalef(segment_thickness, segment_thickness, length)
			gl.glCallList(self.display_lists["cylinder"])
			
			# Restaurer l'état
			gl.glPopMatrix()
	
	def _render_sensor(self, sensor_data: Dict[str, Any], render_mode: RenderMode, 
					 highlight_color: Tuple[float, float, float], x_ray: bool) -> None:
		"""
		Effectue le rendu d'un capteur.
		
		Args:
			sensor_data: Données du capteur
			render_mode: Mode de rendu à utiliser
			highlight_color: Couleur de mise en évidence
			x_ray: Si True, rendu en mode transparence
		"""
		position = sensor_data["position"]
		direction = sensor_data["direction"]
		radius = sensor_data["radius"]
		sensor_type = sensor_data["type"]
		color = sensor_data["color"]
		
		# Appliquer la mise en évidence
		render_color = tuple(c * h for c, h in zip(color, highlight_color))
		
		# Sauvegarder l'état
		gl.glPushMatrix()
		
		# Positionner le capteur
		gl.glTranslatef(position[0], position[1], position[2])
		
		# Orienter le capteur selon sa direction (pour les capteurs directionnels)
		if np.linalg.norm(direction) > 1e-6:
			direction = direction / np.linalg.norm(direction)
			
			if abs(direction[2]) < 0.999:
				angle = math.degrees(math.acos(direction[2]))
				rotation_axis = np.cross([0, 0, 1], direction)
				rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
				gl.glRotatef(angle, rotation_axis[0], rotation_axis[1], rotation_axis[2])
			elif direction[2] < 0:
				gl.glRotatef(180, 1, 0, 0)
		
		# Configurer le matériau
		if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, 
						 [render_color[0], render_color[1], render_color[2], 
						 0.7 if x_ray else 1.0])
			gl.glMaterialfv(gl.GL_FRONT, gl.GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])
			gl.glMaterialf(gl.GL_FRONT, gl.GL_SHININESS, 80.0)
			
			# Appliquer une texture si nécessaire
			if render_mode == RenderMode.TEXTURED:
				gl.glBindTexture(gl.GL_TEXTURE_2D, self.textures["sensor"])
		else:
			gl.glColor4f(render_color[0], render_color[1], render_color[2], 
					   0.7 if x_ray else 1.0)
		
		# Dessiner le capteur en fonction de son type
		if sensor_type == "vision":
			# Capteur de vision: sphère avec un "œil"
			gl.glScalef(radius, radius, radius)
			gl.glCallList(self.display_lists["sphere"])
			
			# Dessiner l'iris
			gl.glPushMatrix()
			gl.glTranslatef(0, 0, 0.7)
			gl.glScalef(0.5, 0.5, 0.1)
			
			if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, [0.1, 0.1, 0.1, 1.0])
			else:
				gl.glColor3f(0.1, 0.1, 0.1)
				
			gl.glCallList(self.display_lists["sphere"])
			gl.glPopMatrix()
			
		elif sensor_type == "proximity" or sensor_type == "electromagnetic":
			# Capteur avec antenne
			gl.glScalef(radius, radius, radius)
			gl.glCallList(self.display_lists["sphere"])
			
			# Dessiner l'antenne
			gl.glPushMatrix()
			gl.glTranslatef(0, 0, 0)
			gl.glRotatef(90, 1, 0, 0)
			gl.glScalef(0.1, 0.1, 2.0)
			
			if render_mode == RenderMode.SHADED or render_mode == RenderMode.TEXTURED:
				gl.glMaterialfv(gl.GL_FRONT, gl.GL_AMBIENT_AND_DIFFUSE, 
							 [render_color[0]*0.8, render_color[1]*0.8, render_color[2]*0.8, 1.0])
			else:
				gl.glColor3f(render_color[0]*0.8, render_color[1]*0.8, render_color[2]*0.8)
				
			gl.glCallList(self.display_lists["cylinder"])
			gl.glPopMatrix()
			
		else:
			# Autres types de capteurs: simple sphère
			gl.glScalef(radius, radius, radius)
			gl.glCallList(self.display_lists["sphere"])
		
		# Restaurer l'état
		gl.glPopMatrix()
		
		# Dessiner le champ de détection pour certains capteurs
		if render_mode != RenderMode.WIREFRAME and sensor_type in ["vision", "proximity"]:
			self._render_sensor_field(position, direction, sensor_type, radius, render_color)
	
	def _render_sensor_field(self, position: np.ndarray, direction: np.ndarray, 
						   sensor_type: str, radius: float, color: Tuple[float, float, float]) -> None:
		"""
		Rend le champ de détection d'un capteur.
		
		Args:
			position: Position du capteur
			direction: Direction du capteur
			sensor_type: Type de capteur
			radius: Rayon du capteur
			color: Couleur du capteur
		"""
		# Normaliser la direction
		if np.linalg.norm(direction) < 1e-6:
			return
			
		norm_direction = direction / np.linalg.norm(direction)
		
		# Configurer le mode transparent
		gl.glEnable(gl.GL_BLEND)
		gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
		gl.glDepthMask(gl.GL_FALSE)
		
		# Désactiver l'éclairage pour le champ de détection
		gl.glDisable(gl.GL_LIGHTING)
		
		# Configuration spécifique au type de capteur
		if sensor_type == "vision":
			# Champ de vision conique
			fov = 60  # Degrés
			distance = 10.0 * radius  # Portée
			
			gl.glBegin(gl.GL_TRIANGLE_FAN)
			
			# Sommet du cône (position du capteur)
			gl.glColor4f(color[0], color[1], color[2], 0.1)
			gl.glVertex3f(position[0], position[1], position[2])
			
			# Base du cône
			segments = 16
			for i in range(segments + 1):
				angle = 2 * math.pi * i / segments
				
				# Calculer un vecteur perpendiculaire à la direction
				if abs(norm_direction[1]) < 0.9:
					perpendicular = np.cross(norm_direction, [0, 1, 0])
				else:
					perpendicular = np.cross(norm_direction, [1, 0, 0])
					
				perpendicular = perpendicular / np.linalg.norm(perpendicular)
				
				# Vecteur de rotation pour créer le cône
				rotation = perpendicular * math.cos(angle) + np.cross(norm_direction, perpendicular) * math.sin(angle)
				
				# Vecteur du rayon à l'angle donné
				ray = norm_direction * math.cos(math.radians(fov/2)) + rotation * math.sin(math.radians(fov/2))
				
				# Point sur la base du cône
				point = position + ray * distance
				
				gl.glColor4f(color[0], color[1], color[2], 0.0)
				gl.glVertex3f(point[0], point[1], point[2])
			
			gl.glEnd()
			
		elif sensor_type == "proximity":
			# Champ de proximité sphérique
			distance = 5.0 * radius  # Portée
			
			gl.glPushMatrix()
			gl.glTranslatef(position[0], position[1], position[2])
			
			# Dessiner une sphère transparente
			gl.glColor4f(color[0], color[1], color[2], 0.05)
			
			quad = glu.gluNewQuadric()
			glu.gluQuadricTexture(quad, gl.GL_FALSE)
			glu.gluSphere(quad, distance, 16, 16)
			glu.gluDeleteQuadric(quad)
			
			gl.glPopMatrix()
		
		# Restaurer les états
		gl.glEnable(gl.GL_LIGHTING)
		gl.glDepthMask(gl.GL_TRUE)
		gl.glDisable(gl.GL_BLEND)
	
	def cleanup(self) -> None:
		"""
		Nettoie les ressources OpenGL (textures, listes d'affichage).
		"""
		try:
			# Supprimer les textures
			if self.textures:
				texture_ids = list(self.textures.values())
				gl.glDeleteTextures(texture_ids)
				self.textures = {}
			
			# Supprimer les listes d'affichage
			if self.display_lists:
				for list_id in self.display_lists.values():
					gl.glDeleteLists(list_id, 1)
				self.display_lists = {}
				
			self.logger.info("Ressources du CreatureRenderer nettoyées", module="visualization")
		except Exception as e:
			self.logger.error(f"Erreur lors du nettoyage des ressources: {str(e)}", 
							module="visualization", exc_info=True)
	
	def set_render_options(self, 
						 show_joints: bool = True, 
						 show_limbs: bool = True, 
						 show_muscles: bool = True, 
						 show_sensors: bool = True, 
						 x_ray_view: bool = False, 
						 highlight_selected: bool = True) -> None:
		"""
		Configure les options de rendu.
		
		Args:
			show_joints: Afficher les articulations
			show_limbs: Afficher les membres
			show_muscles: Afficher les muscles
			show_sensors: Afficher les capteurs
			x_ray_view: Activer le mode vision aux rayons X
			highlight_selected: Mettre en évidence l'entité sélectionnée
		"""
		self.show_joints = show_joints
		self.show_limbs = show_limbs
		self.show_muscles = show_muscles
		self.show_sensors = show_sensors
		self.x_ray_view = x_ray_view
		self.highlight_selected = highlight_selected