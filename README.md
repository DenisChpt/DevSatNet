# Évolution Marine par Apprentissage par Renforcement

Ce projet simule l'évolution des espèces marines, leurs structures et leurs moyens de déplacement en utilisant l'apprentissage par renforcement et les algorithmes génétiques. Il permet d'observer comment différentes morphologies et comportements émergent au fil des générations sous l'influence de la sélection naturelle.

## Fonctionnalités

- **Simulation physique réaliste** de l'environnement marin (courants, température, pression, etc.)
- **Créatures avec morphologie adaptative** (articulations, membres, muscles, capteurs)
- **Évolution génétique** avec mutation, croisement et sélection naturelle
- **Apprentissage par renforcement** utilisant l'algorithme A2C (Advantage Actor-Critic)
- **Visualisation 3D** des créatures et de leur évolution
- **Écosystème dynamique** avec différentes espèces et interactions
- **Enregistrement et chargement** de l'état de la simulation

## Structure du projet

```
marine_evolution/
├── config/                      # Fichiers de configuration
├── core/                        # Modules centraux
│   ├── entities/                # Entités de la simulation (créatures, etc.)
│   ├── environment/             # Environnement marin
│   ├── physics/                 # Moteur physique
│   └── genetics/                # Système génétique
├── learning/                    # Modules d'apprentissage
│   ├── models/                  # Modèles neuronaux
│   ├── trainers/               # Algorithmes d'entraînement
│   └── rewards/                 # Fonctions de récompense
├── visualization/               # Outils de visualisation
├── utils/                       # Utilitaires
├── simulation_manager.py        # Gestionnaire de simulation
├── evolution_engine.py          # Moteur d'évolution
├── main.py                      # Point d'entrée principal
└── README.md                    # Documentation
```

## Prérequis

- Python 3.8 ou supérieur
- PyTorch 1.9 ou supérieur
- NumPy
- Box2D (ou PyBullet pour la 3D)
- Pygame et/ou PyOpenGL pour la visualisation
- noise (pour la génération de terrain)

## Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-utilisateur/marine-evolution.git
cd marine-evolution
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Lancez la simulation principale :
```bash
python main.py
```

2. Options de configuration :
```bash
python main.py --population-size 100 --generations 1000 --render True
```

3. Visualisation des résultats :
```bash
python visualization/visualize_evolution.py --data results/evolution_data.json
```

## Paramètres personnalisables

- Taille de la population
- Nombre de générations
- Facteurs environnementaux (température, courants, etc.)
- Paramètres génétiques (taux de mutation, croisement, etc.)
- Hyperparamètres d'apprentissage par renforcement
- Fonctions de récompense

## Comment fonctionne le projet

1. **Initialisation** : Création de la population initiale avec des génomes aléatoires.
2. **Simulation** : Les créatures interagissent avec l'environnement et sont évaluées.
3. **Apprentissage** : Chaque créature apprend à optimiser son comportement via l'algorithme A2C.
4. **Sélection** : Les créatures les plus performantes sont sélectionnées pour la reproduction.
5. **Reproduction** : De nouvelles créatures sont créées par croisement et mutation.
6. **Itération** : Le processus se répète, conduisant à une évolution progressive.

## Extension du projet

Le projet est conçu de manière modulaire pour faciliter les extensions et modifications :

- Ajout de nouveaux types de capteurs ou d'appendices
- Création de nouvelles fonctions de récompense
- Implémentation d'algorithmes d'apprentissage alternatifs
- Intégration de dynamiques écosystémiques plus complexes

## Exemple de résultats

Après plusieurs centaines de générations, on observe généralement :

1. **Convergence morphologique** : Des structures similaires émergent pour des conditions environnementales similaires
2. **Spécialisation** : Différentes espèces se spécialisent pour différentes niches écologiques
3. **Efficacité énergétique** : Les mouvements deviennent plus efficaces au fil du temps
4. **Stratégies adaptatives** : Développement de comportements spécifiques face aux défis environnementaux

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.