# Builder Assist : Pipeline de Traitement 3D Automatisé

Ce projet implémente un pipeline complet pour transformer des scans 3D bruts en modèles géométriques validés (jumeaux numériques) pour la planification robotique.

## 🚀 Fonctionnalités Clés
- **Prétraitement :** Nettoyage du bruit via Statistical Outlier Removal (SOR) et optimisation par Voxel Downsampling.
- **Segmentation RANSAC :** Extraction mathématique des plans principaux et des structures d'intérêt.
- **Recalage ICP (Point-to-Plane) :** Alignement millimétrique entre le scan brut et un modèle de référence (STL) pour garantir la conformité géométrique.

## 📂 Structure du Projet
Le dépôt est organisé de manière modulaire pour faciliter l'intégration R&D :
- `src/` : Scripts Python principaux (`main.py`, `robot_cleaning.py`).
- `data/` : Modèles de référence (`banche.stl`).
- `docs/` : Documentation technique détaillée.

## 🛠️ Stack Technique
- **Langage :** Python.
- **Bibliothèques :** Open3D pour le traitement géométrique.
- **Environnement :** Entièrement reproductible via Conda/Pip.

## 🏁 Utilisation
```bash
# Lancer le pipeline principal
python src/main.py
