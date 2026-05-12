import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json
import os

class RobotCleaningAnalyzer:
    """
    Classe pour analyser les surfaces et générer des trajectoires de nettoyage robotisé
    Supporte les fichiers STL
    """
    
    def __init__(self, nozzle_distance: float = 0.4, overlap: float = 0.15):
        self.nozzle_distance = nozzle_distance
        self.overlap = overlap
        self.robot_params = {
            'nozzle_distance': nozzle_distance,
            'overlap': overlap,
            'sweep_speed': 0.2,  # m/s
            'max_pressure': 1000,  # Bar
            'water_flow': 20,  # L/min
            'safety_margin': 0.1  # m
        }
    
    def load_and_prepare_stl(self, stl_path: str) -> o3d.geometry.PointCloud:
        """
        Charge un fichier STL et le convertit en nuage de points pour l'analyse
        """
        print(f"Chargement du fichier STL: {stl_path}")
        
        # Chargement du fichier STL
        mesh = o3d.io.read_triangle_mesh(stl_path)
        
        if len(mesh.vertices) == 0:
            raise ValueError(f"Impossible de charger le fichier STL: {stl_path}")
        
        # Vérification du chargement
        print(f"Maillage chargé: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Nettoyage du maillage
        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_non_manifold_edges()
        
        # Calcul des normales si absentes
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        # Échantillonnage pour créer un nuage de points dense
        # Augmentation du nombre de points pour les STL complexes
        pcd = mesh.sample_points_poisson_disk(number_of_points=100000)
        
        # Estimation des normales pour le nuage de points
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Orientation cohérente des normales
        pcd.orient_normals_consistent_tangent_plane(k=30)
        
        print(f"Nuage de points généré: {len(pcd.points)} points")
        
        # Visualisation automatique du chargement initial
        print("\n Visualisation du nuage de points chargé...")
        o3d.visualization.draw_geometries([pcd], window_name="Nuage de points chargé")
        
        return pcd
    
    def detect_all_planes(self, pcd: o3d.geometry.PointCloud, 
                         max_planes: int = 15,  # Augmenté pour STL complexe
                         min_points: int = 300,  # Réduit pour plus de sensibilité
                         distance_threshold: float = 0.01) -> List[Dict]:  # Seuil plus strict
        """
        Détecte tous les plans significatifs dans le nuage de points STL
        """
        planes = []
        remaining_pcd = pcd
        plane_index = 0
        
        print("Début de la détection des plans...")
        
        while len(remaining_pcd.points) > min_points and plane_index < max_planes:
            try:
                # Segmentation RANSAC avec paramètres adaptés pour STL
                plane_model, inliers = remaining_pcd.segment_plane(
                    distance_threshold=distance_threshold,
                    ransac_n=3,
                    num_iterations=2000  # Augmenté pour plus de robustesse
                )
                
                if len(inliers) < min_points:
                    print(f"Arrêt: plus assez de points ({len(inliers)} < {min_points})")
                    break
                
                # Extraire le plan détecté
                plane_cloud = remaining_pcd.select_by_index(inliers)
                
                # Calculer les caractéristiques du plan
                bbox = plane_cloud.get_axis_aligned_bounding_box()
                center = plane_cloud.get_center()
                
                # Calcul de l'orientation
                a, b, c, d = plane_model
                normal = np.array([a, b, c])
                normal_norm = normal / np.linalg.norm(normal)
                
                # Calcul de la surface approximative
                points = np.asarray(plane_cloud.points)
                if len(points) > 0:
                    # Projection sur le plan pour calculer la surface
                    cov_matrix = np.cov(points.T)
                    eigenvalues = np.linalg.eigvals(cov_matrix)
                    surface_area = np.sqrt(np.prod(eigenvalues[eigenvalues > 0.001])) * 10
                else:
                    surface_area = 0
                
                plane_info = {
                    'id': plane_index,
                    'model': plane_model,
                    'normal': normal_norm.tolist(),
                    'cloud': plane_cloud,
                    'inlier_count': len(inliers),
                    'bbox': bbox,
                    'center': center,
                    'points': points,
                    'surface_area': surface_area
                }
                
                planes.append(plane_info)
                print(f"Plan {plane_index} détecté: {len(inliers)} points, surface: {surface_area:.3f} m²")
                
                # Mettre à jour le nuage restant
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                plane_index += 1
                
            except Exception as e:
                print(f"Erreur lors de la détection du plan {plane_index}: {e}")
                break
        
        print(f"Détection terminée: {len(planes)} plans trouvés")
        return planes
    
    def classify_surfaces(self, planes: List[Dict]) -> Dict:
        """
        Classe les surfaces selon leur orientation et accessibilité
        Adaptation pour pièces industrielles STL
        """
        classified = {
            'vertical': [],
            'horizontal_up': [],
            'horizontal_down': [],
            'inclined': [],
            'hard_to_reach': [],
            'small_features': []  # Nouvelle catégorie pour les petites surfaces
        }
        
        for plane in planes:
            normal = np.array(plane['normal'])
            
            # Filtrer les très petites surfaces (bruit)
            if plane['surface_area'] < 0.1:  # moins de 0.1 m²
                plane['type'] = 'small_features'
                plane['orientation'] = 'petite_surface'
                classified['small_features'].append(plane)
                continue
            
            # Déterminer l'orientation
            if abs(normal[2]) < 0.3:  # Verticale (seuil élargi)
                plane['type'] = 'vertical'
                plane['orientation'] = 'vertical'
                classified['vertical'].append(plane)
                
            elif normal[2] > 0.7:  # Horizontale vers le haut (plafond)
                plane['type'] = 'horizontal_up'
                plane['orientation'] = 'plafond'
                classified['horizontal_up'].append(plane)
                
            elif normal[2] < -0.7:  # Horizontale vers le bas (sol)
                plane['type'] = 'horizontal_down'
                plane['orientation'] = 'sol'
                classified['horizontal_down'].append(plane)
                
            else:  # Inclinée
                plane['type'] = 'inclined'
                angle = np.degrees(np.arccos(abs(normal[2])))
                plane['orientation'] = f'incliné_{angle:.1f}°'
                classified['inclined'].append(plane)
            
            # Analyser la difficulté d'accès
            plane['accessibility'] = self.calculate_accessibility(plane)
            if plane['accessibility'] == 'hard':
                classified['hard_to_reach'].append(plane)
        
        return classified
    
    def calculate_accessibility(self, plane: Dict) -> str:
        """
        Évalue la difficulté d'accès d'une surface pour robot de nettoyage
        """
        center = plane['center']
        normal = np.array(plane['normal'])
        
        # Critères de difficulté
        criteria = 0
        
        # Surface verticale très haute
        if plane['type'] == 'vertical' and center[2] > 1.5:
            criteria += 1
        
        # Surface de plafond (difficile d'accès)
        if plane['type'] == 'horizontal_up':
            criteria += 2
        
        # Surface avec orientation complexe
        if plane['type'] == 'inclined':
            criteria += 1
        
        # Grande surface (nécessite plus de manœuvres)
        if plane['surface_area'] > 5.0:  # m²
            criteria += 1
        
        # Surface dans une position encastrée
        bbox = plane['bbox']
        dimensions = bbox.get_extent()
        if min(dimensions) < 0.5:  # Dimension très petite
            criteria += 1
        
        return 'hard' if criteria >= 2 else 'easy'
    
    def generate_vertical_sweep(self, plane: Dict) -> List[np.ndarray]:
        """
        Génère un pattern de balayage vertical en zigzag
        """
        bbox = plane['bbox']
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        
        # Dimensions de la surface
        width = max_bound[0] - min_bound[0]
        height = max_bound[1] - min_bound[1]
        
        # Paramètres de balayage adaptés à la taille
        sweep_resolution = self.nozzle_distance * (1 - self.overlap)
        num_passes = max(3, int(height / sweep_resolution))  # Minimum 3 passes
        
        waypoints = []
        
        for i in range(num_passes):
            # Position Y (hauteur)
            y = min_bound[1] + i * (height / max(1, num_passes - 1))
            
            if i % 2 == 0:  # Ligne paire : gauche → droite
                x_start, x_end = min_bound[0], max_bound[0]
            else:  # Ligne impaire : droite → gauche
                x_start, x_end = max_bound[0], min_bound[0]
            
            # Points de début et fin de ligne
            start_point = self.calculate_robot_position(plane, x_start, y)
            end_point = self.calculate_robot_position(plane, x_end, y)
            
            if start_point is not None and end_point is not None:
                waypoints.extend([start_point, end_point])
        
        return waypoints
    
    def generate_horizontal_sweep(self, plane: Dict) -> List[np.ndarray]:
        """
        Génère un pattern de balayage horizontal
        """
        bbox = plane['bbox']
        min_bound = bbox.get_min_bound()
        max_bound = bbox.get_max_bound()
        
        width = max_bound[0] - min_bound[0]
        length = max_bound[1] - min_bound[1]
        
        sweep_resolution = self.nozzle_distance * (1 - self.overlap)
        num_passes = max(3, int(width / sweep_resolution))
        
        waypoints = []
        
        for i in range(num_passes):
            # Position X
            x = min_bound[0] + i * (width / max(1, num_passes - 1))
            
            if i % 2 == 0:  # Ligne paire
                y_start, y_end = min_bound[1], max_bound[1]
            else:  # Ligne impaire
                y_start, y_end = max_bound[1], min_bound[1]
            
            start_point = self.calculate_robot_position(plane, x, y_start)
            end_point = self.calculate_robot_position(plane, x, y_end)
            
            if start_point is not None and end_point is not None:
                waypoints.extend([start_point, end_point])
        
        return waypoints
    
    def calculate_robot_position(self, plane: Dict, x: float, y: float) -> np.ndarray:
        """
        Calcule la position du robot pour un point donné sur la surface
        """
        a, b, c, d = plane['model']
        
        # Calcul du point sur le plan
        if abs(c) > 1e-6:
            z = (-d - a*x - b*y) / c
        else:
            # Plan vertical - utiliser le centre Z
            z = plane['center'][2]
        
        # Point sur la surface
        surface_point = np.array([x, y, z])
        
        # Normal au plan (orientée vers l'extérieur)
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # Position du robot (à distance de sécurité)
        # Pour les surfaces horizontales vers le bas, ajuster la distance
        if normal[2] < -0.7:  # Sol
            robot_distance = self.nozzle_distance
        else:
            robot_distance = self.nozzle_distance + self.robot_params['safety_margin']
        
        robot_point = surface_point + normal * robot_distance
        
        return robot_point
    
    def estimate_cleaning_time(self, trajectory: List[np.ndarray]) -> float:
        """
        Estime le temps de nettoyage pour une trajectoire
        """
        if len(trajectory) < 2:
            return 0.0
        
        total_distance = 0
        for i in range(len(trajectory) - 1):
            distance = np.linalg.norm(trajectory[i+1] - trajectory[i])
            total_distance += distance
        
        # Temps de nettoyage basé sur la distance
        cleaning_time = total_distance / self.robot_params['sweep_speed']
        
        # Ajouter du temps pour les repositionnements et sécurités
        total_time = cleaning_time * 1.4  # +40% pour les manœuvres complexes
        
        return total_time
    
    def generate_trajectories(self, classified_surfaces: Dict) -> Dict:
        """
        Génère des trajectoires pour toutes les surfaces classifiées
        Ignore les petites surfaces non significatives
        """
        trajectories = {}
        
        for surface_type, surfaces in classified_surfaces.items():
            trajectories[surface_type] = []
            
            # Ignorer les petites surfaces
            if surface_type == 'small_features':
                continue
                
            for surface in surfaces:
                # Sélection du pattern selon le type de surface
                if surface_type == 'vertical':
                    trajectory = self.generate_vertical_sweep(surface)
                elif surface_type in ['horizontal_up', 'horizontal_down']:
                    trajectory = self.generate_horizontal_sweep(surface)
                else:  # inclined
                    trajectory = self.generate_vertical_sweep(surface)  # Par défaut
                
                # Ne garder que les trajectoires significatives
                if len(trajectory) < 4:  # Au moins 2 lignes de balayage
                    continue
                
                # Calcul du temps estimé
                cleaning_time = self.estimate_cleaning_time(trajectory)
                
                trajectory_info = {
                    'surface_id': surface['id'],
                    'orientation': surface.get('orientation', 'unknown'),
                    'trajectory': [point.tolist() for point in trajectory],
                    'waypoints': len(trajectory),
                    'cleaning_time': cleaning_time,
                    'surface_area': surface['surface_area'],
                    'accessibility': surface['accessibility'],
                    'point_count': surface['inlier_count']
                }
                
                trajectories[surface_type].append(trajectory_info)
        
        return trajectories
    
    def visualize_analysis(self, pcd: o3d.geometry.PointCloud, 
                          classified_surfaces: Dict,
                          trajectories: Dict):
        """
        Visualise l'analyse complète avec couleurs différenciées
        """
        # Créer une visualisation avec les surfaces colorisées
        geometries = []
        
        # Couleurs pour les différents types de surfaces
        colors = {
            'vertical': [1, 0, 0],      # Rouge - surfaces verticales
            'horizontal_up': [0, 1, 0], # Vert - plafonds
            'horizontal_down': [0, 0, 1], # Bleu - sols
            'inclined': [1, 1, 0],      # Jaune - surfaces inclinées
            'hard_to_reach': [1, 0, 1], # Magenta - surfaces difficiles
            'small_features': [0.5, 0.5, 0.5]  # Gris - petites surfaces
        }
        
        # Ajouter chaque surface avec sa couleur
        for surface_type, surfaces in classified_surfaces.items():
            color = colors.get(surface_type, [0.5, 0.5, 0.5])
            for surface in surfaces:
                colored_cloud = surface['cloud']
                colored_cloud.paint_uniform_color(color)
                geometries.append(colored_cloud)
        
        # Ajouter le nuage original en fond (transparent)
        pcd_original = pcd
        pcd_original.paint_uniform_color([0.8, 0.8, 0.8])
        geometries.append(pcd_original)
        
        print("\n Visualisation des surfaces détectées:")
        print("🔴 Rouge: surfaces verticales")
        print("🟢 Vert: plafonds") 
        print("🔵 Bleu: sols")
        print("🟡 Jaune: surfaces inclinées")
        print("🟣 Magenta: surfaces difficiles d'accès")
        print("⚪ Gris: petites surfaces (ignorées)")
        
        # Visualiser automatiquement
        o3d.visualization.draw_geometries(geometries, 
                                         window_name="Analyse des Surfaces à Nettoyer - STL")
    
    def generate_report(self, classified_surfaces: Dict, trajectories: Dict):
        """
        Génère un rapport détaillé de l'analyse pour pièce STL
        """
        print("\n" + "="*70)
        print("RAPPORT D'ANALYSE POUR NETTOYAGE ROBOTISÉ - FICHIER STL")
        print("="*70)
        
        total_surfaces = sum(len(surfaces) for surfaces in classified_surfaces.values())
        total_cleaning_surfaces = sum(len(surfaces) for surface_type, surfaces in classified_surfaces.items() 
                                    if surface_type != 'small_features')
        
        total_time = 0
        total_area = 0
        
        print(f"\nSURFACES IDENTIFIÉES: {total_surfaces} (dont {total_cleaning_surfaces} à nettoyer)")
        
        for surface_type, surfaces in classified_surfaces.items():
            surface_count = len(surfaces)
            total_area_type = sum(surface['surface_area'] for surface in surfaces)
            
            print(f"\n--- {surface_type.upper()} ---")
            print(f"Nombre: {surface_count}")
            print(f"Surface totale: {total_area_type:.2f} m²")
            
            for surface in surfaces:
                status = "⚠️ DIFFICILE" if surface['accessibility'] == 'hard' else "✅ FACILE"
                print(f"  Surface {surface['id']}: {surface['inlier_count']} points, "
                      f"{surface['surface_area']:.2f} m² - {status}")
        
        print(f"\n--- TRAJECTOIRES DE NETTOYAGE GÉNÉRÉES ---")
        
        cleaning_plan = []
        
        for surface_type, traj_list in trajectories.items():
            for traj_info in traj_list:
                total_time += traj_info['cleaning_time']
                total_area += traj_info['surface_area']
                
                cleaning_plan.append({
                    'surface_id': traj_info['surface_id'],
                    'type': surface_type,
                    'orientation': traj_info['orientation'],
                    'area': traj_info['surface_area'],
                    'time': traj_info['cleaning_time'],
                    'waypoints': traj_info['waypoints'],
                    'accessibility': traj_info['accessibility']
                })
                
                print(f"\n Surface {traj_info['surface_id']} ({traj_info['orientation']}):")
                print(f"    Surface: {traj_info['surface_area']:.2f} m²")
                print(f"    Temps estimé: {traj_info['cleaning_time']:.1f} s")
                print(f"    Waypoints: {traj_info['waypoints']}")
                print(f"    Accessibilité: {traj_info['accessibility']}")
        
        # Trier par difficulté et temps
        cleaning_plan.sort(key=lambda x: (x['accessibility'] == 'hard', x['time']), reverse=True)
        
        print(f"\n--- SYNTHÈSE ET RECOMMANDATIONS ---")
        print(f" Temps total de nettoyage estimé: {total_time/60:.1f} minutes")
        print(f" Surface totale à nettoyer: {total_area:.2f} m²")
        print(f" Nombre total de waypoints: {sum(traj_info['waypoints'] for traj_list in trajectories.values() for traj_info in traj_list)}")
        
        # Recommandations
        hard_surfaces = [s for s in cleaning_plan if s['accessibility'] == 'hard']
        if hard_surfaces:
            print(f"\n  SURFACES DIFFICILES REQUÉRANT UNE ATTENTION PARTICULIÈRE:")
            for surface in hard_surfaces:
                print(f"   - Surface {surface['surface_id']} ({surface['orientation']}): "
                      f"{surface['time']:.1f}s, {surface['area']:.2f} m²")
        
        # Sauvegarder le rapport dans un fichier
        report_data = {
            'input_file': 'banche.stl',
            'total_cleaning_time_minutes': total_time / 60,
            'total_surface_area_m2': total_area,
            'total_waypoints': sum(traj_info['waypoints'] for traj_list in trajectories.values() for traj_info in traj_list),
            'surfaces_by_type': {k: len(v) for k, v in classified_surfaces.items()},
            'cleaning_plan': cleaning_plan,
            'robot_parameters': self.robot_params
        }
        
        with open('outputs/cleaning_report_banche_stl.json', 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n Rapport détaillé sauvegardé dans: outputs/cleaning_report_banche_stl.json")
        
    
    def analyze_cleaning(self, stl_path: str):
        """
        Méthode principale pour analyser le nettoyage d'une pièce STL
        """
        print(" Début de l'analyse pour le nettoyage robotisé (STL)...")
        
        try:
            # 1. Chargement et préparation (avec visualisation automatique)
            pcd = self.load_and_prepare_stl(stl_path)
            
            # 2. Détection des plans
            planes = self.detect_all_planes(pcd)
            
            if not planes:
                print(" Aucun plan détecté. Arrêt de l'analyse.")
                return
            
            # 3. Classification des surfaces
            classified_surfaces = self.classify_surfaces(planes)
            
            # 4. Génération des trajectoires
            trajectories = self.generate_trajectories(classified_surfaces)
            
            # 5. Visualisation automatique
            print("\n Génération de la visualisation des surfaces classifiées...")
            self.visualize_analysis(pcd, classified_surfaces, trajectories)
            
            # 6. Rapport
            self.generate_report(classified_surfaces, trajectories)
            
            print("\n Analyse terminée avec succès!")
            
        except Exception as e:
            print(f" Erreur lors de l'analyse: {e}")
            import traceback
            traceback.print_exc()


def main():
    """
    Fonction principale pour tester l'analyseur avec fichier STL
    """
    # Créer le dossier outputs
    os.makedirs("outputs", exist_ok=True)
    
    # Initialiser l'analyseur
    analyzer = RobotCleaningAnalyzer(
        nozzle_distance=0.4,
        overlap=0.15
    )
    
    # Analyser la pièce banche.stl
    stl_path = "banche.stl"
    
    if os.path.exists(stl_path):
        print(f" Fichier STL trouvé: {stl_path}")
        analyzer.analyze_cleaning(stl_path)
    else:
        print(f" Fichier {stl_path} non trouvé.")
        print(" Veuillez télécharger le fichier depuis le drive Google fourni.")
        print(" Assurez-vous que le fichier est dans le même dossier que ce script.")


if __name__ == "__main__":
    main()