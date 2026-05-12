import open3d as o3d
import numpy as np
import os

def load_point_cloud(filename):
    """Charge un nuage de points depuis un fichier PLY"""
    pcd = o3d.io.read_point_cloud(filename)
    print(f"Nuage chargé: {len(pcd.points)} points")
    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.05):
    """Applique le downsampling et suppression du bruit"""
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Suppression des outliers statistiques
    pcd_clean, _ = pcd_down.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0
    )
    return pcd_clean

def segment_main_plane(pcd, distance_threshold=0.02):
    """Segmente le plan principal using RANSAC"""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    return plane_model, inliers

def colorize_and_export(pcd, inliers, output_filename):
    """Colorise les points et exporte le résultat"""
    # Créer une copie pour la colorisation
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = pcd.points
    
    # Initialiser les couleurs (rouge par défaut)
    colors = np.zeros((len(pcd.points), 3))
    colors[:, 0] = 1.0  # Rouge
    
    # Coloriser les inliers en vert
    colors[inliers] = [0, 1, 0]  # Vert
    
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    
    # Exporter
    o3d.io.write_point_cloud(output_filename, pcd_colored)
    return pcd_colored

def calculate_plane_dimensions(pcd, inliers):
    """Calcule les dimensions approximatives du plan"""
    inlier_cloud = pcd.select_by_index(inliers)
    points = np.asarray(inlier_cloud.points)
    
    # Calcul de la bounding box
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    width = max_bound[0] - min_bound[0]  # Dimension X
    height = max_bound[1] - min_bound[1] # Dimension Y
    area = width * height
    
    return width, height, area

def visualize_results(pcd_colored, plane_dimensions=None):
    """Visualisation interactive avec informations"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd_colored)
    
    if plane_dimensions:
        print(f"Dimensions: {plane_dimensions[0]:.2f} x {plane_dimensions[1]:.2f}")
        print(f"Surface: {plane_dimensions[2]:.2f} m²")
    
    vis.run()
    vis.destroy_window()
    
def main():
    # Créer le dossier outputs
    os.makedirs("outputs", exist_ok=True)
    
    fichiers = [
        "facade_simple.ply",
        "facade_bruit.ply", 
        "facade_deux_plans.ply"
    ]
    
    for fichier in fichiers:
        print(f"\n--- Traitement de {fichier} ---")
        
        # Chargement
        pcd = load_point_cloud(f"inputs/{fichier}")
        
        # Prétraitement
        pcd_clean = preprocess_point_cloud(pcd)
        
        # Segmentation
        plane_model, inliers = segment_main_plane(pcd_clean)
        print(f"Plan détecté: {len(inliers)} points")
        print(f"Équation du plan: {plane_model}")
        
        # Colorisation et export
        output_name = f"outputs/output_colored_{fichier}"
        pcd_colored = colorize_and_export(pcd_clean, inliers, output_name)
        
        # Calcul des dimensions (bonus)
        dimensions = calculate_plane_dimensions(pcd_clean, inliers)
        
        # Visualisation
        visualize_results(pcd_colored, dimensions)

if __name__ == "__main__":
    main()
    
