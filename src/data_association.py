"""
Data Association for Multi-Object Tracking
Uses Hungarian Algorithm for optimal assignment
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def compute_cost_matrix(detections, predicted_positions, max_distance=50.0):
    """
    Compute cost matrix for Hungarian algorithm
    
    Args:
        detections: List of detection positions [[x, y], ...]
        predicted_positions: List of predicted track positions [[x, y], ...]
        max_distance: Maximum distance for valid association (meters)
        
    Returns:
        cost_matrix: 2D array of distances
    """
    n_detections = len(detections)
    n_tracks = len(predicted_positions)
    
    if n_detections == 0 or n_tracks == 0:
        return np.array([])
    
    cost_matrix = np.zeros((n_detections, n_tracks))
    
    for i, det in enumerate(detections):
        for j, track in enumerate(predicted_positions):
            distance = euclidean_distance(det, track)
            # If too far, make cost very high (unlikely match)
            if distance > max_distance:
                cost_matrix[i, j] = 1e9
            else:
                cost_matrix[i, j] = distance
    
    return cost_matrix

def associate_detections_to_tracks(detections, predicted_positions, 
                                   max_distance=50.0):
    """
    Associate detections to tracks using Hungarian algorithm
    
    Args:
        detections: List of detection positions [[x, y], ...]
        predicted_positions: List of predicted track positions [[x, y], ...]
        max_distance: Maximum distance for valid match
        
    Returns:
        matches: List of (detection_idx, track_idx) tuples
        unmatched_detections: List of detection indices with no match
        unmatched_tracks: List of track indices with no match
    """
    # Handle empty cases
    if len(detections) == 0:
        return [], [], list(range(len(predicted_positions)))
    
    if len(predicted_positions) == 0:
        return [], list(range(len(detections))), []
    
    # Compute cost matrix
    cost_matrix = compute_cost_matrix(detections, predicted_positions, max_distance)
    
    # Solve assignment problem using Hungarian algorithm
    det_indices, track_indices = linear_sum_assignment(cost_matrix)
    
    # Filter out matches that are too far apart
    matches = []
    unmatched_detections = list(range(len(detections)))
    unmatched_tracks = list(range(len(predicted_positions)))
    
    for det_idx, track_idx in zip(det_indices, track_indices):
        if cost_matrix[det_idx, track_idx] < max_distance:
            # Valid match
            matches.append((det_idx, track_idx))
            unmatched_detections.remove(det_idx)
            unmatched_tracks.remove(track_idx)
    
    return matches, unmatched_detections, unmatched_tracks

# Test the implementation
if __name__ == "__main__":
    print("Testing Data Association...")
    
    # Example: 3 tracks, 4 detections
    tracks = [[10, 20], [30, 40], [50, 60]]
    detections = [[12, 22], [31, 41], [48, 58], [70, 80]]
    
    matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
        detections, tracks, max_distance=10.0
    )
    
    print("\n📊 ASSOCIATION RESULTS:")
    print(f"\nMatches: {len(matches)}")
    for det_idx, track_idx in matches:
        print(f"  Detection {det_idx} → Track {track_idx}")
    
    print(f"\nUnmatched Detections: {unmatched_dets}")
    print(f"Unmatched Tracks: {unmatched_tracks}")
    
    print("\n✅ Data association working!")
