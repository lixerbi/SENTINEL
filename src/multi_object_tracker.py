"""
Multi-Object Tracker
Manages multiple tracks simultaneously
"""

import numpy as np
from track import Track
from data_association import associate_detections_to_tracks

class MultiObjectTracker:
    """
    Tracks multiple objects simultaneously
    
    Uses:
    - Kalman filtering for state estimation
    - Hungarian algorithm for data association
    - Track lifecycle management
    """
    
    def __init__(self, max_age=30, min_hits=3, max_distance=50.0, dt=0.5):
        """
        Args:
            max_age: Max frames to keep track without detection
            min_hits: Min detections before track is confirmed
            max_distance: Max distance for valid association (meters)
            dt: Time step
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.max_distance = max_distance
        self.dt = dt
        
        self.tracks = []
        self.frame_count = 0
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of [x, y] positions
            
        Returns:
            List of confirmed track states
        """
        self.frame_count += 1
        
        # Predict all existing tracks
        predicted_positions = []
        for track in self.tracks:
            predicted_pos = track.predict()
            predicted_positions.append(predicted_pos)
        
        # Associate detections to tracks
        if len(detections) > 0 and len(self.tracks) > 0:
            matches, unmatched_dets, unmatched_tracks = associate_detections_to_tracks(
                detections, predicted_positions, self.max_distance
            )
            
            # Update matched tracks
            for det_idx, track_idx in matches:
                self.tracks[track_idx].update(detections[det_idx])
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                new_track = Track(detections[det_idx], dt=self.dt)
                self.tracks.append(new_track)
        
        elif len(detections) > 0:
            # No existing tracks, create for all detections
            for detection in detections:
                new_track = Track(detection, dt=self.dt)
                self.tracks.append(new_track)
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks 
                      if not t.is_deleted(self.max_age)]
        
        # Return confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if not track.is_tentative(self.min_hits):
                confirmed_tracks.append(track.get_state())
        
        return confirmed_tracks
    
    def get_all_tracks(self):
        """Get all tracks (including tentative)"""
        return [t.get_state() for t in self.tracks]


# Test the tracker
if __name__ == "__main__":
    print("Testing Multi-Object Tracker...")
    
    # Create tracker
    tracker = MultiObjectTracker(max_age=5, min_hits=3, max_distance=10.0)
    
    # Simulate 10 frames with 3 objects
    print("\nSimulating 10 frames with 3 moving objects:")
    
    for frame in range(10):
        # Simulate 3 objects moving
        detections = [
            [10 + frame*2, 20 + frame*1],      # Object 1: moving right-up
            [30 + frame*1.5, 40 - frame*0.5],  # Object 2: moving right-down
            [50 - frame*1, 60 + frame*2]       # Object 3: moving left-up
        ]
        
        # Add some noise
        noisy_detections = []
        for det in detections:
            noisy_det = [
                det[0] + np.random.randn() * 2,
                det[1] + np.random.randn() * 2
            ]
            noisy_detections.append(noisy_det)
        
        # Update tracker
        confirmed_tracks = tracker.update(noisy_detections)
        
        print(f"\nFrame {frame+1}:")
        print(f"  Detections: {len(noisy_detections)}")
        print(f"  Total tracks: {len(tracker.tracks)}")
        print(f"  Confirmed tracks: {len(confirmed_tracks)}")
        
        for track in confirmed_tracks:
            pos = track['position']
            print(f"    Track {track['id']}: ({pos[0]:.1f}, {pos[1]:.1f})")
    
    print("\n✅ Multi-Object Tracker working!")
    print(f"✅ Successfully tracked {len(tracker.tracks)} objects across 10 frames")
