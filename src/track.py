"""
Track class - manages individual object tracks
Combines Kalman filter with track lifecycle
"""

import numpy as np
from kalman_filter import KalmanFilter

class Track:
    """
    Single tracked object
    
    Lifecycle:
    1. Tentative (just created, needs confirmation)
    2. Confirmed (seen multiple times)
    3. Deleted (lost for too long)
    """
    
    # Class variable for unique IDs
    next_id = 1
    
    def __init__(self, initial_position, track_id=None, dt=0.5):
        """
        Initialize track with first detection
        
        Args:
            initial_position: [x, y] position
            track_id: Optional ID (auto-assigned if None)
            dt: Time step
        """
        # Assign unique ID
        self.id = track_id if track_id is not None else Track.next_id
        Track.next_id += 1
        
        # Create Kalman filter
        self.kf = KalmanFilter(dt=dt)
        
        # Initialize state with detection
        self.kf.x[0] = initial_position[0]
        self.kf.x[1] = initial_position[1]
        self.kf.x[2] = 0  # Initial velocity = 0
        self.kf.x[3] = 0
        
        # Track metadata
        self.age = 1  # How many frames this track exists
        self.hits = 1  # How many times we've seen this object
        self.time_since_update = 0  # Frames since last detection
        
        # Visual
        self.color = tuple(np.random.randint(50, 255, 3).tolist())
    
    def predict(self):
        """
        Predict next state
        
        Returns:
            Predicted position [x, y]
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.get_position()
    
    def update(self, measurement):
        """
        Update with new detection
        
        Args:
            measurement: [x, y] position
        """
        self.kf.update(measurement)
        self.hits += 1
        self.time_since_update = 0
    
    def get_state(self):
        """Get current state"""
        return {
            'id': self.id,
            'position': self.kf.get_position(),
            'velocity': self.kf.get_velocity(),
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'color': self.color
        }
    
    def is_tentative(self, min_hits=3):
        """Check if track needs more confirmation"""
        return self.hits < min_hits
    
    def is_deleted(self, max_age=30):
        """Check if track should be deleted"""
        return self.time_since_update > max_age


# Test the Track class
if __name__ == "__main__":
    print("Testing Track class...")
    
    # Create track
    track = Track(initial_position=[10, 20], dt=0.5)
    
    print(f"\n✅ Created Track ID: {track.id}")
    print(f"Initial position: {track.get_state()['position']}")
    
    # Simulate 5 time steps
    print("\nSimulating tracking:")
    measurements = [[12, 22], [14, 24], [16, 26], [18, 28], [20, 30]]
    
    for i, meas in enumerate(measurements):
        # Predict
        predicted = track.predict()
        
        # Update
        track.update(meas)
        
        state = track.get_state()
        print(f"Step {i+1}: Pos={state['position']}, Vel={state['velocity']}")
    
    print(f"\n✅ Track stats:")
    print(f"   Age: {track.age} frames")
    print(f"   Hits: {track.hits} detections")
    print(f"   Tentative: {track.is_tentative()}")
    
    print("\n✅ Track class working!")
