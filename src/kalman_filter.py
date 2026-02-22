import numpy as np

class KalmanFilter:
    """
    2D Kalman Filter for object tracking
    
    State vector: [x, y, vx, vy]
    Measurement: [x, y]
    """
    
    def __init__(self, dt=0.5, process_noise=1.0, measurement_noise=5.0):
        """
        Initialize Kalman Filter
        
        Args:
            dt: Time step (seconds)
            process_noise: Process noise standard deviation
            measurement_noise: Measurement noise standard deviation
        """
        self.dt = dt
        
        # State dimension (4: x, y, vx, vy)
        self.state_dim = 4
        
        # Measurement dimension (2: x, y)
        self.meas_dim = 2
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],   # x = x + vx*dt
            [0, 1, 0, dt],   # y = y + vy*dt
            [0, 0, 1, 0],    # vx = vx
            [0, 0, 0, 1]     # vy = vy
        ])
        
        # Measurement matrix (we only measure position)
        self.H = np.array([
            [1, 0, 0, 0],    # measure x
            [0, 1, 0, 0]     # measure y
        ])
        
        # Process noise covariance
        q = process_noise ** 2
        self.Q = np.array([
            [q*dt**4/4, 0, q*dt**3/2, 0],
            [0, q*dt**4/4, 0, q*dt**3/2],
            [q*dt**3/2, 0, q*dt**2, 0],
            [0, q*dt**3/2, 0, q*dt**2]
        ])
        
        # Measurement noise covariance
        r = measurement_noise ** 2
        self.R = np.array([
            [r, 0],
            [0, r]
        ])
        
        # State estimate
        self.x = np.zeros((4, 1))
        
        # Covariance estimate
        self.P = np.eye(4) * 1000  # High initial uncertainty
    
    def predict(self):
        """
        Predict next state
        
        Returns:
            Predicted state [x, y, vx, vy]
        """
        # Predict state: x = F * x
        self.x = self.F @ self.x
        
        # Predict covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, measurement):
        """
        Update state with measurement
        
        Args:
            measurement: [x, y] measurement
            
        Returns:
            Updated state [x, y, vx, vy]
        """
        z = np.array(measurement).reshape(2, 1)
        
        # Innovation (measurement residual): y = z - H*x
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P*H^T*S^(-1)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state: x = x + K*y
        self.x = self.x + K @ y
        
        # Update covariance: P = (I - K*H)*P
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x.copy()
    
    def get_state(self):
        """Get current state estimate"""
        return self.x.copy()
    
    def get_position(self):
        """Get current position estimate [x, y]"""
        return self.x[:2, 0]
    
    def get_velocity(self):
        """Get current velocity estimate [vx, vy]"""
        return self.x[2:, 0]


# Test the filter
if __name__ == "__main__":
    print("Testing Kalman Filter...")
    
    # Create filter
    kf = KalmanFilter(dt=0.5)
    
    # Simulate noisy measurements
    true_pos = [0, 0]
    true_vel = [5, 3]  # 5 m/s in x, 3 m/s in y
    
    measurements = []
    for i in range(10):
        # True position
        true_pos[0] += true_vel[0] * 0.5
        true_pos[1] += true_vel[1] * 0.5
        
        # Add noise
        noisy_meas = [
            true_pos[0] + np.random.randn() * 5,
            true_pos[1] + np.random.randn() * 5
        ]
        measurements.append(noisy_meas)
    
    # Run filter
    print("\nRunning filter on noisy measurements:")
    for i, z in enumerate(measurements):
        kf.predict()
        kf.update(z)
        
        pos = kf.get_position()
        vel = kf.get_velocity()
        
        print(f"Step {i+1}: Meas=({z[0]:.1f}, {z[1]:.1f}) → "
              f"Est=({pos[0]:.1f}, {pos[1]:.1f}), "
              f"Vel=({vel[0]:.1f}, {vel[1]:.1f})")
    
    print("\nKalman Filter working!")
