import numpy as np

def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' given three points a, b, and c.
    Points are passed as [x, y] lists.
    """
    a = np.array(a) # Shoulder
    b = np.array(b) # Elbow
    c = np.array(c) # Wrist
    
    # Calculate the angle in radians and convert to degrees
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # Ensure the angle is the inner angle (0-180 degrees)
    if angle > 180.0:
        angle = 360 - angle
        
    return angle