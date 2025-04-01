# environment/rendering.py
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class PatientMonitoringVisualization:
    """
    A 3D visualization for the Patient Monitoring environment using PyOpenGL.
    This is a static visualization that shows the current state of the patient's vitals.
    """
    def __init__(self, state=None):
        self.state = state if state is not None else [0, 0, 0, 0]
        self.window_width = 800
        self.window_height = 600
        self.vital_labels = ["Heart Rate", "Blood Pressure", "SpO2", "Temperature"]
        self.vital_states = [
            ["Normal", "Elevated", "Critical"],
            ["Normal", "High", "Very High"],
            ["Normal", "Low", "Very Low"],
            ["Normal", "Fever", "High Fever"]
        ]
        # Colors for normal, warning, critical (RGB)
        self.colors = [
            [0.0, 1.0, 0.0],  # Green for normal
            [1.0, 1.0, 0.0],  # Yellow for warning
            [1.0, 0.0, 0.0]   # Red for critical
        ]

    def init_gl(self):
        """Initialize OpenGL settings"""
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.window_width / self.window_height, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def display(self):
        """Main display function"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Position the camera
        gluLookAt(0, 0, 10, 0, 0, 0, 0, 1, 0)
        
        # Draw title
        self.draw_text("Patient Monitoring System", -3, 3, 0)
        
        # Draw vital signs indicators
        self.draw_vital_signs()
        
        # Draw legend
        self.draw_legend()
        
        glutSwapBuffers()

    def draw_vital_signs(self):
        """Draw visual indicators for each vital sign"""
        for i, vital in enumerate(self.vital_labels):
            # Calculate position (arranged in a grid)
            x = -3 if i % 2 == 0 else 1
            y = 1.5 if i < 2 else -1.5
            
            # Get state and color
            state_value = min(2, max(0, self.state[i]))  # Ensure valid state
            color = self.colors[state_value]
            state_text = self.vital_states[i][state_value]
            
            # Draw box with color indicating state
            glPushMatrix()
            glTranslatef(x, y, 0)
            glColor3f(*color)
            self.draw_box(1.5)
            
            # Draw vital sign label
            glColor3f(1.0, 1.0, 1.0)  # White text
            self.draw_text(vital, -0.5, 0.4, 0.8)
            self.draw_text(state_text, -0.5, 0, 0.8)
            glPopMatrix()

    def draw_box(self, size):
        """Draw a simple cube"""
        half_size = size / 2
        
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(-half_size, -half_size, half_size)
        glVertex3f(half_size, -half_size, half_size)
        glVertex3f(half_size, half_size, half_size)
        glVertex3f(-half_size, half_size, half_size)
        glEnd()

    def draw_legend(self):
        """Draw a legend explaining the colors"""
        glColor3f(1.0, 1.0, 1.0)  # White text
        self.draw_text("Legend:", -5, -3, 0)
        
        for i, state in enumerate(["Normal", "Warning", "Critical"]):
            y = -3.5 - (i * 0.5)
            glColor3f(*self.colors[i])
            glBegin(GL_QUADS)
            glVertex3f(-5, y, 0)
            glVertex3f(-4.7, y, 0)
            glVertex3f(-4.7, y+0.3, 0)
            glVertex3f(-5, y+0.3, 0)
            glEnd()
            
            glColor3f(1.0, 1.0, 1.0)  # White text
            self.draw_text(state, -4.5, y, 0)

    def draw_text(self, text, x, y, z):
        """Draw text in 3D space"""
        glRasterPos3f(x, y, z)
        for character in text:
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, ord(character))

    def keyboard(self, key, x, y):
        """Handle keyboard input"""
        if ord(key) == 27:  # ESC key
            glutLeaveMainLoop()

    def run(self):
        """Initialize and run the visualization"""
        # Initialize GLUT
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.window_width, self.window_height)
        glutCreateWindow(b"Patient Monitoring System Visualization")
        
        # Register callbacks
        glutDisplayFunc(self.display)
        glutKeyboardFunc(self.keyboard)
        
        # Initialize OpenGL
        self.init_gl()
        
        # Start the main loop
        print("OpenGL Visualization Running. Press ESC to exit.")
        glutMainLoop()

# For testing the visualization directly
if __name__ == "__main__":
    # Example state: [HR, BP, SpO2, Temp] - values 0 (normal), 1 (warning), 2 (critical)
    test_state = [0, 1, 2, 1]
    viz = PatientMonitoringVisualization(test_state)
    viz.run()