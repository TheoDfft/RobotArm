import numpy as np
from math import atan2, sqrt, cos, sin, pi, atan, acos, degrees, radians
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg') 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, TextBox, Button
import roboticstoolbox as rtb
import sys



class RobotArm():
    """
    A class representing a 6-DOF robotic arm with forward/inverse kinematics and visualization.
    The robot is defined using DH (Denavit-Hartenberg) parameters and includes joint limits.
    """
    
    def __init__(self):
        """
        Initialize the robot arm with DH parameters, joint limits, and necessary state variables.
        
        DH parameters format: [theta, d, a, alpha]
        - theta: rotation about z-axis
        - d: translation along z-axis
        - a: translation along x-axis
        - alpha: rotation about x-axis
        """
        
        # Define DH parameters for each joint
        self.dh_params = [
            [0, 0.15, 0, pi/2],    # Joint 1: Base rotation
            [0, 0, 0.25, 0],       # Joint 2: Shoulder
            [pi/2, 0, 0, pi/2],    # Joint 3: Elbow
            [pi, 0.15, 0, pi/2],   # Joint 4: Wrist 1
            [-pi/2, 0.0, 0, pi/2],  # Joint 5: Wrist 2
            [0, 0.3, 0, 0]        # Joint 6: Wrist 3
        ]

        # Define joint angle limits in degrees
        self.joint_limits = [
            [-180, 180],  # Joint 1 limits
            [-30, 210],   # Joint 2 limits
            [-160, 160],  # Joint 3 limits
            [-360, 360],  # Joint 4 limits
            [-360, 360],  # Joint 5 limits
            [-360, 360],  # Joint 6 limits
        ]
        
        self.joint = [f'Joint{i+1}' for i in range(len(self.dh_params))]
        self.theta = [dh_param[0] for dh_param in self.dh_params]
        self.d = [dh_param[1] for dh_param in self.dh_params]
        self.a = [dh_param[2] for dh_param in self.dh_params]
        self.alpha = [dh_param[3] for dh_param in self.dh_params]
        self.joint_variables = [0 for _ in range(len(self.dh_params))]
        self.joint_positions = [np.zeros(3) for _ in range(len(self.dh_params))]
        self.joint_orientations = [np.zeros(3) for _ in range(len(self.dh_params))]
        self.joint_states = [np.zeros(6) for _ in range(len(self.dh_params))]
        self.joint_rotation_matrices = [np.zeros((3,3)) for _ in range(len(self.dh_params))]
        self.end_effector_rotation_matrix = np.zeros((3,3))
        self.end_effector_position = np.zeros(3)
        self.end_effector_orientation = np.zeros(3)
        self.end_effector_state = np.zeros(6)
        self.T_0_6 = np.zeros((4,4))
        self.sliders = []
        self.bounding_box = []
        self.info_text_annotation = None

        # Create robot using Robotics Toolbox
        self.links = []
        for theta, d, a, alpha in self.dh_params:
            link = rtb.RevoluteDH(
                d=d,
                a=a,
                alpha=alpha,
                offset=theta
            )
            self.links.append(link)

        self.robot = rtb.DHRobot(self.links, name="6DoF Robot")
        
        self.workspace_plots = []  
    

    def homogeneous_transform_matrix(self, theta, d, a, alpha, joint_angle=0):
        """
        Calculate the homogeneous transformation matrix using DH parameters.
        
        Args:
            theta: Joint angle rotation about z-axis
            d: Link offset along z-axis
            a: Link length along x-axis
            alpha: Link twist angle about x-axis
            joint_angle: Additional joint angle for variable joints
            
        Returns:
            4x4 homogeneous transformation matrix
        """
        T = np.array([[cos(theta + joint_angle), -sin(theta + joint_angle)*cos(alpha), sin(theta + joint_angle)*sin(alpha), a*cos(theta + joint_angle)],
                      [sin(theta + joint_angle), cos(theta + joint_angle)*cos(alpha), -cos(theta + joint_angle)*sin(alpha), a*sin(theta + joint_angle)],
                      [0, sin(alpha), cos(alpha), d],
                      [0, 0, 0, 1]])
        return T
    
    def inverse_kinematics(self, end_effector_state):
        """
        Calculate joint angles for a desired end-effector pose using a geometric approach with decoupling of the wrist and arm.
        
        Args:
            end_effector_state: Array [x, y, z, alpha, beta, gamma] representing desired position and orientation
            
        Returns:
            Array of 6 joint angles in radians, or None if no solution exists
        """

        #Full position and orientation of the end effector + current joint variables for optimization
        px = end_effector_state[0]
        py = end_effector_state[1]
        pz = end_effector_state[2]

        #Compute the rotation matrix from the desired end effector state
        R_desired = self.fixed_frame_transformation_matrix(end_effector_state)[:3, :3]

        #Find the wrist position with respect to the desired end effector position
        xw = px - self.d[5]*R_desired[0,2]
        yw = py - self.d[5]*R_desired[1,2]
        zw = pz - self.d[5]*R_desired[2,2]

        #Find joint angle 1
        q1 = [np.arctan2(yw, xw), np.arctan2(yw, xw)+np.pi]
        q1_optimal = self.find_optimal_angle(q1, 0, self.joint_limits[0])
        if q1_optimal is None:
            q1_optimal = self.joint_variables[0]
            q1_k = 0
        else:
            q1_k = q1.index(q1_optimal)

        #Calculate joint angles 2-3
        r = sqrt(xw**2 + yw**2)     #Px of Wrist with respect to frame 1
        s = (zw - self.d[0])        #Pz of Wrist with respect to frame 1
        a2 = self.a[1]              #Length of link 2
        a3 = self.d[3]              #Length of link 3

        c3 = (r**2 + s**2 - a2**2 - a3**2)/(2*a2*a3)

        if abs(c3) > 1:
            print(f'The target point is out of the dexterity workspace.')
            return None
        elif c3 == 1:
            if q1_k == 0:
                q2 = [np.arctan2(s,r)]
                q3 = [0]
            else:
                q2 = [np.arctan2(s,-r)]
                q3 = [0]
        elif c3 == -1 and (r != 0 or s != 0):
            if q1_k == 0:
                q2 = [np.arctan2(s,r)]
                q3 = [np.pi]
            else:
                q2 = [np.arctan2(s,-r)]
                q3 = [np.pi]
        elif c3 == -1 and r == 0 and s == 0:
            q2 = self.joint_variables[1]
            q3 = np.pi
        else:
            if q1_k == 0:
                q3 = [np.arccos(c3), -np.arccos(c3)]
                theta_temp = np.arctan2(s,r)
                q2 = [theta_temp - np.arctan2(a3*np.sin(q3[0]), a2+a3*np.cos(q3[0])), theta_temp - np.arctan2(a3*np.sin(q3[1]), a2+a3*np.cos(q3[1]))]
            else:   
                q3 = [np.arccos(c3), -np.arccos(c3)]
                theta_temp = np.arctan2(s,-r)
                q2 = [theta_temp - np.arctan2(a3*np.sin(q3[0]), a2+a3*np.cos(q3[0])), theta_temp - np.arctan2(a3*np.sin(q3[1]), a2+a3*np.cos(q3[1]))]

        q2_optimal = self.find_optimal_angle(q2, 1, self.joint_limits[1])
        q2_i = q2.index(q2_optimal)
        q3_optimal = q3[q2_i]

        first_three_joint_variables = [q1_optimal, q2_optimal, q3_optimal]
        
        #Find joint angles 4-6:
        #Calculate the transformation matrix of the end effector with respect to the base after positioning the first three joints
        T0_6_theta456_0 = np.eye(4)
        for i in range(6):
            if i < 3:
                T0_6_theta456_0 = np.matmul(T0_6_theta456_0, self.homogeneous_transform_matrix(self.theta[i], self.d[i], self.a[i], self.alpha[i], first_three_joint_variables[i]))
            else:
                T0_6_theta456_0 = np.matmul(T0_6_theta456_0, self.homogeneous_transform_matrix(self.theta[i], self.d[i], self.a[i], self.alpha[i]))

        R0_6_theta456_0 = T0_6_theta456_0[:3, :3]    
        
        #Find the desired Euler rotation matrix for joints 4, 5, 6
        R_D_bar = np.matmul(np.linalg.inv(R0_6_theta456_0), R_desired)

        #From the resulting Euler rotation matrix, find all the possible solutions for joints 4, 5, 6
        if np.arcsin(R_D_bar[0,2]) == pi/2:
            q4_optimal = 0
            q6_optimal = np.arctan2(R_D_bar[2,1], R_D_bar[1,1])
        elif np.arcsin(R_D_bar[0,2]) == -pi/2:
            q4_optimal = 0
            q6_optimal = np.arctan2(R_D_bar[1,0], R_D_bar[2,1])
        else:
            q5 = [np.arctan2(R_D_bar[0,2], np.sqrt(R_D_bar[0,0]**2 + R_D_bar[0,1]**2)), np.arctan2(R_D_bar[0,2], -np.sqrt(R_D_bar[0,0]**2 + R_D_bar[0,1]**2))]
            q5_optimal = self.find_optimal_angle(q5, 4, self.joint_limits[4])
            
            q4_optimal = np.arctan2(-R_D_bar[1,2]/np.cos(q5_optimal), R_D_bar[2,2]/np.cos(q5_optimal))
            q6_optimal = np.arctan2(-R_D_bar[0,1]/np.cos(q5_optimal), R_D_bar[0,0]/np.cos(q5_optimal))
        
        return np.array([q1_optimal, q2_optimal, q3_optimal, q4_optimal, q5_optimal, q6_optimal])
    
    def find_optimal_angle(self, possible_q, joint_index, joint_limits):
        """
        Helper function to find the smallest displacement between the current joint angle and a list of possible angles.

        Args:
            possible_q: List of possible joint angles
            joint_index: Index of the joint to fetch the current joint angles
            joint_limits: Limits of the joint
            
        Returns:
            Optimal joint angle
        """
        current_joint_angle = self.joint_variables[joint_index]
        if len(possible_q) == 1:
            return possible_q[0]
        else:
            for q in possible_q:
                if q <= radians(joint_limits[0]) and q >= radians(joint_limits[1]):
                    possible_q.remove(q)
                else:
                    closest_q = min(possible_q, key=lambda x: abs(x - current_joint_angle))
        
        if closest_q:
            return closest_q
        else:
            print(f'No appropriate solution found for {self.joint[joint_index]}')
            return None

    def frame_orientation(self, rotation_matrix):
        """
        Calculate the fixed frame angles (alpha, beta, gamma) from a rotation matrix.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Array of 3 Euler angles in radians
        """
        r11 = rotation_matrix[0,0]
        r12 = rotation_matrix[0,1]
        r13 = rotation_matrix[0,2]
        r21 = rotation_matrix[1,0]
        r22 = rotation_matrix[1,1]
        r23 = rotation_matrix[1,2]
        r31 = rotation_matrix[2,0]
        r32 = rotation_matrix[2,1]
        r33 = rotation_matrix[2,2]

        beta = atan2(-r31, sqrt(r11**2 + r21**2))
        if cos(beta) != 0:
            alpha = atan2(r21/cos(beta), r11/cos(beta))
            gamma = atan2(r32/cos(beta), r33/cos(beta))
        elif beta == pi/2:
            alpha = 0
            gamma = atan2(r12,r22)
        elif beta == -pi/2:
            alpha = 0
            gamma = -atan2(r12,r22)

        return np.array([alpha, beta, gamma])
    
    def fixed_frame_transformation_matrix(self, desired_state):
        """
        Generates a transformation matrix given a desired translation and rotation
        
        Args:
            desired_state: Array [x, y, z, alpha, beta, gamma] representing desired position and orientation
            
        Returns:
            4x4 transformation matrix
        """
        px = desired_state[0]
        py = desired_state[1]
        pz = desired_state[2]
        alpha = desired_state[3]
        beta = desired_state[4]
        gamma = desired_state[5]

        T = np.array([[cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma), cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma), px],
                      [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma), py],
                      [-sin(beta), cos(beta)*sin(gamma), cos(beta)*cos(gamma), pz],
                      [0, 0, 0, 1]])
        return T

    def forward_kinematics(self, joint_variables=None, print=False):
        """
        Calculate end-effector pose given joint angles using forward kinematics.
        
        Args:
            joint_variables: Array of 6 joint angles in radians
            print: Boolean to control debug output
            
        Returns:
            End-effector state [x, y, z, alpha, beta, gamma]
        """
        #For each joint, calculate the transformation matrix and update the joint states
        #Multiply right each transformation matrix to get the transformation matrix of the end effector
        self.T_0_6 = np.eye(4)
        for i, dh_param in enumerate(self.dh_params):
            local_T = self.homogeneous_transform_matrix(self.theta[i]+joint_variables[i], self.d[i], self.a[i], self.alpha[i])
            self.T_0_6 = np.matmul(self.T_0_6, local_T)
            self.joint_positions[i] = self.T_0_6[:3, 3]
            self.joint_orientations[i] = self.frame_orientation(self.T_0_6[:3, :3])
            self.joint_rotation_matrices[i] = self.T_0_6[:3, :3]
            self.joint_states[i] = np.concatenate([self.joint_positions[i], self.joint_orientations[i]])

        self.end_effector_rotation_matrix = self.T_0_6[:3, :3]
        self.end_effector_position = self.joint_positions[-1]
        self.end_effector_orientation = self.joint_orientations[-1]
        self.end_effector_state = self.joint_states[-1]
        
        #Verbose if needed to use as a standalone method
        if print:
            print(f'Joint Variables: {joint_variables}\n'
                  f'End Effector Position: ({self.end_effector_position[0]:.2f}, {self.end_effector_position[1]:.2f}, {self.end_effector_position[2]:.2f}) mm\n'
                  f'End Effector Orientation: ({degrees(self.end_effector_orientation[0]):.2f}, {degrees(self.end_effector_orientation[1]):.2f}, {degrees(self.end_effector_orientation[2]):.2f}) degrees')
            print("-" * 50)
        
        return self.end_effector_state


    def setup_visualization(self):
        """
        Set up the 3D visualization environment with interactive controls.
        Creates sliders and textboxes for joint control and buttons for workspace visualization and reset of the robot/plot.
        """
        self.env = rtb.backends.PyPlot.PyPlot()      # Initialize the PyPlot backend
        self.env.launch()
        # plt.get_current_fig_manager().resize(1920, 1080)                

        # Adjust the main axes to take up only 75% of the vertical space
        self.env.ax.set_position([0.1, 0.2, 0.8, 0.8])  # [left, bottom, width, height]

        # Create wireframe cube
        r = [-0.5, 0.5]
        x, y = np.meshgrid(r, r)
        z = np.array([[0.7, 0.7], [0.7, 0.7]])
        self.env.ax.plot_wireframe(x, y, z, color='k', alpha=0)
        
        # Draw a base for the robot
        base_vertices = np.array([
            [-0.07, -0.07, 0],
            [-0.07, 0.07, 0], 
            [0.07, 0.07, 0],
            [0.07, -0.07, 0],
            [-0.07, -0.07, -0.3],
            [-0.07, 0.07, -0.3],
            [0.07, 0.07, -0.3], 
            [0.07, -0.07, -0.3]
        ])
        
        # Define triangular faces
        faces = [
            [0,1,2], [0,2,3],  # Top
            [4,5,6], [4,6,7],  # Bottom
            [0,1,5], [0,5,4],  # Front
            [2,3,7], [2,7,6],  # Back
            [0,3,7], [0,7,4],  # Left
            [1,2,6], [1,6,5]   # Right
        ]
        
        self.env.ax.plot_trisurf(base_vertices[:,0], base_vertices[:,1], base_vertices[:,2], 
                                triangles=faces, color='gray', shade=True, alpha=0.8)

        # Add robot to environment
        self.env.add(self.robot,                     # Robot model
                jointaxes=False,                # Show joint axes
                eeframe=True,                  # Show end-effector frame
                shadow=True,                   # Show ground shadow
                display=True
                )
        
        self.ax = self.env.ax
        self.ee_annotation = self.ax.figure.text(
        0.5,  # x position (centered)
        0.95,  # y position (top)
        '',    # initial empty text
        transform=self.ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(
            boxstyle='round,pad=0.5',
            fc='white',
            alpha=0.8,
            edgecolor='red')
        )
        
        # Set labels and title
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('6 DoF Robot Arm')

        slider_axes = [plt.axes([0.25, 0.20 - i*0.03, 0.4, 0.03]) 
                      for i in range(len(self.links))]
        textbox_axes = [plt.axes([0.1, 0.20 - i*0.03, 0.03, 0.03]) 
                         for i in range(len(self.links))]
        button_axes = [plt.axes([0.14, 0.20 - i*0.03, 0.075, 0.03])
                      for i in range(len(self.links))]
        
        #Setup workspace button
        workspace_button_ax = plt.axes([0.7, 0.85, 0.2, 0.04])  # [left, bottom, width, height]
        self.workspace_button = Button(workspace_button_ax, 'Draw Workspace', color='lightgray', hovercolor='0.7')
        self.workspace_button.on_clicked(lambda event: self.plot_workspace())

        #Setup reset button
        reset_button_ax = plt.axes([0.7, 0.80, 0.2, 0.04])  # [left, bottom, width, height]
        self.reset_button = Button(reset_button_ax, 'Reset', color='lightgray', hovercolor='0.9')
        self.reset_button.on_clicked(lambda event: self.reset_robot())
        
        #Setup sliders and textboxes
        self.sliders = []
        self.textboxes = []
        self.buttons = []
        
        for i, (slider_ax, textbox_ax, button_ax) in enumerate(zip(slider_axes, textbox_axes, button_axes)):
            slider = Slider(slider_ax, f'', self.joint_limits[i][0], self.joint_limits[i][1], valinit=0)
            textbox = TextBox(textbox_ax, f'Joint {i+1}:', initial='', color='lightgray')
            button = Button(button_ax, 'Update', color='lightgray', hovercolor='0.9')
            
            self.sliders.append(slider)
            self.textboxes.append(textbox)
            self.buttons.append(button)
            
            slider.on_changed(lambda value: self.update_plot())
            textbox.on_text_change(lambda value, idx=i: self.update_textbox_value(value, idx))
            button.on_clicked(lambda event, idx=i: self.update_from_textbox(idx))

        plt.get_current_fig_manager().full_screen_toggle()

        

    def update_textbox_value(self, value, index):
        """
        Updates the textbox value and joint variables based on the input value.
        """
        try:
            angle = float(value)
            if self.joint_limits[index][0] <= angle <= self.joint_limits[index][1]:
                self.joint_variables[index] = angle
                # self.textboxes[index].begin_typing(value)
                # self.textboxes[index].stop_typing(value)
                self.textboxes[index].set_val(value)
            else:
                print(f"Invalid input for Joint {index+1}. Please enter an angle between -180 and 180 degrees.")
        except ValueError:
            print(f"Invalid input for Joint {index+1}. Please enter an angle in degrees.")

    def update_from_textbox(self, index):
        angle = float(self.textboxes[index].text)
        self.sliders[index].set_val(angle)
        self.update_plot()

    def set_joint_variables(self, joint_variables):
        self.joint_variables = joint_variables
        for i in range(len(self.sliders)):
            self.sliders[i].set_val(degrees(joint_variables[i]))
        self.update_plot()

    def update_plot(self):
        self.joint_variables = [radians(slider.val) for slider in self.sliders]
        self.forward_kinematics(self.joint_variables)
        self.plot_robot_arm()
        return

    def plot_robot_arm(self):
        # Create backend if not exists
        if not hasattr(self, 'env'):
            self.env = rtb.backends.PyPlot()
            self.env.launch()
        
        # Update robot configuration
        self.robot.q = self.joint_variables
        
        ee_text = f'End Effector:\n Position: ({self.end_effector_position[0]:.2f}, {self.end_effector_position[1]:.2f}, {self.end_effector_position[2]:.2f}) mm\n' \
                    f'Orientation: ({degrees(self.end_effector_orientation[0]):.2f}, {degrees(self.end_effector_orientation[1]):.2f}, {degrees(self.end_effector_orientation[2]):.2f}) degrees'
        self.ee_annotation.set_text(ee_text)

        # Display robot
        self.env.step()
        return

    def reset_robot(self):
        # Remove all workspace plots
        for plot in self.workspace_plots:
            plot.remove()
        self.workspace_plots = []

        self.set_joint_variables([0, 0, 0, 0, 0, 0])
        return
   

    def plot_surface(self, joint_limits, resolution=0.5, color='green'):
        points = []

        # Define step sizes for each joint
        steps = [resolution, resolution, resolution, resolution]  # Adjust these values to control density
        
        # Generate joint configurations
        def create_range(start, end, step):
            if start == end:
                return np.array([radians(start)])
            step = abs(step) * (-1 if end < start else 1)
            return np.arange(radians(start), radians(end+step/2), radians(step))  # Added step/2 to include end point

        q1_range = create_range(joint_limits[0][0], joint_limits[0][1], steps[0])
        q2_range = create_range(joint_limits[1][0], joint_limits[1][1], steps[1])
        q3_range = create_range(joint_limits[2][0], joint_limits[2][1], steps[2])
        q5_range = create_range(joint_limits[3][0], joint_limits[3][1], steps[3])
        
        total_points = len(q1_range) * len(q2_range) * len(q3_range) * len(q5_range)
        current_point = 0
        for i in q1_range:
            for j in q2_range:
                for k in q3_range:  
                    for l in q5_range:
                        new_joint_variables = [i, j, k, 0, l, 0]
                        self.forward_kinematics(new_joint_variables)
                        points.append(self.end_effector_position)  # Convert numpy array to list
                        current_point += 1
                        print(f"{current_point}/{total_points}", end='\r')
                        # self.set_joint_variables(new_joint_variables)
            
        
        # Convert points to numpy array and reshape
        points = np.array(points)
        
        # # #Plot the surface mesh
        # self.env.ax.plot_trisurf(
        #     points[:, 0],
        #     points[:, 1],
        #     points[:, 2],
        #     alpha=0.2,
        #     color=color,
        #     shade=True,
        #     antialiased=True
        # )
        
        # # Also plot the points
        scatter = self.env.ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            alpha=0.5,
            c=color,
            marker='.',
            s=0.5
        )
        self.workspace_plots.append(scatter)  # Store the reference

    def plot_workspace(self):
        """
        Visualize the robot's workspace by sampling different joint configurations.
        Creates a point cloud representation of reachable positions.
        """
        # Sample different configurations to show workspace boundaries
        self.plot_surface([[-180, 180], [0, 90], [0, 0], [90, 90]], color='green')
        self.plot_surface([[-180, 180], [-20, 90], [0, 0], [90, 90]], color='green')
        self.plot_surface([[-180, 180], [-20, -20], [-75, 0], [90, 90]], color='green')
        self.plot_surface([[-180, 180], [-20, -20], [-75, -75], [0, 90]], color='green')

if __name__ == '__main__':
    """
    Main execution block that creates a robot instance and runs the interactive visualization.
    Allows users to input desired end-effector states for inverse kinematics testing.
    """
    robot = RobotArm()
    robot.setup_visualization()
    robot.plot_robot_arm()
    
    try:
        while True:
            try:
                #Prompt the user to input a desired end effector state
                print("\n----------------------- Inverse Kinematic Solver -----------------------")
                desired_state = input("Enter the desired end effector state (meters/degrees values separated by spaces: 'x y z alpha beta gamma'):\n")
                desired_state = [float(x) for x in desired_state.split(' ')]
                desired_state[3:] = [radians(x) for x in desired_state[3:]]
                joint_variables = robot.inverse_kinematics(desired_state)
                robot.set_joint_variables(joint_variables)
            except Exception as e:
                print(f"Invalid input for end effector state. Error: {e}")
        
    except KeyboardInterrupt:
        plt.close('all')
        sys.exit(0)
