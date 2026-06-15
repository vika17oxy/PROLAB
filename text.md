PRO Lab: Probabilistic Robotics with ROS2 and TurtleBot4
In this project, you have to implement and evaluate three fundamental probabilistic state estimation
methods in mobile robotics:
• Kalman Filter (KF)
• Extended Kalman Filter (EKF)
• Particle Filter (PF)
All methods are implemented as ROS2 nodes and applied to a TurtleBot4 simulation to estimate the
robot state.
To complete the assignment, you have to
• implement KF, EKF and PF in a robotics context
• understand the role of motion and measurement models
• analyze the impact of uncertainty and noise
• evaluate estimation methods using quantitative metrics
• understand limitations of different filtering approaches
• design and analyze experiments in robotic systems
• solve the specific task you find in the given table
Core Tasks:
• Implement Filters: three separate ros2 nodes
• Subscribe to your sensors
• Publish an estimated pose
Use Common System Setup
• Same input data for all filters
• Same coordinate frame
• Same test trajectories
• Same evaluation conditions
Perform Mandatory Experiments
• Process Noise (Q) Variation
o test different Q values
o analyze model confidence
• Measurement Noise (R) Variation
o test different R values
o analyze sensor trust
• Runtime / Performance
o compare computational cost
o discuss real-time capability
• Ground Truth Evaluation
o compute error metrics (e.g. RMSE)
o compare trajectories
• Landmark detection
o Define a landmark by yourself
For the evaluation
• compare KF, EKF and PF
• visualize results (plots and RViz2)
• discuss strengths and weaknesses
Grading of the Project:
• Code Submission: 40%
• Presentation: 20%
• Documentation: 40%
Submission:
• One GitHub repo (with a README.md)
• PowerPoint presentation (on the last two sessions you have to show your results)
• Documentation (Paper Style)
time-delayed measurements
Introduce a delay between measurement and processing
use c++