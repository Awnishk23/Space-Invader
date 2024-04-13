import numpy as np


class FabrikRobotArm:
    def __init__(self, link_lengths):
        self.link_lengths = link_lengths
        self.num_joints = len(link_lengths)
        self.max_iterations = 1000
        self.tolerance = 1e-5

    def forward_kinematics(self, angles):
        x = np.zeros(self.num_joints + 1)
        y = np.zeros(self.num_joints + 1)
        z = np.zeros(self.num_joints + 1)
        x[0] = 0
        y[0] = 0
        z[0] = 0
        for i in range(1, self.num_joints + 1):
            x[i] = x[i - 1] + self.link_lengths[i - 1] * np.cos(np.radians(angles[i - 1]))
            y[i] = y[i - 1] + self.link_lengths[i - 1] * np.sin(np.radians(angles[i - 1]))
            z[i] = 0
        return x, y, z

    def distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2)

    def fabrik_algorithm(self, initial_angles, target_position):
        current_angles = initial_angles.copy()
        target_distance = self.distance(target_position, [0, 0, 0])
        for iteration in range(self.max_iterations):
            # Forward reaching
            x, y, z = self.forward_kinematics(current_angles)
            end_effector_distance = self.distance([x[-1], y[-1], z[-1]], target_position)
            if end_effector_distance < self.tolerance:
                print(f"Converged in {iteration + 1} iterations.")
                return current_angles

            # Backward reaching
            x[-1], y[-1], z[-1] = target_position
            for i in range(self.num_joints - 1, 0, -1):
                dist_ratio = self.link_lengths[i] / self.distance([x[i], y[i], z[i]], [x[i + 1], y[i + 1], z[i + 1]])
                x[i] = x[i + 1] + dist_ratio * (x[i] - x[i + 1])
                y[i] = y[i + 1] + dist_ratio * (y[i] - y[i + 1])
                z[i] = z[i + 1] + dist_ratio * (z[i] - z[i + 1])

            # Forward reaching again
            for i in range(1, self.num_joints + 1):
                dist_ratio = self.link_lengths[i - 1] / self.distance([x[i - 1], y[i - 1], z[i - 1]],
                                                                      [x[i], y[i], z[i]])
                x[i] = x[i - 1] + dist_ratio * (x[i] - x[i - 1])
                y[i] = y[i - 1] + dist_ratio * (y[i] - y[i - 1])
                z[i] = z[i - 1] + dist_ratio * (z[i] - z[i - 1])

            # Calculate new joint angles
            for i in range(self.num_joints):
                current_angles[i] = np.degrees(np.arctan2(y[i + 1] - y[i], x[i + 1] - x[i]))

            print(f"Iteration {iteration + 1}: {current_angles}")

        print("Did not converge.")
        return None

    def check_reachability(self, initial_angles, target_position):
        x, y, z = self.forward_kinematics(initial_angles)
        end_effector_position = [x[-1], y[-1], z[-1]]
        end_effector_distance = self.distance(end_effector_position, target_position)
        return end_effector_distance <= sum(self.link_lengths)


if __name__ == "__main__":
    link_lengths = [23, 15, 7]  # As my roll number is 230250
    robot_arm = FabrikRobotArm(link_lengths)

    print("Moving Robotic Arm to Target Position")

    # Getting Input initial joint angles
    initial_angles = []
    for i in range(robot_arm.num_joints):
        angle = float(input(f"Enter initial angle for joint {i + 1}: "))
        initial_angles.append(angle)

    # Input target position
    target_x = float(input("Enter target x-coordinate: "))
    target_y = float(input("Enter target y-coordinate: "))
    target_z = float(input("Enter target z-coordinate: "))
    target_position = [target_x, target_y, target_z]

    print("Initial Joint Angles:", initial_angles)
    print("Target Position:", target_position)

    if robot_arm.check_reachability(initial_angles, target_position):
        print("TARGET POSITION REACHABLE")
        optimal_angles = robot_arm.fabrik_algorithm(initial_angles, target_position)
        if optimal_angles:
            print("PART-C: Optimal Joint Angles Found:", optimal_angles)
    else:
        print("PART-B: Target Position is not reachable by the Robotic Arm.")
