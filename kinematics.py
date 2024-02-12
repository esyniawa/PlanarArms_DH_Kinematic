import os
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PlanarArms:
    # joint limits
    l_upper_arm_limit, u_upper_arm_limit = np.radians((-20, 160))  # in degrees [°]
    l_forearm_limit, u_forearm_limit = np.radians((0, 180))  # in degrees [°]

    # DH parameter
    scale = 1.0
    shoulder_length = scale * 50.0  # in [mm]
    upper_arm_length = scale * 220.0  # in [mm]
    forearm_length = scale * 160.0  # in [mm]

    def __init__(self,
                 init_angles_left: np.ndarray,
                 init_angles_right: np.ndarray,
                 radians: bool = False):

        """Constructor: initialize current joint angles, positions and trajectories"""
        self.angles_left = self.check_values(init_angles_left, radians)
        self.angles_right = self.check_values(init_angles_right, radians)

        self.trajectory_thetas_left = [self.angles_left]
        self.trajectory_thetas_right = [self.angles_right]

        self.trajectory_gradient_left = [np.array((0, 0))]
        self.trajectory_gradient_right = [np.array((0, 0))]

        self.end_effector_left = [PlanarArms.forward_kinematics(arm='left',
                                                                thetas=self.angles_left,
                                                                radians=True)[:, -1]]
        self.end_effector_right = [PlanarArms.forward_kinematics(arm='right',
                                                                 thetas=self.angles_right,
                                                                 radians=True)[:, -1]]


    @staticmethod
    def check_values(angles: np.ndarray, radians: bool):
        assert angles.size == 2, "Arm must contain two angles: angle shoulder, angle elbow"

        if not radians:
            angles = np.radians(angles)

        if angles[0] <= PlanarArms.l_upper_arm_limit or angles[0] >= PlanarArms.u_upper_arm_limit:
            raise AssertionError('Check joint limits for upper arm')
        elif angles[1] <= PlanarArms.l_forearm_limit or angles[1] >= PlanarArms.u_forearm_limit:
            raise AssertionError('Check joint limits for forearm')

        return angles

    @staticmethod
    def __circular_wrap(x: float, x_min: int | float, x_max: int | float):
        # Calculate the range of the interval
        interval_range = x_max - x_min

        # Calculate the wrapped value of x
        wrapped_x = x_min + ((x - x_min) % interval_range)

        return wrapped_x

    @staticmethod
    def circ_values(thetas: np.ndarray, radians: bool = True):
        """
        This wrapper function is intended to prevent phase jumps in the inverse kinematics due to large errors in the
        gradient calculation. This means that joint angles are only possible within the given limits.

        :param thetas:
        :param radians:
        :return:
        """
        if not radians:
            theta1, theta2 = np.radians(thetas)
        else:
            theta1, theta2 = thetas

        theta1 = PlanarArms.__circular_wrap(x=theta1,
                                            x_min=PlanarArms.l_upper_arm_limit,
                                            x_max=PlanarArms.u_upper_arm_limit)

        theta2 = PlanarArms.__circular_wrap(x=theta2,
                                            x_min=PlanarArms.l_forearm_limit,
                                            x_max=PlanarArms.u_forearm_limit)

        return np.array((theta1, theta2))

    @staticmethod
    def forward_kinematics(arm: str, thetas: np.ndarray, radians: bool = False):
        from functions import create_dh_matrix

        theta1, theta2 = PlanarArms.check_values(thetas, radians)

        if arm == 'right':
            const = 1
        elif arm == 'left':
            const = - 1
            theta1 = np.pi - theta1
            theta2 = - theta2
        else:
            raise ValueError('Please specify if the arm is right or left!')

        A0 = create_dh_matrix(a=const * PlanarArms.shoulder_length, d=0,
                              alpha=0, theta=0)

        A1 = create_dh_matrix(a=PlanarArms.upper_arm_length, d=0,
                              alpha=0, theta=theta1)

        A2 = create_dh_matrix(a=PlanarArms.forearm_length, d=0,
                              alpha=0, theta=theta2)

        # Shoulder -> elbow
        A01 = A0 @ A1
        # Elbow -> hand
        A12 = A01 @ A2

        return np.column_stack(([0, 0], A0[:2, 3], A01[:2, 3], A12[:2, 3]))

    @staticmethod
    def inverse_kinematics(arm: str,
                           end_effector: np.ndarray,
                           starting_angles: np.ndarray,
                           learning_rate: float = 0.01,
                           max_iterations: int = 5000,
                           abort_criteria: float = 1,  # in [mm]
                           radians: bool = False):

        from functions import create_jacobian

        if not radians:
            starting_angles = np.radians(starting_angles)

        thetas = starting_angles.copy()
        for i in range(max_iterations):
            # Compute the forward kinematics for the current joint angles
            current_position = PlanarArms.forward_kinematics(arm=arm,
                                                             thetas=thetas,
                                                             radians=True)[:, -1]

            # Calculate the error between the current end effector position and the desired end point
            error = end_effector - current_position

            # abort when error is smaller than the breaking condition
            if np.linalg.norm(error) < abort_criteria:
                break

            # Calculate the Jacobian matrix for the current joint angles
            J = create_jacobian(thetas=thetas, arm=arm,
                                a_sh=PlanarArms.upper_arm_length,
                                a_el=PlanarArms.forearm_length,
                                radians=True)

            delta_thetas = learning_rate * np.linalg.inv(J) @ error
            thetas += delta_thetas
            # prevent phase jumps due to large errors
            thetas = PlanarArms.circ_values(thetas, radians=True)

        return thetas

    @staticmethod
    def random_theta(return_radians=True):
        """
        Returns random joint angles within the limits.
        """
        theta1 = np.random.uniform(PlanarArms.l_upper_arm_limit, PlanarArms.u_upper_arm_limit)
        theta2 = np.random.uniform(PlanarArms.l_forearm_limit, PlanarArms.u_forearm_limit)

        if return_radians:
            return np.array((theta1, theta2))
        else:
            return np.degrees((theta1, theta2))

    @staticmethod
    def __cos_space(start: np.ndarray, stop: np.ndarray, num: int):
        """
        For the calculation of gradients and trajectories. Derivation of this function is sin(x),
        so that the maximal change in the middle of the trajectory.
        """
        if not start.size == stop.size:
            raise ValueError('Start and stop vector must have the same dimensions.')

        # calc changes
        offset = stop - start

        # function to modulate the movement.
        x_lim = np.repeat(np.pi, repeats=start.size)
        x = - np.cos(np.linspace(0, x_lim, num, endpoint=True)) + 1.0
        x /= np.amax(x)

        # linear space
        y = np.linspace(0, offset, num, endpoint=True)

        return start + x * y

    def reset_all(self):
        """Reset position to default and delete trajectories"""
        self.__init__(init_angles_left=self.trajectory_thetas_left[0],
                      init_angles_right=self.trajectory_thetas_right[0],
                      radians=True)

    def reset_arm_to_angle(self, arm: str, thetas: np.ndarray, radians: bool = True):
        """
        Set the joint angle of one arm to a new joint angle without movement. Thetas = must be in degrees
        """
        thetas = PlanarArms.check_values(thetas, radians=radians)
        if arm == 'right':
            self.__init__(init_angles_right=thetas,
                          init_angles_left=self.trajectory_thetas_left[-1],
                          radians=True)

        elif arm == 'left':
            self.__init__(init_angles_left=thetas,
                          init_angles_right=self.trajectory_thetas_right[-1],
                          radians=True)
        else:
            raise ValueError('Please specify if the arm is right or left!')

    def change_angle(self, arm: str, new_thetas: np.ndarray, num_iterations: int = 100, radians: bool = False):
        """
        Change the joint angle of one arm to a new joint angle.
        """
        new_thetas = self.check_values(new_thetas, radians=radians)
        if arm == 'right':

            trajectory = self.__cos_space(start=self.angles_right, stop=new_thetas, num=num_iterations)

            for delta_theta in trajectory:
                self.trajectory_gradient_right.append(delta_theta - self.trajectory_thetas_right[-1])
                self.trajectory_gradient_left.append(np.array((0, 0)))

                self.trajectory_thetas_right.append(delta_theta)
                self.trajectory_thetas_left.append(self.angles_left)

                self.end_effector_right.append(PlanarArms.forward_kinematics(arm='right',
                                                                             thetas=delta_theta,
                                                                             radians=True)[:, -1])
                self.end_effector_left.append(self.end_effector_left[-1])

            # set current angle to the new thetas
            self.angles_right = new_thetas

        elif arm == 'left':

            trajectory = self.__cos_space(start=self.angles_left, stop=new_thetas, num=num_iterations)[:, -1]

            for delta_theta in trajectory:
                self.trajectory_gradient_left.append(delta_theta - self.trajectory_thetas_left[-1])
                self.trajectory_gradient_right.append(np.array((0, 0)))

                self.trajectory_thetas_left.append(delta_theta)
                self.trajectory_thetas_right.append(self.angles_right)

                self.end_effector_left.append(PlanarArms.forward_kinematics(arm='left',
                                                                            thetas=delta_theta,
                                                                            radians=True)[:, -1])
                self.end_effector_right.append(self.end_effector_right[-1])

            # set current angle to the new thetas
            self.angles_left = new_thetas

        else:
            raise ValueError('Please specify if the arm is right or left!')

    def move_to_position(self, arm: str, end_effector: np.ndarray, num_iterations: int = 100):
        """
        Move to a certain coordinate within the peripersonal space.
        """
        if arm == 'right':
            new_thetas_to_position = self.inverse_kinematics(arm=arm, end_effector=end_effector,
                                                             starting_angles=self.angles_right, radians=True)

            self.change_angle(arm=arm, new_thetas=new_thetas_to_position, num_iterations=num_iterations, radians=True)

        elif arm == 'left':
            new_thetas_to_position = self.inverse_kinematics(arm=arm, end_effector=end_effector,
                                                             starting_angles=self.angles_left, radians=True)

            self.change_angle(arm=arm, new_thetas=new_thetas_to_position, num_iterations=num_iterations, radians=True)

    def touch_arm(self,
                  resting_arm: str,
                  joint_to_touch: int,
                  percentile: float,
                  num_iterations: int = 100):

        """
        Touch one point of the arm with the other arm. The variable joint_to_touch defines the modality:
        0 = shoulder; 1 = upper arm; 2 = forearm
        The variable percentile indicates the position relative to the modality.
        """

        if joint_to_touch not in [0, 1, 2]:
            raise ValueError('joint_to_touch must be either: 0 = shoulder; 1 = upper arm; 2 = forearm ')

        assert percentile >= 0, "percentile must be positive"
        assert percentile <= 1, "percentile must be lower than 1.0"

        if resting_arm == 'right':
            moving_arm = 'left'
            coordinates_rest_arm = self.forward_kinematics(arm=resting_arm, thetas=self.angles_right, radians=True)
        elif resting_arm == 'left':
            moving_arm = 'right'
            coordinates_rest_arm = self.forward_kinematics(arm=resting_arm, thetas=self.angles_left, radians=True)
        else:
            raise ValueError('Please specify if the arm is right or left!')

        # calculate coordinate to touch
        start = coordinates_rest_arm[:, joint_to_touch]
        end = coordinates_rest_arm[:, joint_to_touch + 1]

        touch_point = start + percentile * (end - start)

        # calculate movement trajectory to the touch point
        if moving_arm == 'right':
            self.move_to_position(arm=moving_arm,
                                  end_effector=touch_point,
                                  num_iterations=num_iterations)
        elif moving_arm == 'left':
            self.move_to_position(arm=moving_arm,
                                  end_effector=touch_point,
                                  num_iterations=num_iterations)

    def wait(self, time_steps: int):
        for t in range(time_steps):
            self.trajectory_thetas_right.append(self.angles_right)
            self.trajectory_thetas_left.append(self.angles_left)

            self.trajectory_gradient_right.append(np.array((0, 0)))
            self.trajectory_gradient_left.append(np.array((0, 0)))

            self.end_effector_right.append(self.end_effector_right[-1])
            self.end_effector_left.append(self.end_effector_left[-1])

    def training_trial(self,
                       arm: str,
                       position: np.ndarray,
                       t_min: int, t_max: int, t_wait: int = 10,
                       min_distance: float = 50.0,
                       trajectory_save_name: str = None):

        assert position.size == 2, "End effector should be 2-dimensional"

        distance = -1
        while distance <= min_distance:
            start_angles = PlanarArms.random_theta(return_radians=True)
            start_coordinate = PlanarArms.forward_kinematics(arm=arm,
                                                             thetas=start_angles,
                                                             radians=True)[:, -1]

            distance = np.linalg.norm(position - start_coordinate)

        time_interval = int(random.uniform(t_min, t_max))

        self.reset_arm_to_angle(arm=arm, thetas=start_angles, radians=True)
        self.move_to_position(arm=arm, end_effector=position, num_iterations=time_interval)
        self.wait(t_wait)
        if trajectory_save_name is not None:
            self.save_state(trajectory_save_name)

    def training_fixed_position(self,
                                arm: str,
                                init_angles: np.ndarray,
                                radians: bool,
                                position: np.ndarray,
                                t_min: int, t_max: int, t_wait: int = 10,
                                trajectory_save_name: str = None):

        assert position.size == 2, "End effector should be 2-dimensional"

        time_interval = int(random.uniform(t_min, t_max))

        self.reset_arm_to_angle(arm=arm, thetas=init_angles, radians=radians)
        self.move_to_position(arm=arm, end_effector=position, num_iterations=time_interval)
        self.wait(t_wait)
        if trajectory_save_name is not None:
            self.save_state(trajectory_save_name)

    def save_state(self, data_name: str = None):
        import datetime

        d = {
            'trajectory_left': self.trajectory_thetas_left,
            'gradients_left': self.trajectory_gradient_left,
            'end_effectors_left': self.end_effector_left,

            'trajectory_right': self.trajectory_thetas_right,
            'gradients_right': self.trajectory_gradient_right,
            'end_effectors_right': self.end_effector_right
        }

        df = pd.DataFrame(d)

        if data_name is not None:
            folder, _ = os.path.split(data_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)
        else:
            # get current date
            current_date = datetime.date.today()
            data_name = "PlanarArm_" + current_date.strftime('%Y%m%d')

        df.to_csv(data_name + '.csv', index=False)

    def import_state(self, file: str):
        df = pd.read_csv(file, sep=',')

        # convert type back to np.ndarray because pandas imports them as strings...
        regex_magic = lambda x: np.fromstring(x.replace('[', '').replace(']', ''), sep=' ', dtype=float)
        for column in df.columns:
            df[column] = df[column].apply(regex_magic)

        # set states
        self.angles_left = df['trajectory_left'].tolist()[-1]
        self.angles_right = df['trajectory_right'].tolist()[-1]

        self.trajectory_thetas_left = df['trajectory_left'].tolist()
        self.trajectory_thetas_right = df['trajectory_right'].tolist()

        self.trajectory_gradient_left = df['gradients_left'].tolist()
        self.trajectory_gradient_right = df['gradients_right'].tolist()

        self.end_effector_left = df['end_effectors_left'].tolist()
        self.end_effector_right = df['end_effectors_right'].tolist()


class VisPlanarArms(PlanarArms):

    x_limits = (-450, 450)
    y_limits = (-50, 400)

    def __init__(self,
                 init_angles_left: np.ndarray,
                 init_angles_right: np.ndarray,
                 radians: bool = False):

        super().__init__(
            init_angles_left,
            init_angles_right,
            radians
        )

    # Functions for visualisation
    def plot_current_position(self, plot_name=None, fig_size=(12, 8)):
        """
        Plots the current position of the arms.

        :param plot_name: Define the name of your figure. If none the plot is not saved!
        :param fig_size: Size of the Figure
        """
        coordinates_left = VisPlanarArms.forward_kinematics(arm='left', thetas=self.angles_left, radians=True)
        coordinates_right = VisPlanarArms.forward_kinematics(arm='right', thetas=self.angles_right, radians=True)

        fig, ax = plt.subplots(figsize=fig_size)

        ax.plot(coordinates_left[0, :], coordinates_left[1, :], 'b')
        ax.plot(coordinates_right[0, :], coordinates_right[1, :], 'b')

        ax.set_xlabel('x in [mm]')
        ax.set_ylabel('y in [mm]')

        ax.set_xlim(VisPlanarArms.x_limits)
        ax.set_ylim(VisPlanarArms.y_limits)

        # save
        if plot_name is not None:
            folder, _ = os.path.split(plot_name)
            if folder and not os.path.exists(folder):
                os.makedirs(folder)

            plt.savefig(plot_name)

        plt.show()

    def plot_trajectory(self, fig_size=(12, 8)):
        """
        Visualizes the movements performed so far. Use the slider to set the time

        :param fig_size:
        :return: None
        """
        from matplotlib.widgets import Slider

        init_t = 0
        num_t = len(self.trajectory_thetas_left)

        coordinates_left = []
        coordinates_right = []

        for i_traj in range(num_t):
            coordinates_left.append(VisPlanarArms.forward_kinematics(arm='left',
                                                                     thetas=self.trajectory_thetas_left[i_traj],
                                                                     radians=True))

            coordinates_right.append(VisPlanarArms.forward_kinematics(arm='right',
                                                                      thetas=self.trajectory_thetas_right[i_traj],
                                                                      radians=True))

        fig, ax = plt.subplots(figsize=fig_size)

        ax.set_xlabel('x in [mm]')
        ax.set_ylabel('y in [mm]')

        ax.set_xlim(VisPlanarArms.x_limits)
        ax.set_ylim(VisPlanarArms.y_limits)

        l, = ax.plot(coordinates_left[init_t][0, :], coordinates_left[init_t][1, :], 'b')
        r, = ax.plot(coordinates_right[init_t][0, :], coordinates_right[init_t][1, :], 'b')

        ax_slider = plt.axes((0.25, 0.05, 0.5, 0.03))
        time_slider = Slider(
            ax=ax_slider,
            label='n iteration',
            valmin=0,
            valmax=num_t - 1,
            valinit=0,
        )

        def update(val):
            t = int(time_slider.val)
            l.set_data(coordinates_left[t][0, :], coordinates_left[t][1, :])
            r.set_data(coordinates_right[t][0, :], coordinates_right[t][1, :])
            time_slider.valtext.set_text(t)

        time_slider.on_changed(update)

        plt.show()
