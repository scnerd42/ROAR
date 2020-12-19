import cv2 as cv
import numpy as np
from numpy.matlib import repmat
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl

from ROAR.perception_module.legacy.ground_plane_point_cloud_detector import GroundPlanePointCloudDetector
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.control_module.pure_pursuit_control import PurePursuitController
from pathlib import Path
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.visualization_module.visualizer import Visualizer
from ROAR.utilities_module.data_structures_models import Transform
from ROAR.utilities_module.camera_models import Camera
from typing import Union
import math



# import matplotlib.pyplot as plt
class LaneTrackerAgent(Agent):
    def __init__(self, **kwargs):
        super(LaneTrackerAgent, self).__init__(**kwargs)
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.controller = PurePursuitController(agent=self, target_speed=100)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = SimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1)
        """
        self.gp_pointcloud_detector = GroundPlanePointCloudDetector(agent=self,
                                                                    max_points_to_convert=10000,
                                                                    nb_neighbors=100,
                                                                    std_ratio=1)
        """
        self.gp_pointcloud_detector = GroundPlanePointCloudDetector(agent=self,
                                                                    max_points_to_convert=10000,
                                                                    nb_neighbors=100,
                                                                    std_ratio=1)

        self.occupancy_grid_map = OccupancyGridMap(absolute_maximum_map_size=800)
        self.visualizer = Visualizer(agent=self)
        #self.generated_way_point = np.array()

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(LaneTrackerAgent, self).run_step(sensors_data, vehicle)
        try:
            vehicle_transform: Union[Transform, None] = self.vehicle.transform
            depth_arr = None if self.front_depth_camera.data is None else self.front_depth_camera.data.copy()
            #cv.imshow("depth", depth_arr)
            image = self.front_rgb_camera.data.copy()
            frame = image
            cv.imshow("original", frame)
            canny = do_canny(frame)
            #cv.imshow("canny", canny)
            # plt.imshow(frame)
            # plt.show()
            segment = do_segment(canny)
            #cv.imshow("segment", segment)
            hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength=10, maxLineGap=10)

            # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
            lines = calculate_lines(frame, hough)
            #middle_line = calc_middle(lines)
            # Visualizes the lines
            lines_visualize = visualize_lines(frame, lines)
            #cv.imshow("hough", lines_visualize)
            # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
            output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
            # Opens a new window and displays the output frame
            cv.imshow("output", output)
            # Generating waypoint from image



        except Exception as e:
            self.logger.error(f"Point cloud RunStep Error: {e}")
        finally:

            return self.local_planner.run_in_series()

def do_canny(frame):
    # Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # Applies a 5x5 gaussian blur with deviation of 0 to frame - not mandatory since Canny will do this for us
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    # Applies Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv.Canny(blur, 50, 150)
    return canny


def do_segment(frame):
    # Since an image is a multi-directional array containing the relative intensities of each pixel in the image, we can use frame.shape to return a tuple: [number of rows, number of columns, number of channels] of the dimensions of the frame
    # frame.shape[0] give us the number of rows of pixels the frame has. Since height begins from 0 at the top, the y-coordinate of the bottom of the frame is its height
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by three (x, y) coordinates
    polygons = np.array([
        [(0, height), (800, height), (380, 290)]
    ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv.bitwise_and(frame, mask)
    return segment


def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])


def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = int(y1 - 200)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

def visualize_lines2(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is not None:
        for xm, ym in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            cv.line(lines_visualize, (xm, ym), (0, 255, 0), 5)
    return lines_visualize

def get_global_position(self):
        """
        Compute the location of the car in the global frame
        :return: a array of (x, y, z, roll, pitch, yaw)
        """
        if self.prev_config is None:
            return None

        roll, pitch, yaw = cv.decomposeProjectionMatrix(self.prev_config[:3])[-1].reshape((3,))
        x, y, z = self.prev_config[:3, 3]
        return np.array([x, y, z, roll, pitch, yaw])

# def get_waypoint(self,lines_visualize):
#     # create xyz point with orientation for the vehicle to travel to
#     vehicle_transform: Union[Transform, None] = self.agent.vehicle.transform
#     heading = vehicle_transform - lane_line
#     way_point = image2world(lines_visualize) + heading
#     return way_point


def depth_to_local_point_cloud(image, color=None, max_depth=0.9):
    """
    Convert an image containing CARLA encoded depth-map to a 2D array containing
    the 3D position (relative to the camera) of each pixel and its corresponding
    RGB color of an array.
    "max_depth" is used to omit the points that are far enough.
    """
    far = 1000.0  # max depth in meters.
    normalized_depth = image
    width = 800
    height = 600

    # (Intrinsic) K Matrix
    # k = Camera.calculate_default_intrinsics_matrix()
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / \
        (2.0 * math.tan(70 * math.pi / 360.0))

    # 2d pixel coordinates
    pixel_length = width * height
    u_coord = repmat(np.r_[width-1:-1:-1],
                     height, 1).reshape(pixel_length)
    v_coord = repmat(np.c_[height-1:-1:-1],
                     1, width).reshape(pixel_length)
    if color is not None:
        color = color.reshape(pixel_length, 3)
    normalized_depth = np.reshape(normalized_depth, pixel_length)

    # Search for pixels where the depth is greater than max_depth to
    # delete them
    max_depth_indexes = np.where(normalized_depth > max_depth)
    normalized_depth = np.delete(normalized_depth, max_depth_indexes)
    u_coord = np.delete(u_coord, max_depth_indexes)
    v_coord = np.delete(v_coord, max_depth_indexes)
    if color is not None:
        color = np.delete(color, max_depth_indexes, axis=0)

    # pd2 = [u,v,1]
    p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])

    # P = [X,Y,Z]
    p3d = np.dot(np.linalg.inv(k), p2d)
    p3d *= normalized_depth * far

    # Formating the output to:
    # [[X1,Y1,Z1,R1,G1,B1],[X2,Y2,Z2,R2,G2,B2], ... [Xn,Yn,Zn,Rn,Gn,Bn]]
    if color is not None:
        # np.concatenate((np.transpose(p3d), color), axis=1)
        return p3d
    # [[X1,Y1,Z1],[X2,Y2,Z2], ... [Xn,Yn,Zn]]
    return p3d

