import cv2
import numpy as np

from Utils import rotationMatrixToEulerAngles, draw_pose_info


class HeadPoseEstimator:

    def __init__(self, camera_matrix=None, dist_coeffs=None, show_axis: bool = False):
        

        self.show_axis = show_axis
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def get_pose(self, frame, landmarks):
        
        self.keypoints = landmarks 
        self.frame = frame 

        self.axis = np.float32([[200, 0, 0],
                                [0, 200, 0],
                                [0, 0, 200]])
        # array that specify the length of the 3 projected axis from the nose

        if self.camera_matrix is None:
            self.size = frame.shape
            self.focal_length = self.size[1]
            self.center = (self.size[1] / 2, self.size[0] / 2)
            self.camera_matrix = np.array(
                [[self.focal_length, 0, self.center[0]],
                 [0, self.focal_length, self.center[1]],
                 [0, 0, 1]], dtype="double"
            )

        if self.dist_coeffs is None: 
            self.dist_coeffs = np.zeros((4, 1))

        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # 2D Point position of dlib face keypoints used for pose estimation
        self.image_points = np.array([
            (landmarks.part(30).x, landmarks.part(30).y),  # Nose tip
            (landmarks.part(8).x, landmarks.part(8).y),  # Chin
            (landmarks.part(36).x, landmarks.part(
                36).y),  # Left eye left corner
            (landmarks.part(45).x, landmarks.part(
                45).y),  # Right eye right corne
            (landmarks.part(48).x, landmarks.part(
                48).y),  # Left Mouth corner
            (landmarks.part(54).x, landmarks.part(
                54).y)  # Right mouth corner
        ], dtype="double")

        # compute the pose of the head using the image points and the 3D model points
        (success, rvec, tvec) = cv2.solvePnP(self.model_points, self.image_points,
                                             self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        

        if success:  

            rvec, tvec = cv2.solvePnPRefineVVS(
                self.model_points, self.image_points, self.camera_matrix, self.dist_coeffs, rvec, tvec)

            nose = (int(self.image_points[0][0]), int(self.image_points[0][1]))

            (nose_end_point2D, _) = cv2.projectPoints(
                self.axis, rvec, tvec, self.camera_matrix, self.dist_coeffs)

            Rmat = cv2.Rodrigues(rvec)[0]

            roll, pitch, yaw = rotationMatrixToEulerAngles(Rmat) * 180/np.pi


            if self.show_axis:
                self.frame = draw_pose_info(
                    self.frame, nose, nose_end_point2D, roll, pitch, yaw)
                # draws 3d axis from the nose and to the computed projection points
                for point in self.image_points:
                    cv2.circle(self.frame, tuple(
                        point.ravel().astype(int)), 2, (0, 255, 255), -1)
                # draws the 6 keypoints used for the pose estimation

            return self.frame, roll, pitch, yaw

        else:
            return None, None, None, None
