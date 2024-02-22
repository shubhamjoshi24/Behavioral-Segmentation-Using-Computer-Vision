import time


class Activity:

    def __init__(self, capture_fps: int, ear_tresh, gaze_tresh, perclos_tresh=0.2, ear_time_tresh=4.0, pitch_tresh=35,
                 yaw_tresh=30, gaze_time_tresh=4.0, roll_tresh=None, pose_time_tresh=4.0):
       

        self.fps = capture_fps
        self.delta_time_frame = (1.0 / capture_fps)  # estimated frame time
        self.prev_time = 0 
        # default time period for PERCLOS (60 seconds)
        self.perclos_time_period = 60
        self.perclos_tresh = perclos_tresh

        self.ear_tresh = ear_tresh
        self.ear_act_tresh = ear_time_tresh / self.delta_time_frame
        self.ear_counter = 0
        self.eye_closure_counter = 0

        self.gaze_tresh = gaze_tresh
        self.gaze_act_tresh = gaze_time_tresh / self.delta_time_frame
        self.gaze_counter = 0

        self.roll_tresh = roll_tresh
        self.pitch_tresh = pitch_tresh
        self.yaw_tresh = yaw_tresh
        self.pose_act_tresh = pose_time_tresh / self.delta_time_frame
        self.pose_counter = 0

    def Distraction_Type(self, ear_score, gaze_score, head_roll, head_pitch, head_yaw):
    
        # instantiating state of attention variables
        asleep = False
        looking_away = False
        distracted = False

        if self.ear_counter >= self.ear_act_tresh:  # check if the ear counter surpassed the threshold
            asleep = True

        if self.gaze_counter >= self.gaze_act_tresh:  # check if the gaze counter surpassed the threshold
            looking_away = True

        if self.pose_counter >= self.pose_act_tresh:  # check if the pose Counter surpassed the threshold
            distracted = True

        if (ear_score is not None) and (ear_score <= self.ear_tresh):
            if not asleep:
                self.ear_counter += 1
        elif self.ear_counter > 0:
            self.ear_counter -= 1

        if (gaze_score is not None) and (gaze_score >= self.gaze_tresh):
            if not looking_away:
                self.gaze_counter += 1
        elif self.gaze_counter > 0:
            self.gaze_counter -= 1

        if ((self.roll_tresh is not None and head_roll is not None and head_roll > self.roll_tresh) or (
                head_pitch is not None and abs(head_pitch) > self.pitch_tresh) or (
                head_yaw is not None and abs(head_yaw) > self.yaw_tresh)):
            if not distracted:
                self.pose_counter += 1
        elif self.pose_counter > 0:
            self.pose_counter -= 1
            # ##################################################################################################

        return asleep, looking_away, distracted

    def get_PERCLOS(self, ear_score):

        delta = time.time() - self.prev_time  # set delta timer
        tired = False 

        # if the ear_score is lower or equal than the threshold, increase the eye_closure_counter
        if (ear_score is not None) and (ear_score <= self.ear_tresh):
            self.eye_closure_counter += 1

        # compute the cumulative eye closure time
        closure_time = (self.eye_closure_counter * self.delta_time_frame)
        perclos_score = (closure_time) / self.perclos_time_period

        if perclos_score >= self.perclos_tresh: 
            tired = True

        if delta >= self.perclos_time_period:  # at every end of the given time period, reset the counter and the timer
            self.eye_closure_counter = 0
            self.prev_time = time.time()

        return tired
