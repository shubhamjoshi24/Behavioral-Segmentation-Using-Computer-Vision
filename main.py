import os
import time

import cv2
import dlib

from Detecting_Activity import Activity as ActFinder
from Get_Eye import EyeDetector as EyeDet
from Getting_Pose import HeadPoseEstimator as HeadPoseEst
from Utils import get_face_area


def main(counter=None):
    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except:
            print(
                "OpenCV optimization could not be set to True")

    camera = 0
    fps_limit = 11
    show_fps = True
    show_proc_time = True
    show_eye_proc = False  #############################
    show_axis = True
    ctime = 0
    ptime = 0
    prev_time = 0  # used to set the FPS limit
    fps_lim = fps_limit
    time_lim = 1. / fps_lim

    # Attention Scorer parameters (EAR, Gaze Score, Pose)
    ear_tresh = 0.15
    ear_time_tresh = 2
    gaze_tresh = 0.2
    gaze_time_tresh = 2
    pitch_tresh = 35
    yaw_tresh = 28
    pose_time_tresh = 2.5

    Detector = dlib.get_frontal_face_detector()
    Predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")  # keypoint in face 

    Eye_det = EyeDet(show_processing=show_eye_proc)  ##########################
    Head_pose = HeadPoseEst(show_axis=show_axis)
    counter1 = 1;
    counter2 = 1;
    counter3 = 1;
    path1 = "Asleep"
    path2 = "Destracted"
    path3 = "LookingAway"
    flag = 0;

    Scorer = ActFinder(fps_lim, ear_tresh=ear_tresh, ear_time_tresh=ear_time_tresh, gaze_tresh=gaze_tresh,
                       gaze_time_tresh=gaze_time_tresh, pitch_tresh=pitch_tresh, yaw_tresh=yaw_tresh,
                       pose_time_tresh=pose_time_tresh)
    # Camera capture here
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:  # infinite loop 

        delta_time = time.perf_counter() - prev_time  # FPS capping
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame from camera/stream end")
            break

        if camera == 0:
            frame = cv2.flip(frame, 2)

        if delta_time >= time_lim:  # if the time passed is bigger or equal than the frame time, process the frame
            prev_time = time.perf_counter()

            # compute the FPS 
            ctime = time.perf_counter()
            fps = 1.0 / float(ctime - ptime)
            ptime = ctime

            e1 = cv2.getTickCount()
            cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 1)  # display fps

            # transform the BGR frame in grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 5, 10, 10)  # reduce noise
            faces = Detector(gray)

            if len(faces) > 0:  # if face is found
                faces = sorted(faces, key=get_face_area, reverse=True)
                student_face = faces[0]
                landmarks = Predictor(gray, student_face)  # predict the 68 facial keypoints positionR

                Eye_det.show_eye_keypoints(color_frame=frame, landmarks=landmarks)  # shows the eye keypoints

                # compute the EAR score of the eyes
                ear = Eye_det.get_EAR(frame=gray, landmarks=landmarks)

                # compute the PERCLOS score and state of tiredness
                tired = Scorer.get_PERCLOS(ear)

                # compute the Gaze Score
                gaze = Eye_det.get_Gaze_Score(
                    frame=gray, landmarks=landmarks)

                frame_det, roll, pitch, yaw = Head_pose.get_pose(
                    frame=frame, landmarks=landmarks)

                if frame_det is not None:
                    frame = frame_det

                if ear is not None:
                    cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                if gaze is not None:
                    cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)

                if tired:
                    cv2.putText(frame, "TIRED!", (10, 340),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                asleep, looking_away, distracted = Scorer.Distraction_Type(ear, gaze, roll, pitch, yaw)

                # if the state of attention of the student is not normal, show an alert on screen
                if asleep:
                    cv2.putText(frame, "ASLEEP!", (10, 300),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    if flag != 1:
                        flag = 1
                        img_name = "Asleep{}.png".format(counter1)
                        counter1 = counter1 + 1
                        cv2.imwrite(os.path.join(path1, img_name), frame)

                elif distracted:
                    cv2.putText(frame, "DISTRACTED!", (10, 340),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    if flag != 2:
                        flag = 2
                        img_name = "destracted{}.png".format(counter2)
                        counter2 = counter2 + 1
                        cv2.imwrite(os.path.join(path2, img_name), frame)

                elif looking_away:
                    cv2.putText(frame, "LOOKING AWAY!", (20, 320),
                                cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    if flag != 3:
                        flag = 3
                        img_name = "LookingAway{}.png".format(counter3)
                        counter3 = counter3 + 1
                        cv2.imwrite(os.path.join(path3, img_name), frame)
                if looking_away == distracted == asleep == tired == False:
                    cv2.putText(frame, "NORMAL!", (180, 60),
                                cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No Face Detected!! Sit Properly!", (40, 60),
                            cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # stop the tick counter for computing the processing time for each frame
            e2 = cv2.getTickCount()
            # processign time in milliseconds
            proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000

            if show_fps:
                cv2.putText(frame, "FPS:" + str(round(fps, 0)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)

            cv2.imshow("Frame", frame)
        #######################################
        if cv2.waitKey(1) % 256 == 32:
            img_name = "myImg{}.png".format(counter)
            cv2.imwrite(img_name, frame)
            counter += 1;
        ##########################
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty('Frame', 0) < 0:
            break;

    cap.release()
    cv2.destroyAllWindows()

    return


if __name__ == "__main__":
    main()
