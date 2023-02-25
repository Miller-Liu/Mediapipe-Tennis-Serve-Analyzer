import os
from Supervised_Learning.Scripts.Angles_Support_Vector_Machine_Multiclass_Classification import SVM
from Supervised_Learning.Scripts.Angles_Nearest_Centroid import NC
from Supervised_Learning.Scripts.Angles_Nearest_Component_Analysis import NCA
from Supervised_Learning.Scripts.Angles_Nearest_Neighbors_Classification import NNC
import cv2
import mediapipe as mp
import math
import numpy as np
from pathlib import Path
from moviepy.editor import *
import moviepy.video.fx.all as vfx

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
DESIRED_HEIGHT = 400
DESIRED_WIDTH = 500
root_dir = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'Tennis_AI', 'Tennis_AI_Complete')
current_dir = os.path.join(root_dir, 'Supervised_Learning')


def create_model(input_choice):
    if input_choice == "SVM":
        return SVM()
    if input_choice == "NC":
        return NC()
    if input_choice == "NCA":
        return NCA()
    if input_choice == "NNC":
        return NNC()


# Resizing image into specified width and height
def resize(image):
    h, w = image.shape[:2]
    # Set the longer side to the desired amount and scale the smaller side accordingly
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    extra_column = np.zeros((img.shape[0], 200, 3), np.uint8)
    return np.concatenate((extra_column, img), axis=1)


text_height = 0


def Process_Video(file_name):
    global text_height
    # initialize counter that counts frames
    counter = 0
    # angle list
    angles = []
    # set the video being processed
    video_cap = cv2.VideoCapture(file_name)

    # Find angle function
    def Find_Angle(a, b, c, input_image, label):
        global text_height
        use_z = False
        # calculate distances with and without z value
        if use_z:
            a_to_b = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
            a_to_c = math.sqrt((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2 + (c[2] - a[2]) ** 2)
            b_to_c = math.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2 + (c[2] - b[2]) ** 2)
        else:
            a_to_b = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
            a_to_c = math.sqrt((c[0] - a[0]) ** 2 + (c[1] - a[1]) ** 2)
            b_to_c = math.sqrt((c[0] - b[0]) ** 2 + (c[1] - b[1]) ** 2)

        # Law of cosines for Angles
        angle = math.acos((b_to_c ** 2 + a_to_b ** 2 - a_to_c ** 2) / (2 * b_to_c * a_to_b))
        angle = str(round(angle * 180 / math.pi, 2))
        final_image = Add_Text((20, text_height), input_image, label + ": " + angle)
        text_height += 20
        return float(angle), final_image

    # Put text on image
    def Add_Text(position, input_image, angle):
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # position
        position = position
        # fontScale
        fontScale = 0.4
        # Blue color in BGR
        color = (255, 255, 255)
        # Line thickness of 2 px
        thickness = 1
        # Using cv2.putText() method
        final_image = cv2.putText(input_image, angle, position, font,
                                  fontScale, color, thickness, cv2.LINE_AA)
        return final_image

    # Create MP Model
    with mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) as pose:
        while video_cap.isOpened():
            flag, frame = video_cap.read()
            counter += 1
            text_height = 20

            # If doesn't receive frame
            if not flag:
                break

            image = resize(frame)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            image_height, image_width, _ = image.shape

            if not results.pose_landmarks:
                print("no pose landmarks")
                pass

            # Create black image with same dimensions as image1
            black_picture = np.zeros((image_height, image_width, 3), np.uint8)

            mp_drawing.draw_landmarks(
                black_picture,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            try:
                data = {"leftShoulder": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                                         results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z],
                        "leftElbow": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z],
                        "leftWrist": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z],
                        "leftHip": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,
                                    results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z],
                        "leftKnee": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z],
                        "leftAnkle": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z],
                        "rightShoulder": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                                          results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                                          results.pose_landmarks.landmark[
                                              mp_pose.PoseLandmark.RIGHT_SHOULDER].z],
                        "rightElbow": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z],
                        "rightWrist": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z],
                        "rightHip": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,
                                     results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z],
                        "rightKnee": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                                      results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z],
                        "rightAnkle": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
                                       results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z],
                        "nose": [results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x,
                                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y,
                                 results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z],
                        }
                left_Elbow, black_picture = Find_Angle(data["leftWrist"], data["leftElbow"], data["leftShoulder"],
                                                       black_picture, "Left Elbow")
                left_Shoulder, black_picture = Find_Angle(data["leftElbow"], data["leftShoulder"], data["leftHip"],
                                                          black_picture, "Left Shoulder")
                left_Hip, black_picture = Find_Angle(data["leftShoulder"], data["leftHip"], data["leftKnee"],
                                                     black_picture, "Left Hip")
                left_Knee, black_picture = Find_Angle(data["leftHip"], data["leftKnee"], data["leftAnkle"],
                                                      black_picture, "Left Knee")
                right_Elbow, black_picture = Find_Angle(data["rightWrist"], data["rightElbow"], data["rightShoulder"],
                                                        black_picture, "Right Elbow")
                right_Shoulder, black_picture = Find_Angle(data["rightElbow"], data["rightShoulder"], data["rightHip"],
                                                           black_picture, "Right Shoulder")
                right_Hip, black_picture = Find_Angle(data["rightShoulder"], data["rightHip"], data["rightKnee"],
                                                      black_picture, "Right Hip")
                right_Knee, black_picture = Find_Angle(data["rightHip"], data["rightKnee"], data["rightAnkle"],
                                                       black_picture, "Right Knee")
                angles.append([left_Elbow, left_Shoulder, left_Hip, left_Knee,
                               right_Elbow, right_Shoulder, right_Hip, right_Knee])
                # Stack images together and save it
                stack = np.concatenate((image, black_picture), axis=0)
                # cv2.imshow("Test", image)
                path = Path(file_name).stem
                if not cv2.imwrite(
                        os.path.join(root_dir, "Supervised_Learning", "Video_Frames", f"{path}_{counter}.jpg"), stack):
                    raise Exception("Could not write image")
            except Exception as E:
                print(E)
                pass
        video_cap.release()
        cv2.destroyAllWindows()
        return angles


def Process(input_array):
    processed_array_1 = []

    def process1(arr):
        processed_angles_1 = [[], [], [], [], [], []]
        last_point = 0
        for i in range(1, len(arr)):
            if arr[i] != arr[i - 1]:
                processed_angles_1[arr[i - 1]].append((last_point, i - 1))
                last_point = i
        # print(processed_angles_1)
        return processed_angles_1

    def process2(arr):
        processed_angles_2 = [[], [], [], [], [], []]
        for i in range(len(arr)):
            for j in arr[i]:
                if j[1] - j[0] >= 3:
                    processed_angles_2[i].append(j)
        # print(processed_angles_2)
        return processed_angles_2

    def process3(arr, index, curr, length):
        current = curr
        if index < 6:
            if arr[index]:  # if the label at index is not empty
                for i in range(len(arr[index])):
                    if not curr:  # curr empty
                        current.append(arr[index][i])
                        # print(1, index, current)
                        process3(arr, index + 1, current, 1)
                        break  # loop shouldn't continue after recursion above (index can be at 5, but goes back to 1 in for loop)
                    elif arr[index][i][0] > curr[-1][-1]:  # current tuple comes after previous
                        current.append(arr[index][i])
                        # print(2, index, current)
                        process3(arr, index + 1, current, length + 1)
                        break  # loop shouldn't continue after recursion above (index can be at 5, but goes back to 1 in for loop)
                    elif arr[index][i][0] < curr[-1][-1]:  # current tuple begins before previous
                        # print(current)
                        if current not in processed_array_1:
                            processed_array_1.append(current)
            else:
                process3(arr, index + 1, current, length)
        else:
            # print(current)
            if current not in processed_array_1:
                processed_array_1.append(current)

    temparr = process2(process1(input_array))
    for i in range(5):
        process3(temparr, i, [], 0)

    max_length = [[]]
    for i in processed_array_1:
        if len(i) > len(max_length[0]):
            max_length[0] = i
        if len(i) == len(max_length[0]) and i not in max_length:
            max_length.append(i)

    notation_array = []
    for i in range(len(max_length)):
        notation_array.append([])
        for j in max_length[i]:
            for k in range(len(temparr)):
                if j in temparr[k]:
                    notation_array[-1].append(k)

    if len(max_length) == 1 and len(notation_array) == 1:
        # print(max_length[0], notation_array[0])
        return max_length[0], notation_array[0]
    else:
        max_summation = (0, 0)
        for i in range(len(max_length)):
            summation = 0
            for j in range(len(max_length[i])):
                summation += max_length[i][j][1] - max_length[i][j][0]
            if summation > max_summation[0]:
                max_summation = (summation, i)
        # print(max_length[max_summation[1]], notation_array[max_summation[1]])
        return max_length[max_summation[1]], notation_array[max_summation[1]]


def synchronize_videos(vid1, vid2, predictions1, predictions2, labels1, labels2, end1, end2):
    video1 = VideoFileClip(vid1)
    video2 = VideoFileClip(vid2)
    video_1_subclips = []
    video_2_subclips = []
    fps = [video1.fps, video2.fps]
    matched_frames = []
    for i in range(6):
        if i in labels1 and i in labels2:
            matched_frames.append([predictions1[labels1.index(i)][0], predictions2[labels2.index(i)][0]])
            matched_frames.append([predictions1[labels1.index(i)][1], predictions2[labels2.index(i)][1]])

    matched_frames.append([end1, end2])

    for i in range(1, len(matched_frames)):
        temp_1_clip = video1.subclip(matched_frames[i - 1][0] / fps[0], (matched_frames[i][0] + 1) / fps[0])
        temp_2_clip = video2.subclip(matched_frames[i - 1][1] / fps[1], (matched_frames[i][1] + 1) / fps[1])
        temp_1_clip_time = (matched_frames[i][0] + 1) / fps[0] - matched_frames[i - 1][0] / fps[0]
        temp_2_clip_time = (matched_frames[i][1] + 1) / fps[1] - matched_frames[i - 1][1] / fps[1]
        temp_fps = [temp_1_clip.fps, temp_2_clip.fps]
        # fps1 * t1 = f1, fps2 * t2 = f2 => t1 = f1 / fps1, t2 = f2 / fps2
        if temp_1_clip_time < temp_2_clip_time:
            # t1 < t2 => f1 / fps1 < f2 / fps2
            # f1 / fps1 = f2 / fps2  =>  fps1 * f1 * fps2 / f2 / fps1 = fps1 * t1 / t2
            adjustment_ratio = temp_1_clip_time / temp_2_clip_time
            temp_1_clip = temp_1_clip.set_fps(temp_fps[0] * adjustment_ratio)
            temp_1_clip = temp_1_clip.fx(vfx.speedx, adjustment_ratio)
        elif temp_1_clip_time > temp_2_clip_time:
            # t1 > t2 => f1 / fps1 > f2 / fps2
            # f1 / fps1 = f2 / fps2  =>  fps2 * f2 * fps1 / f1 / fps2 = fps2 * t2 / t1
            adjustment_ratio = temp_2_clip_time / temp_1_clip_time
            temp_2_clip = temp_2_clip.set_fps(temp_fps[1] * adjustment_ratio)
            temp_2_clip = temp_2_clip.fx(vfx.speedx, adjustment_ratio)
        video_1_subclips.append(temp_1_clip)
        video_2_subclips.append(temp_2_clip)
    final_video_1 = concatenate_videoclips(video_1_subclips)
    final_video_2 = concatenate_videoclips(video_2_subclips)
    path1 = Path(vid1).stem
    path2 = Path(vid2).stem
    final_video_1.write_videofile(os.path.join(root_dir, "Static", "Videos", f"{path1}.mp4"))
    final_video_2.write_videofile(os.path.join(root_dir, "Static", "Videos", f"{path2}.mp4"))
    return path1 + ".mp4", path2 + ".mp4"


def main(path1, path2, choice):
    angle1 = Process_Video(path1)
    angle2 = Process_Video(path2)
    model = create_model(choice)
    predictions1 = model.predict(angle1).tolist()
    predictions2 = model.predict(angle2).tolist()

    # Modified LIS
    print(predictions1, predictions2)

    processed_predictions_1, annotate_1 = Process(predictions1)
    processed_predictions_2, annotate_2 = Process(predictions2)

    synchronize_videos(path1, path2, processed_predictions_1, processed_predictions_2, annotate_1, annotate_2, len(angle1) - 1, len(angle2) - 1)


def remove_all_files():
    for files in os.listdir(os.path.join(root_dir, "Supervised_Learning", "Video_Frames")):
        os.remove(os.path.join(root_dir, "Supervised_Learning", "Video_Frames", files))


def supervised_learning(path1, path2, choice):
    path1 = os.path.join(root_dir, "Static", "uploads", path1)
    path2 = os.path.join(root_dir, "Static", "uploads", path2)
    main(path1, path2, choice)


def execute():
    supervised_learning("video1.mp4", "video2.mp4", "NNC")
    remove_all_files()
