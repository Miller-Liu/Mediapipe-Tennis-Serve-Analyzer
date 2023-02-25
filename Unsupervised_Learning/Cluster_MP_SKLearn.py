import mediapipe as mp
import cv2 as cv2
import numpy as np
import math
from pathlib import Path
from sklearn.cluster import KMeans
from numpy import linalg as LA
from moviepy.editor import *
import moviepy.video.fx.all as vfx

# initialization for mediapipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
DESIRED_HEIGHT = 400
DESIRED_WIDTH = 500
root_dir = os.path.join(os.path.expanduser('~'), 'PycharmProjects', 'Tennis_AI', 'Tennis_AI_Complete')


# Configure K-means model
def K_means(angles_frames):
    X = np.array(angles_frames)
    kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
    print(kmeans.labels_)
    return kmeans


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
                if not cv2.imwrite(os.path.join(root_dir, "Unsupervised_Learning", "Video_Frames", f"{path}_{counter}.jpg"), stack):
                    raise Exception("Could not write image")
            except Exception as E:
                print(E)
                pass
        video_cap.release()
        cv2.destroyAllWindows()
        return angles


def main(path1, path2):
    data1 = Process_Video(path1)  # Angle list
    result1 = K_means(data1)  # K-means to group frames of vid1

    # Separate the frames processed by the labels from K-means
    def process_result_labels(results_labels):
        array = results_labels.tolist()
        return_array = [(0, 0)]
        # loop through array and return ranges for each label
        for i in range(1, len(array)):
            if array[i] != array[i - 1]:
                # if the length of this range is too small
                if i - return_array[-1][1] + 1 <= 2:
                    return_array[-1] = (return_array[-1][0], i)
                # general case
                else:
                    return_array.append((return_array[-1][1] + 1, i))
        return_array.append((return_array[-1][1] + 1, len(array)))
        return return_array[1:]

    data2 = Process_Video(path2)  # Angle list
    result2 = K_means(data2)  # K-means to group frames of vid2

    groups = process_result_labels(result1.labels_)  # Get where each group is (ex: first group is from 0 to 20)
    processed_groups = []
    matches = []

    def calculate_distance(angles, frame, labels, prediction_label):
        # take the frame being predicted, along with its matching number.
        min_value = [0, 0]
        test_frame = np.array(frame)
        # return the indexes where the predicted label occur in vid2 (use [0] because it returns list, and then dtype)
        indexes = np.where(prediction_label == labels)[0]
        for i in indexes:
            current_angles = np.array(angles[i])
            score = LA.norm(current_angles - test_frame)
            if min_value == [0, 0] or score < min_value[1]:
                min_value = [i + 1, score]
        # return the frame with the least value (most matched)
        return min_value[0]

    # Take the frame numbers of the groups of vid1
    for i in groups:
        processed_groups.append(int((i[0] + i[1]) / 2))

    # Use the results from vid2 to predict where the frames of the groups of vid1 would match with
    for i in processed_groups:
        prediction = result2.predict([data1[i]])
        matched_frame = calculate_distance(data2, data1[i], result2.labels_, prediction)
        matches.append([i, matched_frame])

    def percent_match(a, b):
        percent_match_list = []
        for i in range(8):
            percent_match_list.append(round(float(data2[b][i] / data1[a][i] * 100), 2))
        return percent_match_list

    # print(matches)
    percentage_match = []
    for i in matches:
        # find percentage match between angles of matching frames
        percentage_match.append(percent_match(i[0], i[1]))
    # print(percentage_match)

    lengths = [[0]]
    previous_term = matches[0][1]
    # Go through matching and break it up into lists of increasing frames
    # ex: matching from [0-5] is increasing, but went from 90 -> 30 on number [6], and then increased, break it apart
    for i in range(1, len(matches)):
        if matches[i][1] > previous_term:
            lengths[-1].append(i)
        else:
            lengths.append([i])
        previous_term = matches[i][1]

    max_sequence = [0]
    # Take all the subsegments and find the longest one
    for i in lengths:
        if len(i) > len(max_sequence):
            max_sequence = i

    path = []
    temp_counter = 0
    for i in max_sequence:
        # Take the sequence of frames and save images of vid1 and vid2 frames concatenated together
        temp_counter += 1
        img1 = cv2.imread(os.path.join(root_dir, "Unsupervised_Learning", "Video_Frames", Path(path1).stem + "_" + str(matches[i][0]) + ".jpg"))
        img2 = cv2.imread(os.path.join(root_dir, "Unsupervised_Learning", "Video_Frames", Path(path2).stem + "_" + str(matches[i][1]) + ".jpg"))
        if img1.shape[0] > img2.shape[0]:
            extra_part = np.zeros((img1.shape[0] - img2.shape[0], img2.shape[1], 3), np.uint8)
            img2 = np.concatenate((extra_part, img2), axis=0)
        if img1.shape[0] < img2.shape[0]:
            extra_part = np.zeros((img2.shape[0] - img1.shape[0], img1.shape[1], 3), np.uint8)
            img1 = np.concatenate((extra_part, img1), axis=0)
        final_img = np.concatenate((img1, img2), axis=1)
        path.append(str(Path(path1).stem) + f"{temp_counter}.jpg")
        cv2.imwrite(os.path.join(root_dir, "Static", "Images", str(Path(path1).stem) + f"{temp_counter}.jpg"), final_img)

    return_array = []
    # return an array of the matched frame numbers
    for i in max_sequence:
        return_array.append([matches[i][0], matches[i][1]])

    return path1, path2, return_array, path


def Play_Video(vid1, vid2, frame_array):
    # print(frame_array)
    frame_array.append([0, 0])
    video1 = VideoFileClip(vid1)
    video2 = VideoFileClip(vid2)
    video_1_subclips = []
    video_2_subclips = []
    fps = [video1.fps, video2.fps]
    # Adjust the times for the subsegments and piece them together in one video
    for i in range(len(frame_array) - 1):
        # print(frame_array[i - 1], frame_array[i])
        temp_1_clip = video1.subclip(frame_array[i - 1][0] / fps[0], (frame_array[i][0] + 1) / fps[0])
        temp_2_clip = video2.subclip(frame_array[i - 1][1] / fps[1], (frame_array[i][1] + 1) / fps[1])
        temp_1_clip_time = (frame_array[i][0] + 1) / fps[0] - frame_array[i - 1][0] / fps[0]
        temp_2_clip_time = (frame_array[i][1] + 1) / fps[1] - frame_array[i - 1][1] / fps[1]
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
    final_video_1 = concatenate_videoclips(video_1_subclips[1:])
    final_video_2 = concatenate_videoclips(video_2_subclips[1:])
    path1 = Path(vid1).stem
    path2 = Path(vid2).stem
    final_video_1.write_videofile(os.path.join(root_dir, "Static", "Videos", f"{path1}.mp4"))
    final_video_2.write_videofile(os.path.join(root_dir, "Static", "Videos", f"{path2}.mp4"))
    return path1 + ".mp4", path2 + ".mp4"


def remove_all_files():
    for files in os.listdir(os.path.join(root_dir, "Unsupervised_Learning", "Video_Frames")):
        os.remove(os.path.join(root_dir, "Unsupervised_Learning", "Video_Frames", files))


def execute_all(path1, path2):
    path1 = os.path.join(root_dir, "Static", "uploads", path1)
    path2 = os.path.join(root_dir, "Static", "uploads", path2)
    a, b, c, d = main(path1, path2)
    e = Play_Video(a, b, c)
    remove_all_files()
    return d, e

