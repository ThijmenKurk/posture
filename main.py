import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import time
from datetime import datetime
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def print_usage_and_exit():
    print("usage: python3 main.py <capture_device_id> record good|bad")
    print("usage: python3 main.py <capture_device_id> classify [beep]")
    exit(1)


# Should use a proper parsing library for this, but it is a hobby project
if len(sys.argv) < 3:
    print_usage_and_exit()

DEFAULT_DESK_STATUS = "stand"
CAPTURE_DEVICE_ID = int(sys.argv[1])
# It is possible to add multiple video source to capture your posture from multiple angles
SOURCES = [
    (
        # Source 1
        cv2.VideoCapture(CAPTURE_DEVICE_ID),
        mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ),
        -90,
    ),
    # (
    #     # Source 2
    #     cv2.VideoCapture(1),
    #     mp_pose.Pose(
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5,
    #         model_complexity=1,
    #     ),
    #     0,
    # ),
]
# Landmarks are basically the points identified by mediapipe, each representing a certain joint or body part.
# This array is used to determine the landmarks (which may differ per source) to train the model with.
LANDMARK_INDICES = [list(range(12)), list()]  # Source 1  # Source 2
MODE = sys.argv[2]

if not MODE in ["record", "classify"]:
    print_usage_and_exit()


if MODE == "record":
    if len(sys.argv) != 3 or not sys.argv[3] in ["good", "bad"]:
        print_usage_and_exit()

    STATUS_VALUE = 1.0 if sys.argv[3] == "good" else 0.0

if MODE == "classify":
    BEEP = len(sys.argv) == 4 and sys.argv[3] == "beep"

timings = {}

# I keep track of the status of my electric desk inside of this file
# This file can be any of these values: sit|stand|transitioning
def get_desk_status():
    if os.path.exists("/home/thijmen/desk_status"):
        with open("/home/thijmen/desk_status") as f:
            return f.read().strip()
    else:
        return DEFAULT_DESK_STATUS


# Wait for desk transition to finish: sit->stand, stand->sit
def wait_for_desk_transition():
    while get_desk_status() == "transitioning":
        print("waiting for transition...")
        time.sleep(1)


# We use different datasets for different desk statuses
def get_samples_filename(desk_status):
    return f"{desk_status}_samples.csv"


# Rotate an image by a certain number of degrees
def rotate_image(frame, degs=-90):
    (h, w) = frame.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degs, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


# [[a], [b, c]] -> [a, b, c]
def flatten_array(t):
    return [item for sublist in t for item in sublist]


def current_time_in_milliseconds():
    return time.time() * 1000


# Identify the landmarks that can be found in a certain frame using the mediapipe library
# May return None if no landmark can be found.
def identify_landmarks(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return results.pose_landmarks.landmark if results.pose_landmarks else None


# This function saves the identified landmarks from all the sources to a csv file, along with the status value (good or bad).
def record(results):
    global timings

    current_time = current_time_in_milliseconds()
    desk_status = get_desk_status()
    file_name = get_samples_filename(desk_status)

    # If csv file does not exist, create new one with the proper column header
    if not os.path.exists(file_name):
        columns = "ts,ds,ps"
        for (idx, _) in enumerate(results):
            columns += "," + ",".join(
                flatten_array(
                    map(
                        lambda x: [
                            f"s{idx}l{x}x",
                            f"s{idx}l{x}y",
                            f"s{idx}l{x}z",
                            f"s{idx}l{x}v",
                        ],
                        range(33),
                    )
                )
            )

        with open(file_name, "w") as f:
            f.write(columns + "\n")

    # Append the recorded landmarks and specified status to the csv file
    with open(file_name, "a") as f:
        line = f"{datetime.now()},{desk_status},{STATUS_VALUE}"
        for _, landmarks in results:
            for landmark in landmarks:
                line += f",{landmark.x},{landmark.y},{landmark.z},{landmark.visibility}"

        f.write(f"{line}\n")

    timings["record"] = current_time_in_milliseconds() - current_time
    return STATUS_VALUE


# Classify if the posture is good or bad based on identified landmarks from all sources
def classify(model, results):
    global timings

    t = current_time_in_milliseconds()
    x = []

    for source_idx, (_, landmarks) in enumerate(results):
        for idx, landmark in enumerate(landmarks):
            if (
                idx not in LANDMARK_INDICES[source_idx]
            ):  # Only use the specified landmarks
                continue

            x.append(landmark.x)
            x.append(landmark.y)
            x.append(landmark.z)
            x.append(landmark.visibility)

    prediction = model.predict(np.array([x]))
    timings["predict"] = current_time_in_milliseconds() - t

    return prediction


# Draw skeleton on frame
def draw_pose(frame, landmarks):
    mp_drawing.draw_landmarks(
        frame,
        landmark_pb2.NormalizedLandmarkList(landmark=landmarks),
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )

    return frame


# Draw text on frame
def draw_text(frame, text, x, y, col=(255, 0, 0)):
    cv2.putText(
        frame,
        str(text),
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        col,
        3,
    )


# Draw timing information on frame
def draw_stats(frame, prediction, timings):
    print(timings)
    draw_text(frame, int(1 / (sum(timings.values()) / 1000)), 50, 50)
    draw_text(frame, get_desk_status(), 150, 50)
    draw_text(frame, "good" if prediction == 1 else "bad", 250, 50)


# Pipeline for a sinle source: Capture frame -> Rotate frame -> Identify landmarks
def pipeline_single(idx, vid, pose, rotate_degs):
    global timings

    t = current_time_in_milliseconds()
    success, frame = vid.read()
    if not success:
        print("failed on vid.read()")
        return False, None, None, None
    timings[f"read{idx}"] = current_time_in_milliseconds() - t

    if rotate_image:
        t = current_time_in_milliseconds()
        frame = rotate_image(frame, rotate_degs)
        timings[f"rotate{idx}"] = current_time_in_milliseconds() - t

    t = current_time_in_milliseconds()
    landmarks = identify_landmarks(frame, pose)
    if not landmarks:
        print("failed on process")
        return False, None, None, None
    timings[f"process{idx}"] = current_time_in_milliseconds() - t

    return True, frame, landmarks


# Pipeline for all sources
def pipeline(sources):
    results = []
    for idx, source in enumerate(sources):
        result = pipeline_single(idx, *source)
        if not result[0]:  # Fail if not all sources identified landmarks
            results.clear()
            break
        results.append(result[1:])
    return results


# Post pipeline for single source: Draw skeleton on frame -> Draw statistics
def post_pipeline_single(idx, prediction, frame, landmarks):
    global timings

    t = current_time_in_milliseconds()
    frame = draw_pose(frame, landmarks)
    timings[f"draw_pose{idx}"] = current_time_in_milliseconds() - t

    t = current_time_in_milliseconds()
    draw_stats(frame, prediction, timings)
    timings[f"draw_stats{idx}"] = current_time_in_milliseconds() - t

    # frame = cv2.resize(frame, (320, 240))

    cv2.imshow(f"feed{idx}", frame)
    cv2.namedWindow(f"feed{idx}", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(f"feed{idx}", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# Post pipeline for all sources
def post_pipeline(results, prediction):
    for idx, result in enumerate(results):
        post_pipeline_single(idx, prediction, *result)


# Train model based on a dataset which is determined using the desk value (stand|sit)
def train_model():
    desk_status = get_desk_status()

    print(f"training model for {desk_status}...")
    df = pd.read_csv(get_samples_filename(desk_status))
    df = df.drop(["ts", "ds"], axis=1)

    X = df[
        flatten_array(
            flatten_array(
                map(
                    lambda sx: map(  # sx[0] == source index, sx[1] == landmarks to use for the source
                        lambda x: [
                            f"s{sx[0]}l{x}x",
                            f"s{sx[0]}l{x}y",
                            f"s{sx[0]}l{x}z",
                            f"s{sx[0]}l{x}v",
                        ],
                        sx[1],
                    ),
                    enumerate(LANDMARK_INDICES),
                )
            )
        )
    ]
    y = df["ps"]  # posture status

    # Train model on entire dataset
    model_full = SVC(kernel="linear")
    model_full.fit(X, y)

    # Give score based on train-test-split for reference
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)
    print("score:", model.score(X_test, y_test))

    return desk_status, model_full


desk_status = None
i = 0
time_of_last_good_posture = current_time_in_milliseconds()
time_of_last_beep = current_time_in_milliseconds()
while True:
    wait_for_desk_transition()

    if (
        MODE == "classify" and desk_status != get_desk_status()
    ):  # Retrain model if desk transitioned
        desk_status, model = train_model()

    results = pipeline(SOURCES)
    if len(results) == 0:
        print("skipping since pose could not be determined for all sources")
        time.sleep(0.5)
        continue

    prediction = record(results) if MODE == "record" else classify(model, results)

    if prediction == 1.0:
        time_of_last_good_posture = current_time_in_milliseconds()

    if (
        MODE == "classify"
        and BEEP
        and current_time_in_milliseconds() - time_of_last_good_posture > 3000
    ):
        if current_time_in_milliseconds() - time_of_last_beep > 1500:
            os.system("play -nq -t alsa synth 0.2 sine 440")
            time_of_last_beep = current_time_in_milliseconds()

    post_pipeline(results, prediction)

    if MODE == "record" and i % 100 == 0:
        print(f"{i} sample(s) collected...")

    i += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    time.sleep(0.1)

# After the loop release the cap object
for source, _, _ in SOURCES:
    source.release()
# Destroy all the windows
cv2.destroyAllWindows()
