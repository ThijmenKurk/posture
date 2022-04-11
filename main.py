# import the opencv library
from re import S
from black import T
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
from sklearn import metrics
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

timings = {}

os.system("bash /home/thijmen/projects/posture/connect_droidcam.sh")

MODE = sys.argv[1]
assert MODE in ["record", "classify"]

LANDMARK_INDICES = [list(range(33)), list()]

if MODE == "record":
    assert sys.argv[2] in ["good", "bad"]

    STATUS_VALUE = 1.0 if sys.argv[2] == "good" else 0.0

if MODE == "classify":
    BEEP = len(sys.argv) == 3 and sys.argv[2] == "beep"


def desk_status():
    with open("/home/thijmen/desk_status") as f:
        return f.read().strip()


def get_samples_filename(desk_status):
    return f"{desk_status}_samples.csv"


def rotate(frame, degs=-90):
    (h, w) = frame.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), degs, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


def flatten(t):
    return [item for sublist in t for item in sublist]


def landmark_columns(s):
    return ",".join(
        flatten(
            map(
                lambda x: [
                    f"s{s}l{x}x",
                    f"s{s}l{x}y",
                    f"s{s}l{x}z",
                    f"s{s}l{x}v",
                ],
                range(33),
            )
        )
    )


def process(frame, pose):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    return results.pose_landmarks.landmark if results.pose_landmarks else None


def record(results):
    global timings

    t = time_ms()
    ds = desk_status()
    file_name = get_samples_filename(ds)

    if not os.path.exists(file_name):
        columns = "ts,ds,ps"
        for (idx, _) in enumerate(results):
            columns += "," + landmark_columns(idx)

        with open(file_name, "w") as f:
            f.write(columns + "\n")

    with open(file_name, "a") as f:
        line = f"{datetime.now()},{ds},{STATUS_VALUE}"
        for _, landmarks in results:
            for landmark in landmarks:
                line += f",{landmark.x},{landmark.y},{landmark.z},{landmark.visibility}"

        f.write(f"{line}\n")

    timings["record"] = time_ms() - t
    return STATUS_VALUE


def classify(model, results):
    global timings

    t = time_ms()
    x = []

    for source_idx, (_, landmarks) in enumerate(results):
        for idx, landmark in enumerate(landmarks):
            if idx not in LANDMARK_INDICES[source_idx]:
                continue

            x.append(landmark.x)
            x.append(landmark.y)
            x.append(landmark.z)
            x.append(landmark.visibility)

    prediction = model.predict(np.array([x]))
    timings["predict"] = time_ms() - t

    return prediction


def draw_pose(frame, landmarks):
    mp_drawing.draw_landmarks(
        frame,
        landmark_pb2.NormalizedLandmarkList(landmark=landmarks),
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )

    return frame


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


def draw_stats(frame, prediction, timings):
    draw_text(frame, int(1 / (sum(timings.values()) / 1000)), 50, 50)
    draw_text(frame, desk_status(), 150, 50)
    draw_text(frame, "good" if prediction == 1 else "bad", 250, 50)


def time_ms():
    return time.time() * 1000


def pipeline_single(idx, vid, pose, rotate_degs):
    global timings

    t = time_ms()
    success, frame = vid.read()
    if not success:
        print("failed on vid.read()")
        return False, None, None, None
    timings[f"read{idx}"] = time_ms() - t

    if rotate:
        t = time_ms()
        frame = rotate(frame, rotate_degs)
        timings[f"rotate{idx}"] = time_ms() - t

    t = time_ms()
    landmarks = process(frame, pose)
    if not landmarks:
        print("failed on process")
        return False, None, None, None
    timings[f"process{idx}"] = time_ms() - t

    return True, frame, landmarks


def pipeline(sources):
    results = []
    for idx, source in enumerate(sources):
        result = pipeline_single(idx, *source)
        if not result[0]:
            results.clear()
            break
        results.append(result[1:])
    return results


def post_pipeline_single(idx, prediction, frame, landmarks):
    global timings

    t = time_ms()
    frame = draw_pose(frame, landmarks)
    timings[f"draw_pose{idx}"] = time_ms() - t

    t = time_ms()
    draw_stats(frame, prediction, timings)
    timings[f"draw_stats{idx}"] = time_ms() - t

    # frame = cv2.resize(frame, (320, 240))

    cv2.imshow(f"feed{idx}", frame)
    cv2.namedWindow(f"feed{idx}", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(f"feed{idx}", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def post_pipeline(results, prediction):
    for idx, result in enumerate(results):
        post_pipeline_single(idx, prediction, *result)


def train_model():
    ds = desk_status()

    print(f"training model for {ds}...")
    df = pd.read_csv(get_samples_filename(ds))
    df = df.drop(["ts", "ds"], axis=1)

    X = df[
        flatten(
            flatten(
                map(
                    lambda sx: map(
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
    y = df["ps"]

    model = SVC(kernel="linear")
    model.fit(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model_t = SVC(kernel="linear")
    model_t.fit(X_train, y_train)

    print("score:", model_t.score(X_test, y_test))

    return ds, model


def wait_for_desk_transition():
    while desk_status() == "transitioning":
        print("waiting for transition...")
        time.sleep(1)


sources = [
    (
        cv2.VideoCapture(2),
        mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        ),
        -90,
    ),
    # (
    #     cv2.VideoCapture(1),
    #     mp_pose.Pose(
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5,
    #         model_complexity=1,
    #     ),
    #     0,
    # ),
]

ds = None
i = 0
last_good = time_ms()
last_play = time_ms()
while True:
    wait_for_desk_transition()

    if MODE == "classify" and ds != desk_status():
        ds, model = train_model()

    results = pipeline(sources)
    if len(results) == 0:
        print("skipping since pose could not be determined for all sources")
        time.sleep(0.5)
        continue

    prediction = record(results) if MODE == "record" else classify(model, results)

    if prediction == 1.0:
        last_good = time_ms()

    post_pipeline(results, prediction)

    if MODE == "record" and i % 100 == 0:
        print(f"{i} sample(s) collected...")

    if MODE == "classify" and BEEP and time_ms() - last_good > 3000:
        if time_ms() - last_play > 1500:
            os.system("play -nq -t alsa synth 0.2 sine 440")
            last_play = time_ms()

    i += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# After the loop release the cap object
for source, _, _ in sources:
    source.release()
# Destroy all the windows
cv2.destroyAllWindows()
