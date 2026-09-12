import cv2
import math
import numpy as np


VIDEO_PATH = "/home/xhm/ikun_ws/video/input.mp4"
OUTPUT_PATH = "/home/xhm/ikun_ws/video/output_2_framewise_v4.mp4"


# 最后保存的 Trackbar 参数
HSV_LOWER = np.array([0, 120, 70], dtype=np.uint8)
HSV_UPPER = np.array([30, 255, 255], dtype=np.uint8)
RGB_LOWER = np.array([0, 0, 0], dtype=np.uint8)
RGB_UPPER = np.array([255, 255, 255], dtype=np.uint8)

OPEN_VALUE = 1
CLOSE_VALUE = 3

INIT_MIN_AREA = 250
TRACK_MIN_AREA = 40
MIN_BALL_RADIUS = 15
MAX_BALL_RADIUS = 110
MAX_MISSING_FRAMES = 18
MAX_TRAJECTORY = 80
MIN_REGION_AREA = 250
BOX_SCALE = 1.15


def create_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    kalman.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman


def make_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    rgb_mask = cv2.inRange(frame, RGB_LOWER, RGB_UPPER)
    mask = cv2.bitwise_and(hsv_mask, rgb_mask)

    if OPEN_VALUE > 0:
        size = OPEN_VALUE * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if CLOSE_VALUE > 0:
        size = CLOSE_VALUE * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (size, size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def detect_initial(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < INIT_MIN_AREA or circularity(contour) < 0.35:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        ratio = width / float(height) if height else 0.0
        if not 0.55 < ratio < 1.55:
            continue

        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        if not MIN_BALL_RADIUS <= radius <= MAX_BALL_RADIUS:
            continue

        candidates.append({
            "x": center_x,
            "y": center_y,
            "radius": radius,
            "area": area,
            "score": area * circularity(contour)
        })

    return max(candidates, key=lambda item: item["score"]) if candidates else None


def detect_near_prediction(mask, pred_x, pred_y, expected_radius):
    image_height, image_width = mask.shape[:2]
    search_radius = max(int(expected_radius * 3.5), 90)
    x1 = max(0, int(pred_x - search_radius))
    y1 = max(0, int(pred_y - search_radius))
    x2 = min(image_width, int(pred_x + search_radius))
    y2 = min(image_height, int(pred_y + search_radius))
    roi = mask[y1:y2, x1:x2]

    contours, _ = cv2.findContours(
        roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    expected_area = math.pi * expected_radius * expected_radius
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < TRACK_MIN_AREA or area > expected_area * 1.8:
            continue

        global_contour = contour.copy()
        global_contour[:, :, 0] += x1
        global_contour[:, :, 1] += y1
        moments = cv2.moments(global_contour)
        if moments["m00"] <= 0:
            continue

        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        distance = math.hypot(center_x - pred_x, center_y - pred_y)
        if distance > expected_radius * 2.5:
            continue

        candidates.append({
            "contour": global_contour,
            "area": area,
            "center_x": center_x,
            "center_y": center_y,
            "distance": distance
        })

    if not candidates:
        return None

    anchor = max(
        candidates,
        key=lambda item: item["area"] / (1.0 + item["distance"])
    )
    selected = []
    visible_area = 0.0

    for candidate in candidates:
        distance_to_anchor = math.hypot(
            candidate["center_x"] - anchor["center_x"],
            candidate["center_y"] - anchor["center_y"]
        )
        if distance_to_anchor <= expected_radius * 1.8:
            selected.append(candidate["contour"])
            visible_area += candidate["area"]

    hull = cv2.convexHull(np.vstack(selected))
    (measured_x, measured_y), measured_radius = cv2.minEnclosingCircle(hull)
    if measured_radius > expected_radius * 1.8:
        return None

    visible_ratio = float(np.clip(visible_area / expected_area, 0.0, 1.0))
    if visible_ratio >= 0.60:
        image_weight = 0.90
        status = "BALL"
    elif visible_ratio >= 0.30:
        image_weight = 0.50
        status = "PARTIAL"
    elif visible_ratio >= 0.10:
        image_weight = 0.20
        status = "OCCLUDED"
    else:
        image_weight = 0.10
        status = "HEAVY OCCLUSION"

    return {
        "x": image_weight * measured_x + (1.0 - image_weight) * pred_x,
        "y": image_weight * measured_y + (1.0 - image_weight) * pred_y,
        "radius": measured_radius,
        "visible_ratio": visible_ratio,
        "hull": hull,
        "status": status
    }


def draw_box(frame, center_x, center_y, radius, color, thickness=3):
    half_size = max(int(radius), MIN_BALL_RADIUS)
    height, width = frame.shape[:2]
    x1 = max(0, int(center_x - half_size))
    y1 = max(0, int(center_y - half_size))
    x2 = min(width - 1, int(center_x + half_size))
    y2 = min(height - 1, int(center_y + half_size))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def draw_trajectory(frame, trajectory):
    for index in range(1, len(trajectory)):
        if math.hypot(
                trajectory[index][0] - trajectory[index - 1][0],
                trajectory[index][1] - trajectory[index - 1][1]
        ) < 200:
            cv2.line(
                frame,
                trajectory[index - 1],
                trajectory[index],
                (255, 255, 0),
                2
            )


def detect_largest_region(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_REGION_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 0 or height <= 0:
            continue

        candidates.append({
            "contour": contour,
            "area": area,
            "x": x,
            "y": y,
            "width": width,
            "height": height
        })

    if not candidates:
        return None

    return max(candidates, key=lambda item: item["area"])


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"无法创建输出视频：{OUTPUT_PATH}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        mask = make_mask(frame)
        detection = detect_largest_region(mask)

        if detection is not None:
            x = detection["x"]
            y = detection["y"]
            region_width = detection["width"]
            region_height = detection["height"]

            center_x = x + region_width / 2.0
            center_y = y + region_height / 2.0
            box_width = region_width * BOX_SCALE
            box_height = region_height * BOX_SCALE
            x = max(0, int(center_x - box_width / 2.0))
            y = max(0, int(center_y - box_height / 2.0))
            x2 = min(width - 1, int(center_x + box_width / 2.0))
            y2 = min(height - 1, int(center_y + box_height / 2.0))

            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                3
            )
            cv2.putText(
                frame,
                f"AREA: {detection['area']:.0f}",
                (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Frame: {frame_count}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        writer.write(frame)

        if frame_count % 100 == 0:
            print(f"已处理 {frame_count} 帧")

    cap.release()
    writer.release()
    print(f"处理完成，共 {frame_count} 帧")
    print(f"输出视频：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()