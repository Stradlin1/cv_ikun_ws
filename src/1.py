import cv2
import numpy as np
import math
import json


# ============================================================
# 路径
# ============================================================

VIDEO_PATH = "/home/xhm/ikun_ws/video/input.mp4"
OUTPUT_PATH = "/home/xhm/ikun_ws/video/output.mp4"
PARAMETER_RECORD_PATH = "/home/xhm/ikun_ws/video/trackbar_records.json"


# ============================================================
# 默认参数
# ============================================================

DEFAULT = {
    # HSV
    "H_min": 5,
    "H_max": 30,
    "S_min": 90,
    "S_max": 255,
    "V_min": 70,
    "V_max": 255,

    # RGB
    # 默认全放开，所以开始时不会影响结果
    "R_min": 0,
    "R_max": 255,
    "G_min": 0,
    "G_max": 255,
    "B_min": 0,
    "B_max": 255,

}

# HSV 和 RGB 固定使用交集，形态学参数固定，避免控制窗口过高。
MASK_MODE = 2
OPEN_VALUE = 1
CLOSE_VALUE = 3


# ============================================================
# 篮球跟踪参数
# ============================================================

INIT_MIN_AREA = 250
TRACK_MIN_AREA = 40

MIN_BALL_RADIUS = 15
MAX_BALL_RADIUS = 110

MAX_MISSING_FRAMES = 18
MAX_TRAJECTORY = 80


# ============================================================
# Trackbar
# ============================================================

CONTROL_WINDOW = "Basketball Frame Tuner"
MAX_PREVIEW_WIDTH = 600
MAX_PREVIEW_HEIGHT = 280


def nothing(x):
    pass


def create_trackbars():

    cv2.namedWindow(
        CONTROL_WINDOW,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        CONTROL_WINDOW,
        620,
        400
    )

    # ---------------- HSV ----------------

    cv2.createTrackbar(
        "H_min",
        CONTROL_WINDOW,
        DEFAULT["H_min"],
        179,
        nothing
    )

    cv2.createTrackbar(
        "H_max",
        CONTROL_WINDOW,
        DEFAULT["H_max"],
        179,
        nothing
    )

    cv2.createTrackbar(
        "S_min",
        CONTROL_WINDOW,
        DEFAULT["S_min"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "S_max",
        CONTROL_WINDOW,
        DEFAULT["S_max"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "V_min",
        CONTROL_WINDOW,
        DEFAULT["V_min"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "V_max",
        CONTROL_WINDOW,
        DEFAULT["V_max"],
        255,
        nothing
    )

    # ---------------- RGB ----------------

    cv2.createTrackbar(
        "R_min",
        CONTROL_WINDOW,
        DEFAULT["R_min"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "R_max",
        CONTROL_WINDOW,
        DEFAULT["R_max"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "G_min",
        CONTROL_WINDOW,
        DEFAULT["G_min"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "G_max",
        CONTROL_WINDOW,
        DEFAULT["G_max"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "B_min",
        CONTROL_WINDOW,
        DEFAULT["B_min"],
        255,
        nothing
    )

    cv2.createTrackbar(
        "B_max",
        CONTROL_WINDOW,
        DEFAULT["B_max"],
        255,
        nothing
    )

# ============================================================
# 获取 Trackbar 参数
# ============================================================

def get_trackbar_values():

    values = {}

    for name in DEFAULT.keys():

        values[name] = cv2.getTrackbarPos(
            name,
            CONTROL_WINDOW
        )

    return values


# ============================================================
# 恢复 Trackbar 默认值
# ============================================================

def reset_trackbars():

    for name, value in DEFAULT.items():

        cv2.setTrackbarPos(
            name,
            CONTROL_WINDOW,
            value
        )

    print("Trackbar 已恢复默认值")


# ============================================================
# 打印当前参数
# ============================================================

def print_current_parameters():

    p = get_trackbar_values()

    print()
    print("=" * 60)
    print("当前参数")
    print("=" * 60)

    print(
        f"HSV lower = [{p['H_min']}, {p['S_min']}, {p['V_min']}]"
    )

    print(
        f"HSV upper = [{p['H_max']}, {p['S_max']}, {p['V_max']}]"
    )

    print(
        f"RGB lower = [{p['R_min']}, {p['G_min']}, {p['B_min']}]"
    )

    print(
        f"RGB upper = [{p['R_max']}, {p['G_max']}, {p['B_max']}]"
    )

    print("Mode = HSV & RGB")
    print("Open =", OPEN_VALUE)
    print("Close =", CLOSE_VALUE)

    print("=" * 60)
    print()


# ============================================================
# 根据 Trackbar 生成 Mask
# ============================================================

def create_masks(frame):

    p = get_trackbar_values()

    # ========================================================
    # HSV
    # ========================================================

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    hsv_lower = np.array([
        p["H_min"],
        p["S_min"],
        p["V_min"]
    ], dtype=np.uint8)

    hsv_upper = np.array([
        p["H_max"],
        p["S_max"],
        p["V_max"]
    ], dtype=np.uint8)

    hsv_mask = cv2.inRange(
        hsv,
        hsv_lower,
        hsv_upper
    )

    # ========================================================
    # RGB
    #
    # 注意：
    # OpenCV 内部读取顺序是 BGR
    #
    # 所以这里需要：
    #
    # [B, G, R]
    # ========================================================

    bgr_lower = np.array([
        p["B_min"],
        p["G_min"],
        p["R_min"]
    ], dtype=np.uint8)

    bgr_upper = np.array([
        p["B_max"],
        p["G_max"],
        p["R_max"]
    ], dtype=np.uint8)

    rgb_mask = cv2.inRange(
        frame,
        bgr_lower,
        bgr_upper
    )

    # ========================================================
    # 组合方式
    # ========================================================

    mode = MASK_MODE

    if mode == 0:

        final_mask = hsv_mask.copy()

    elif mode == 1:

        final_mask = rgb_mask.copy()

    elif mode == 2:

        final_mask = cv2.bitwise_and(
            hsv_mask,
            rgb_mask
        )

    else:

        final_mask = cv2.bitwise_or(
            hsv_mask,
            rgb_mask
        )

    # ========================================================
    # 开运算
    # ========================================================

    open_value = OPEN_VALUE

    if open_value > 0:

        kernel_size = (
            open_value * 2 + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size
            )
        )

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_OPEN,
            kernel
        )

    # ========================================================
    # 闭运算
    # ========================================================

    close_value = CLOSE_VALUE

    if close_value > 0:

        kernel_size = (
            close_value * 2 + 1
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (
                kernel_size,
                kernel_size
            )
        )

        final_mask = cv2.morphologyEx(
            final_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

    return (
        hsv_mask,
        rgb_mask,
        final_mask,
        p
    )


# ============================================================
# Kalman Filter
# ============================================================

def create_kalman():

    kalman = cv2.KalmanFilter(
        4,
        2
    )

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

    kalman.processNoiseCov = (
        np.eye(
            4,
            dtype=np.float32
        )
        * 0.05
    )

    kalman.measurementNoiseCov = (
        np.eye(
            2,
            dtype=np.float32
        )
        * 3.0
    )

    kalman.errorCovPost = np.eye(
        4,
        dtype=np.float32
    )

    return kalman


# ============================================================
# 圆度
# ============================================================

def get_circularity(contour):

    area = cv2.contourArea(
        contour
    )

    perimeter = cv2.arcLength(
        contour,
        True
    )

    if perimeter <= 0:
        return 0.0

    return (
        4.0
        * math.pi
        * area
        / (perimeter * perimeter)
    )


def get_ball_box(center_x, center_y, radius, image_shape):

    image_height, image_width = image_shape[:2]

    half_size = max(
        int(radius),
        MIN_BALL_RADIUS
    )

    x1 = max(
        0,
        int(center_x - half_size)
    )

    y1 = max(
        0,
        int(center_y - half_size)
    )

    x2 = min(
        image_width - 1,
        int(center_x + half_size)
    )

    y2 = min(
        image_height - 1,
        int(center_y + half_size)
    )

    return x1, y1, x2, y2


def draw_ball_box(image, center_x, center_y, radius, color, thickness=3):

    x1, y1, x2, y2 = get_ball_box(
        center_x,
        center_y,
        radius,
        image.shape
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    return x1, y1, x2, y2


# ============================================================
# 初次寻找篮球
# ============================================================

def detect_initial_ball(mask):

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < INIT_MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if h <= 0:
            continue

        aspect_ratio = (
            w / float(h)
        )

        if not (
            0.60
            <
            aspect_ratio
            <
            1.40
        ):
            continue

        circularity = get_circularity(
            contour
        )

        if circularity < 0.40:
            continue

        (cx, cy), radius = \
            cv2.minEnclosingCircle(
                contour
            )

        if radius < MIN_BALL_RADIUS:
            continue

        if radius > MAX_BALL_RADIUS:
            continue

        circle_area = (
            math.pi
            * radius
            * radius
        )

        if circle_area <= 0:
            continue

        fill_ratio = (
            area
            / circle_area
        )

        if fill_ratio < 0.30:
            continue

        aspect_score = (
            1.0
            -
            abs(
                aspect_ratio - 1.0
            )
        )

        score = (
            area
            * circularity
            * max(
                aspect_score,
                0.1
            )
            * max(
                fill_ratio,
                0.1
            )
        )

        candidates.append({
            "x": cx,
            "y": cy,
            "radius": radius,
            "area": area,
            "score": score,
            "contour": contour
        })

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: item["score"]
    )


# ============================================================
# 跟踪状态寻找篮球
# ============================================================

def detect_tracked_ball(
        mask,
        pred_x,
        pred_y,
        expected_radius):

    image_h, image_w = \
        mask.shape[:2]

    search_radius = max(
        int(
            expected_radius
            * 3.5
        ),
        90
    )

    x1 = max(
        0,
        int(
            pred_x
            - search_radius
        )
    )

    y1 = max(
        0,
        int(
            pred_y
            - search_radius
        )
    )

    x2 = min(
        image_w,
        int(
            pred_x
            + search_radius
        )
    )

    y2 = min(
        image_h,
        int(
            pred_y
            + search_radius
        )
    )

    if x2 <= x1 or y2 <= y1:
        return None

    roi = mask[
        y1:y2,
        x1:x2
    ]

    contours, _ = cv2.findContours(
        roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    selected = []

    total_visible_area = 0.0

    expected_ball_area = (
        math.pi
        * expected_radius
        * expected_radius
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < TRACK_MIN_AREA:
            continue

        # 过大的区域通常不是篮球
        if (
            area
            >
            expected_ball_area
            * 1.8
        ):
            continue

        global_contour = \
            contour.copy()

        global_contour[
            :, :, 0
        ] += x1

        global_contour[
            :, :, 1
        ] += y1

        M = cv2.moments(
            global_contour
        )

        if M["m00"] <= 0:
            continue

        cx = (
            M["m10"]
            / M["m00"]
        )

        cy = (
            M["m01"]
            / M["m00"]
        )

        distance = math.hypot(
            cx - pred_x,
            cy - pred_y
        )

        if (
            distance
            >
            expected_radius
            * 2.5
        ):
            continue

        selected.append(
            global_contour
        )

        total_visible_area += area

    if not selected:
        return None

    # ========================================================
    # 合并被人体切开的多个橙色部分
    # ========================================================

    all_points = np.vstack(
        selected
    )

    hull = cv2.convexHull(
        all_points
    )

    (
        measured_center,
        measured_radius
    ) = cv2.minEnclosingCircle(
        hull
    )

    measured_x = \
        measured_center[0]

    measured_y = \
        measured_center[1]

    # 异常过大
    if (
        measured_radius
        >
        expected_radius
        * 1.8
    ):
        return None

    visible_ratio = (
        total_visible_area
        /
        expected_ball_area
    )

    visible_ratio = float(
        np.clip(
            visible_ratio,
            0.0,
            1.0
        )
    )

    # ========================================================
    # 遮挡程度
    # ========================================================

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

    final_x = (
        image_weight
        * measured_x
        +
        (
            1.0
            - image_weight
        )
        * pred_x
    )

    final_y = (
        image_weight
        * measured_y
        +
        (
            1.0
            - image_weight
        )
        * pred_y
    )

    return {
        "x": final_x,
        "y": final_y,

        "measured_radius":
            measured_radius,

        "visible_ratio":
            visible_ratio,

        "status":
            status,

        "hull":
            hull
    }


# ============================================================
# 暂停状态下的预览
#
# 注意：
# 不修改 Kalman
# 只用于拖 Trackbar 看当前参数效果
# ============================================================

def draw_tuning_preview(
        frame,
        mask,
        initialized,
        last_x,
        last_y,
        expected_radius):

    preview = frame.copy()

    if not initialized:

        detection = \
            detect_initial_ball(
                mask
            )

        if detection is not None:

            draw_ball_box(
                preview,
                detection["x"],
                detection["y"],
                detection["radius"],
                (255, 0, 255),
                3
            )

            cv2.putText(
                preview,
                "TUNING CANDIDATE",
                (
                    int(
                        detection["x"]
                    ),
                    int(
                        detection["y"]
                        -
                        detection[
                            "radius"
                        ]
                        - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

    elif (
        last_x is not None
        and
        last_y is not None
        and
        expected_radius is not None
    ):

        detection = \
            detect_tracked_ball(
                mask,
                last_x,
                last_y,
                expected_radius
            )

        if detection is not None:

            draw_ball_box(
                preview,
                detection["x"],
                detection["y"],
                expected_radius,
                (255, 0, 255),
                3
            )

            cv2.drawContours(
                preview,
                [
                    detection["hull"]
                ],
                -1,
                (255, 255, 0),
                2
            )

    cv2.putText(
        preview,
        "PAUSED - TRACKBAR TUNING",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 0, 255),
        2
    )

    return preview


# ============================================================
# 单帧参数调试工具
# ============================================================

def load_frames():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{VIDEO_PATH}")

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames


def load_parameter_records():

    try:
        with open(PARAMETER_RECORD_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)

        if isinstance(data, dict):
            return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return {
        "records": [],
        "ranges": {}
    }


def save_parameter_record(records, frame_index, params):

    records.setdefault("records", [])
    records.setdefault("ranges", {})

    records["records"].append({
        "frame": frame_index + 1,
        "parameters": params
    })

    for name, value in params.items():
        values = [
            item["parameters"][name]
            for item in records["records"]
            if name in item.get("parameters", {})
        ]

        if values:
            records["ranges"][name] = {
                "min": min(values),
                "max": max(values)
            }

    with open(PARAMETER_RECORD_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def resize_for_display(image, target_height):

    image_height, image_width = image.shape[:2]

    if image_height == target_height:
        return image

    target_width = max(
        1,
        int(image_width * target_height / image_height)
    )

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )


def make_preview(frame, mask, frame_index, total_frames, params):

    original = frame.copy()
    detection = detect_initial_ball(mask)

    if detection is not None:
        draw_ball_box(
            original,
            detection["x"],
            detection["y"],
            detection["radius"],
            (0, 255, 0),
            3
        )

        cv2.putText(
            original,
            "BALL CANDIDATE",
            (30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            original,
            "NO BALL CANDIDATE",
            (30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.putText(
        original,
        f"Frame: {frame_index + 1}/{total_frames}",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    target_height = max(original.shape[0], mask_bgr.shape[0])
    original = resize_for_display(original, target_height)
    mask_bgr = resize_for_display(mask_bgr, target_height)

    separator = np.zeros((target_height, 8, 3), dtype=np.uint8)
    preview = np.hstack((original, separator, mask_bgr))

    preview_height, preview_width = preview.shape[:2]
    scale = min(
        1.0,
        MAX_PREVIEW_WIDTH / float(preview_width),
        MAX_PREVIEW_HEIGHT / float(preview_height)
    )

    if scale < 1.0:
        preview = cv2.resize(
            preview,
            (
                int(preview_width * scale),
                int(preview_height * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    cv2.putText(
        preview,
        "ORIGINAL",
        (15, target_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        preview,
        "BINARY MASK",
        (original.shape[1] + 23, target_height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return preview


def run_frame_tuner():

    frames = load_frames()

    if not frames:
        raise RuntimeError("视频中没有可读取的帧")

    create_trackbars()
    records = load_parameter_records()
    frame_index = 0

    print("单帧调参模式：A=上一帧，D=下一帧，S=保存当前参数，R=恢复默认，Q 或 ESC=退出")
    print(f"参数记录文件：{PARAMETER_RECORD_PATH}")

    while True:
        hsv_mask, rgb_mask, mask, params = create_masks(frames[frame_index])
        preview = make_preview(
            frames[frame_index],
            mask,
            frame_index,
            len(frames),
            params
        )

        cv2.imshow(CONTROL_WINDOW, preview)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord("a"):
            frame_index = max(0, frame_index - 1)

        elif key == ord("d"):
            frame_index = min(len(frames) - 1, frame_index + 1)

        elif key == ord("s"):
            save_parameter_record(records, frame_index, params)
            print(
                f"已记录第 {frame_index + 1} 帧参数，"
                f"累计 {len(records['records'])} 条记录"
            )

        elif key == ord("r"):
            reset_trackbars()

    cv2.destroyAllWindows()
    print(f"参数记录已保存到：{PARAMETER_RECORD_PATH}")


if __name__ == "__main__":

    run_frame_tuner()


"""
# ============================================================
# 旧的整段视频跟踪入口（暂不执行）
# ============================================================

while True:

    # ========================================================
    # 没暂停时读取新帧
    # ========================================================

    new_frame = False

    if not paused or current_frame is None:

        ret, frame = cap.read()

        if not ret:
            break

        current_frame = frame.copy()

        frame_id += 1

        new_frame = True


    # 当前原图
    frame = current_frame.copy()


    # ========================================================
    # 每一次循环都重新读取 Trackbar
    #
    # 所以即使暂停，
    # 拖 Trackbar 也会实时刷新 Mask
    # ========================================================

    (
        hsv_mask,
        rgb_mask,
        mask,
        params
    ) = create_masks(
        current_frame
    )


    # ========================================================
    # 暂停状态
    #
    # 不修改 Kalman
    # 不写视频
    #
    # 单纯用于调参数
    # ========================================================

    if paused:

        display = draw_tuning_preview(
            current_frame,
            mask,
            initialized,
            last_track_x,
            last_track_y,
            expected_radius
        )


    # ========================================================
    # 正常运行
    # ========================================================

    else:

        display = frame.copy()


        # ====================================================
        # 没锁定球
        # ====================================================

        if not initialized:

            detection = \
                detect_initial_ball(
                    mask
                )

            if detection is not None:

                ball_x = \
                    detection["x"]

                ball_y = \
                    detection["y"]

                expected_radius = \
                    detection["radius"]

                kalman = \
                    create_kalman()

                kalman.statePost = \
                    np.array([
                        [ball_x],
                        [ball_y],
                        [0],
                        [0]
                    ], dtype=np.float32)

                kalman.statePre = \
                    kalman.statePost.copy()

                initialized = True

                missing_frames = 0

                trajectory.clear()

                last_track_x = ball_x
                last_track_y = ball_y

                draw_ball_box(
                    display,
                    ball_x,
                    ball_y,
                    expected_radius,
                    (0, 255, 0),
                    3
                )

                cv2.putText(
                    display,
                    "INITIALIZED",
                    (
                        int(ball_x),
                        int(
                            ball_y
                            -
                            expected_radius
                            - 10
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


        # ====================================================
        # 已锁定球
        # ====================================================

        else:

            prediction = \
                kalman.predict()

            pred_x = float(
                prediction[
                    0,
                    0
                ]
            )

            pred_y = float(
                prediction[
                    1,
                    0
                ]
            )


            detection = \
                detect_tracked_ball(
                    mask,
                    pred_x,
                    pred_y,
                    expected_radius
                )


            # =================================================
            # 检测成功
            # =================================================

            if detection is not None:

                detect_x = \
                    detection["x"]

                detect_y = \
                    detection["y"]

                visible_ratio = \
                    detection[
                        "visible_ratio"
                    ]

                measured_radius = \
                    detection[
                        "measured_radius"
                    ]

                status = \
                    detection["status"]

                hull = \
                    detection["hull"]

                measurement = np.array([
                    [
                        np.float32(
                            detect_x
                        )
                    ],
                    [
                        np.float32(
                            detect_y
                        )
                    ]
                ])

                corrected = \
                    kalman.correct(
                        measurement
                    )

                track_x = float(
                    corrected[
                        0,
                        0
                    ]
                )

                track_y = float(
                    corrected[
                        1,
                        0
                    ]
                )

                last_track_x = track_x
                last_track_y = track_y

                missing_frames = 0


                # 更新半径
                if (
                    visible_ratio
                    >= 0.60
                ):

                    if (
                        expected_radius
                        * 0.70
                        <
                        measured_radius
                        <
                        expected_radius
                        * 1.35
                    ):

                        expected_radius = (
                            expected_radius
                            * 0.90
                            +
                            measured_radius
                            * 0.10
                        )


                trajectory.append(
                    (
                        int(track_x),
                        int(track_y)
                    )
                )

                if (
                    len(trajectory)
                    >
                    MAX_TRAJECTORY
                ):

                    trajectory.pop(0)


                if status == "BALL":

                    color = (
                        0,
                        255,
                        0
                    )

                elif status == "PARTIAL":

                    color = (
                        0,
                        255,
                        255
                    )

                else:

                    color = (
                        0,
                        165,
                        255
                    )


                draw_ball_box(
                    display,
                    track_x,
                    track_y,
                    expected_radius,
                    color,
                    3
                )


                cv2.drawContours(
                    display,
                    [hull],
                    -1,
                    (
                        255,
                        0,
                        0
                    ),
                    2
                )


                cv2.putText(
                    display,
                    status,
                    (
                        int(
                            track_x
                            -
                            expected_radius
                        ),
                        int(
                            track_y
                            -
                            expected_radius
                            - 12
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )


                cv2.putText(
                    display,
                    f"Visible: {visible_ratio:.2f}",
                    (
                        30,
                        70
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )


            # =================================================
            # 完全没检测到
            # =================================================

            else:

                missing_frames += 1

                track_x = pred_x
                track_y = pred_y

                last_track_x = track_x
                last_track_y = track_y


                trajectory.append(
                    (
                        int(track_x),
                        int(track_y)
                    )
                )

                if (
                    len(trajectory)
                    >
                    MAX_TRAJECTORY
                ):

                    trajectory.pop(0)


                draw_ball_box(
                    display,
                    track_x,
                    track_y,
                    expected_radius,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )


                cv2.putText(
                    display,
                    "PREDICTION",
                    (
                        int(
                            track_x
                            -
                            expected_radius
                        ),
                        int(
                            track_y
                            -
                            expected_radius
                            - 12
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (
                        0,
                        0,
                        255
                    ),
                    2
                )


                if (
                    missing_frames
                    >=
                    MAX_MISSING_FRAMES
                ):

                    print(
                        f"[Frame {frame_id}] "
                        "篮球丢失，重新搜索"
                    )

                    initialized = False

                    expected_radius = None

                    missing_frames = 0

                    trajectory.clear()

                    last_track_x = None
                    last_track_y = None


            # =================================================
            # 轨迹
            # =================================================

            for i in range(
                1,
                len(trajectory)
            ):

                p1 = trajectory[
                    i - 1
                ]

                p2 = trajectory[
                    i
                ]

                d = math.hypot(
                    p2[0] - p1[0],
                    p2[1] - p1[1]
                )

                if d < 200:

                    cv2.line(
                        display,
                        p1,
                        p2,
                        (
                            255,
                            255,
                            0
                        ),
                        2
                    )


        # ====================================================
        # 只在真正读取新帧时写输出
        # ====================================================

        if new_frame:
            writer.write(
                display
            )


    # ========================================================
    # 当前 Mask 模式显示
    # ========================================================

    mode_names = [
        "HSV",
        "RGB",
        "HSV & RGB",
        "HSV | RGB"
    ]

    mode_name = \
        mode_names[
            params["Mode"]
        ]


    cv2.putText(
        display,
        f"Frame: {frame_id}",
        (
            30,
            30
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2
    )


    cv2.putText(
        display,
        f"Mask: {mode_name}",
        (
            30,
            105
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (
            255,
            255,
            255
        ),
        2
    )


    # ========================================================
    # 显示
    # ========================================================

    cv2.imshow(
        "Basketball Tracker",
        display
    )

    cv2.imshow(
        "HSV Mask",
        hsv_mask
    )

    cv2.imshow(
        "RGB Mask",
        rgb_mask
    )

    cv2.imshow(
        "Final Mask",
        mask
    )


    key = cv2.waitKey(
        10
    ) & 0xFF


    # ========================================================
    # 键盘操作
    # ========================================================

    if (
        key == 27
        or
        key == ord("q")
    ):
        break


    # P 或空格：
    # 暂停 / 继续
    if (
        key == ord("p")
        or
        key == 32
    ):

        paused = not paused

        if paused:

            print()
            print("========== 已暂停 ==========")
            print("现在可以拖动 Trackbar")
            print("Mask 会实时刷新")
            print("============================")

        else:

            print("继续播放")


    # S：
    # 打印当前参数
    if key == ord("s"):

        print_current_parameters()


    # R：
    # 参数恢复默认
    if key == ord("r"):

        reset_trackbars()


# ============================================================
# 结束
# ============================================================

cap.release()
writer.release()

cv2.destroyAllWindows()

print()
print("处理结束")
print("最终参数：")

print_current_parameters()

print(
    "输出视频：",
    OUTPUT_PATH
)
"""