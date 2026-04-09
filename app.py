from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# ── MediaPipe ────────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# ── State ─────────────────────────────────────────────────────────────────────
BRUSH_SIZE   = 8
ERASER_SIZE  = 40
SMOOTHING    = 0.4

COLORS = [
    (255, 255, 255),  # white
    (255, 200,   0),  # cyan
    ( 50, 220,  50),  # green
    ( 50,  50, 230),  # red
    (  0, 230, 230),  # yellow
    (200,  50, 200),  # purple
]

canvas       = None
prev_x       = None
prev_y       = None
smooth_x     = 0
smooth_y     = 0
selected_idx = 0
current_color = COLORS[0]


def finger_is_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def get_gesture(lm):
    index_up  = finger_is_up(lm, 8, 6)
    middle_up = finger_is_up(lm, 12, 10)
    ring_up   = finger_is_up(lm, 16, 14)
    thumb_tip  = np.array([lm[4].x, lm[4].y])
    index_tip  = np.array([lm[8].x, lm[8].y])
    if np.linalg.norm(thumb_tip - index_tip) < 0.06:
        return "erase"
    if index_up and not middle_up and not ring_up:
        return "draw"
    return "hover"


def draw_palette(frame, selected, pal_x, pal_y):
    centers = []
    r = 18
    for i, color in enumerate(COLORS):
        cx = pal_x + i * (r * 2 + 8)
        cv2.circle(frame, (cx, pal_y), r, color, -1)
        cv2.circle(frame, (cx, pal_y), r, (200, 200, 200), 1)
        if i == selected:
            cv2.circle(frame, (cx, pal_y), r + 4, (255, 255, 255), 2)
        centers.append((cx, pal_y))
    return centers


def check_palette_tap(fingertip, centers, radius=24):
    for i, (cx, cy) in enumerate(centers):
        if np.linalg.norm(np.array(fingertip) - np.array([cx, cy])) < radius:
            return i
    return -1


def generate_frames():
    global canvas, prev_x, prev_y, smooth_x, smooth_y
    global selected_idx, current_color

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    pal_x = w - len(COLORS) * 52 + 18
    pal_y = 40

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture   = "hover"
        fingertip = None

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                lm      = hand_lm.landmark
                gesture = get_gesture(lm)

                raw_x = int(lm[8].x * w)
                raw_y = int(lm[8].y * h)
                smooth_x = int(smooth_x * SMOOTHING + raw_x * (1 - SMOOTHING))
                smooth_y = int(smooth_y * SMOOTHING + raw_y * (1 - SMOOTHING))
                fingertip = (smooth_x, smooth_y)

                palette_centers = draw_palette(frame, selected_idx, pal_x, pal_y)
                tapped = check_palette_tap(fingertip, palette_centers)
                if tapped >= 0 and gesture == "draw":
                    selected_idx  = tapped
                    current_color = COLORS[selected_idx]

                if gesture == "draw" and prev_x is not None:
                    cv2.line(canvas, (prev_x, prev_y), fingertip,
                             (*current_color, 255), BRUSH_SIZE, cv2.LINE_AA)
                elif gesture == "erase" and fingertip:
                    cv2.circle(canvas, fingertip, ERASER_SIZE, (0, 0, 0, 0), -1)

                # Glow effect
                if gesture == "draw":
                    for r in [30, 20, 10]:
                        overlay = frame.copy()
                        cv2.circle(overlay, fingertip, r, current_color, 1)
                        cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 150, 200), thickness=1),
                )

                prev_x, prev_y = (smooth_x, smooth_y) if gesture == "draw" else (None, None)
        else:
            prev_x, prev_y = None, None

        # Blend canvas
        alpha      = canvas[:, :, 3:4] / 255.0
        canvas_bgr = canvas[:, :, :3].astype(np.float32)
        frame_f    = frame.astype(np.float32)
        frame_f    = frame_f * (1 - alpha) + canvas_bgr * alpha
        frame      = np.clip(frame_f, 0, 255).astype(np.uint8)

        draw_palette(frame, selected_idx, pal_x, pal_y)

        # HUD
        gesture_colors = {"draw": (50, 220, 50), "erase": (50, 50, 220), "hover": (200, 200, 200)}
        cv2.putText(frame, f"Mode: {gesture.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, gesture_colors.get(gesture, (200,200,200)), 2)
        if gesture == "erase" and fingertip:
            cv2.circle(frame, fingertip, ERASER_SIZE, (50, 50, 220), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/clear')
def clear():
    global canvas
    if canvas is not None:
        canvas[:] = 0
    return ('', 204)


if __name__ == '__main__':
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, threaded=True)