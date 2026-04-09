"""
Air Writing System
------------------
Draw in the air using your index finger!

Controls:
  - Index finger UP only  → Draw mode
  - Index + Middle UP     → Move / hover (no drawing)
  - Thumb + Index pinch   → Erase mode
  - Press 'c'             → Clear canvas
  - Press 'q'             → Quit

Color palette (shown in top-right):
  Click the colored circles with your fingertip to change color.
"""

import cv2
import mediapipe as mp
import numpy as np
import time

# ── Config ──────────────────────────────────────────────────────────────────
BRUSH_SIZE     = 8
ERASER_SIZE    = 40
SMOOTHING      = 0.4     # 0 = no smoothing, 1 = fully frozen
EFFECT_FADE    = False    # paint fades over time (set False to keep forever)
FADE_RATE      = 2       # opacity lost per frame when fading

# Color palette (BGR)
COLORS = {
    "white":  (255, 255, 255),
    "cyan":   (255, 200, 0),
    "green":  (50,  220, 50),
    "red":    (50,   50, 230),
    "yellow": (0,   230, 230),
    "purple": (200,  50, 200),
}
COLOR_LIST = list(COLORS.values())

# ── MediaPipe setup ─────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def finger_is_up(landmarks, tip_id, pip_id):
    """Returns True if the finger tip is above its PIP joint (finger extended)."""
    return landmarks[tip_id].y < landmarks[pip_id].y


def get_gesture(lm):
    """
    Returns: 'draw', 'hover', or 'erase'
    - draw:  only index finger up
    - hover: index + middle up
    - erase: thumb close to index tip (pinch)
    """
    index_up  = finger_is_up(lm, 8, 6)
    middle_up = finger_is_up(lm, 12, 10)
    ring_up   = finger_is_up(lm, 16, 14)

    # Pinch detection: thumb tip (4) near index tip (8)
    thumb_tip = np.array([lm[4].x, lm[4].y])
    index_tip = np.array([lm[8].x, lm[8].y])
    pinch_dist = np.linalg.norm(thumb_tip - index_tip)

    if pinch_dist < 0.06:
        return "erase"
    if index_up and not middle_up and not ring_up:
        return "draw"
    return "hover"


def draw_color_palette(frame, selected_idx, palette_x, palette_y):
    """Draw color circles in the top-right corner."""
    r = 18
    for i, color in enumerate(COLOR_LIST):
        cx = palette_x + i * (r * 2 + 8)
        cv2.circle(frame, (cx, palette_y), r, color, -1)
        cv2.circle(frame, (cx, palette_y), r, (200, 200, 200), 1)
        if i == selected_idx:
            cv2.circle(frame, (cx, palette_y), r + 4, (255, 255, 255), 2)
    return [(palette_x + i * (r * 2 + 8), palette_y) for i in range(len(COLOR_LIST))]


def check_palette_tap(fingertip, palette_centers, radius=24):
    """Returns color index if fingertip is on a palette circle, else -1."""
    for i, (cx, cy) in enumerate(palette_centers):
        dist = np.linalg.norm(np.array(fingertip) - np.array([cx, cy]))
        if dist < radius:
            return i
    return -1


def draw_plasma_effect(frame, x, y, color):
    """Simple electric-glow effect around fingertip."""
    for r in [30, 20, 10]:
        alpha = 0.05 + (30 - r) * 0.012
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), r, color, 1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    ret, frame = cap.read()
    h, w = frame.shape[:2]

    # Drawing canvas (BGRA so we can fade alpha)
    canvas = np.zeros((h, w, 4), dtype=np.uint8)

    prev_x, prev_y = None, None
    smooth_x, smooth_y = 0, 0
    selected_color_idx = 0
    current_color = COLOR_LIST[selected_color_idx]
    last_gesture = "hover"

    # Palette position (top-right)
    pal_x = w - len(COLOR_LIST) * 52 + 18
    pal_y = 40

    print("Air Writing started! Press 'q' to quit, 'c' to clear.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "hover"
        fingertip = None

        if result.multi_hand_landmarks:
            for hand_lm in result.multi_hand_landmarks:
                lm = hand_lm.landmark
                gesture = get_gesture(lm)

                # Raw fingertip coords
                raw_x = int(lm[8].x * w)
                raw_y = int(lm[8].y * h)

                # Exponential smoothing
                smooth_x = int(smooth_x * SMOOTHING + raw_x * (1 - SMOOTHING))
                smooth_y = int(smooth_y * SMOOTHING + raw_y * (1 - SMOOTHING))
                fingertip = (smooth_x, smooth_y)

                # Check palette selection
                palette_centers = draw_color_palette(frame, selected_color_idx, pal_x, pal_y)
                tapped = check_palette_tap(fingertip, palette_centers)
                if tapped >= 0 and gesture == "draw":
                    selected_color_idx = tapped
                    current_color = COLOR_LIST[selected_color_idx]

                # Draw / erase on canvas
                if gesture == "draw" and prev_x is not None:
                    color_bgra = (*current_color, 255)
                    cv2.line(canvas, (prev_x, prev_y), fingertip,
                             color_bgra, BRUSH_SIZE, cv2.LINE_AA)

                elif gesture == "erase" and fingertip:
                    cv2.circle(canvas, fingertip, ERASER_SIZE, (0, 0, 0, 0), -1)

                # Plasma effect around fingertip
                if gesture == "draw":
                    draw_plasma_effect(frame, smooth_x, smooth_y, current_color)

                # Draw hand skeleton
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=1, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 150, 200), thickness=1),
                )

                prev_x, prev_y = (smooth_x, smooth_y) if gesture == "draw" else (None, None)
                last_gesture = gesture
        else:
            prev_x, prev_y = None, None

        # Optional fade effect
        if EFFECT_FADE:
            alpha_channel = canvas[:, :, 3].astype(np.int16)
            alpha_channel = np.clip(alpha_channel - FADE_RATE, 0, 255).astype(np.uint8)
            canvas[:, :, 3] = alpha_channel

        # Blend canvas onto frame
        alpha = canvas[:, :, 3:4] / 255.0
        canvas_bgr = canvas[:, :, :3].astype(np.float32)
        frame_f    = frame.astype(np.float32)
        frame_f    = frame_f * (1 - alpha) + canvas_bgr * alpha
        frame      = np.clip(frame_f, 0, 255).astype(np.uint8)

        # Draw palette on top
        draw_color_palette(frame, selected_color_idx, pal_x, pal_y)

        # HUD
        gesture_colors = {"draw": (50, 220, 50), "erase": (50, 50, 220), "hover": (200, 200, 200)}
        hud_color = gesture_colors.get(gesture, (200, 200, 200))
        cv2.putText(frame, f"Mode: {gesture.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, hud_color, 2)
        cv2.putText(frame, "c=clear  q=quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        # Eraser circle indicator
        if gesture == "erase" and fingertip:
            cv2.circle(frame, fingertip, ERASER_SIZE, (50, 50, 220), 2)

        cv2.imshow("Air Writing PRO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            canvas[:] = 0

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
