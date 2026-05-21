import cv2
import numpy as np

# ══════════════════════════════════════════════════════
#  ROI  —  what portion of the frame to analyse
#  Tweak these 4 values (0.0–1.0) to shrink/expand the box
# ══════════════════════════════════════════════════════
ROI_LEFT   = 0.15   # cut 15 % from left edge
ROI_RIGHT  = 0.85   # cut 15 % from right edge
ROI_TOP    = 0.10   # cut 10 % from top
ROI_BOTTOM = 0.85   # cut 15 % from bottom

# ══════════════════════════════════════════════════════
#  DEFAULTS  (trackbars override these live)
# ══════════════════════════════════════════════════════
DEFAULT_MIN_AREA = 1800   # px²  — ignore tiny blobs
DEFAULT_S_MIN    = 160    # aggressive: skin ~ 30-140
DEFAULT_V_MIN    = 110    # rejects dark shadows

# ══════════════════════════════════════════════════════
#  FIXED HSV UPPER BOUNDS  (only lower bounds are dynamic)
# ══════════════════════════════════════════════════════
RED_UPPER1  = np.array([ 10, 255, 255])
RED_UPPER2  = np.array([180, 255, 255])
GREEN_UPPER = np.array([ 85, 255, 255])

# Morphology kernels
KERNEL_OPEN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
KERNEL_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))


# ──────────────────────────────────────────────────────
def create_trackbars(win: str) -> None:
    cv2.namedWindow(win)
    cv2.createTrackbar("MIN_AREA", win, DEFAULT_MIN_AREA, 25000, lambda x: None)
    cv2.createTrackbar("S_MIN",    win, DEFAULT_S_MIN,      255, lambda x: None)
    cv2.createTrackbar("V_MIN",    win, DEFAULT_V_MIN,      255, lambda x: None)


def read_trackbars(win: str) -> tuple[int, int, int]:
    return (
        max(cv2.getTrackbarPos("MIN_AREA", win), 200),
        cv2.getTrackbarPos("S_MIN", win),
        cv2.getTrackbarPos("V_MIN", win),
    )


# ──────────────────────────────────────────────────────
def get_roi_coords(h: int, w: int) -> tuple[int, int, int, int]:
    """Return (x1, y1, x2, y2) pixel coords of the analysis rectangle."""
    x1 = int(w * ROI_LEFT)
    x2 = int(w * ROI_RIGHT)
    y1 = int(h * ROI_TOP)
    y2 = int(h * ROI_BOTTOM)
    return x1, y1, x2, y2


def draw_roi_and_zones(frame: np.ndarray,
                       x1: int, y1: int,
                       x2: int, y2: int) -> None:
    """
    Draw the ROI border and THREE HORIZONTAL zones inside it.
    Zones:  TOP / MIDDLE / BOTTOM  (stacked vertically).
    """
    roi_h = y2 - y1

    # Horizontal dividers at 1/3 and 2/3 of ROI height
    div1 = y1 + roi_h // 3
    div2 = y1 + 2 * roi_h // 3

    # Zone fill (subtle tint so it's visible but not distracting)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1),  (x2, div1), (40, 40, 40),  -1)
    cv2.rectangle(overlay, (x1, div1),(x2, div2), (30, 30, 30),  -1)
    cv2.rectangle(overlay, (x1, div2),(x2, y2),   (40, 40, 40),  -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)

    # Horizontal divider lines
    cv2.line(frame, (x1, div1), (x2, div1), (255, 255, 255), 2)
    cv2.line(frame, (x1, div2), (x2, div2), (255, 255, 255), 2)

    # Zone labels (right-aligned inside ROI)
    label_x = x2 - 90
    for label, ly in [("TOP",    y1 + 18),
                       ("MIDDLE", div1 + 18),
                       ("BOTTOM", div2 + 18)]:
        cv2.putText(frame, label, (label_x, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    # ROI bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)


# ──────────────────────────────────────────────────────
def build_masks(hsv_roi: np.ndarray,
                s_min: int, v_min: int) -> tuple[np.ndarray, np.ndarray]:
    """Build morphologically cleaned masks for red & green."""
    # Dynamic lower bounds (S and V from trackbars)
    red_lo1  = np.array([  0, s_min, v_min])
    red_lo2  = np.array([160, s_min, v_min])
    green_lo = np.array([ 40, s_min, v_min])

    mask_r = (cv2.inRange(hsv_roi, red_lo1, RED_UPPER1) |
              cv2.inRange(hsv_roi, red_lo2, RED_UPPER2))
    mask_g = cv2.inRange(hsv_roi, green_lo, GREEN_UPPER)

    for mask in (mask_r, mask_g):
        cv2.morphologyEx(mask, cv2.MORPH_OPEN,  KERNEL_OPEN,  dst=mask)
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE, dst=mask)

    return mask_r, mask_g


def zone_label_h(cy_roi: int, roi_h: int) -> str:
    """Horizontal zone from centroid y inside ROI."""
    if   cy_roi < roi_h // 3:      return "TOP"
    elif cy_roi < 2 * roi_h // 3:  return "MIDDLE"
    else:                           return "BOTTOM"


def process_contours(frame:      np.ndarray,
                     mask:       np.ndarray,
                     color_name: str,
                     bbox_color: tuple,
                     min_area:   int,
                     x1: int,    y1: int,
                     roi_h: int) -> list[str]:
    """
    Detect blobs in mask, draw bboxes on frame (offset by ROI origin),
    return list of horizontal zones hit.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    zones: list[str] = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        rx, ry, rw, rh = cv2.boundingRect(cnt)

        # Map ROI-local coords back to full-frame coords
        fx, fy = rx + x1, ry + y1

        cy_roi = ry + rh // 2
        zone   = zone_label_h(cy_roi, roi_h)
        zones.append(zone)

        cv2.rectangle(frame, (fx, fy), (fx + rw, fy + rh), bbox_color, 2)
        cv2.putText(frame, f"{color_name} [{zone}]",
                    (fx, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, bbox_color, 2)

    return zones


# ──────────────────────────────────────────────────────
def overlay_command(frame:       np.ndarray,
                    red_zones:   list[str],
                    green_zones: list[str]) -> None:
    h, w = frame.shape[:2]

    if   green_zones and not red_zones:   cmd, color = "TURN RIGHT",    (0, 210,   0)
    elif red_zones   and not green_zones: cmd, color = "TURN LEFT",     (0,   0, 210)
    elif red_zones   and green_zones:     cmd, color = "BOTH VISIBLE",  (0, 200, 200)
    else:                                 cmd, color = "GO STRAIGHT",   (220, 220, 220)

    (tw, _), _ = cv2.getTextSize(cmd, cv2.FONT_HERSHEY_DUPLEX, 1.4, 3)
    tx = (w - tw) // 2
    ty = h - 25
    cv2.putText(frame, cmd, (tx + 2, ty + 2),   # shadow
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 4)
    cv2.putText(frame, cmd, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main() -> None:
    TUNER = "WRO Tuner"
    create_trackbars(TUNER)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam (index 0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("[WRO Vision v2] Running — press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed — exiting.")
            break

        min_area, s_min, v_min = read_trackbars(TUNER)
        h, w = frame.shape[:2]

        # ── ROI coords ───────────────────────
        x1, y1, x2, y2 = get_roi_coords(h, w)
        roi_h = y2 - y1

        # ── Crop & convert only the ROI ──────
        roi     = frame[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        hsv_roi = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # ── Masks ────────────────────────────
        mask_r, mask_g = build_masks(hsv_roi, s_min, v_min)

        # ── Draw ROI box + zones on frame ────
        draw_roi_and_zones(frame, x1, y1, x2, y2)

        # ── Detect & annotate ────────────────
        red_zones   = process_contours(frame, mask_r, "RED",   (30,  30, 220),
                                       min_area, x1, y1, roi_h)
        green_zones = process_contours(frame, mask_g, "GREEN", (30, 200,  30),
                                       min_area, x1, y1, roi_h)

        # ── Command banner ───────────────────
        overlay_command(frame, red_zones, green_zones)

        # ── Debug HUD ────────────────────────
        cv2.putText(frame,
                    f"MIN_AREA={min_area}  S_MIN={s_min}  V_MIN={v_min}",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (160, 160, 160), 1)

        cv2.imshow("WRO Vision v2", frame)

        # Uncomment to debug masks:
        # cv2.imshow("RED  mask", mask_r)
        # cv2.imshow("GREEN mask", mask_g)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if _name_ == "_main_":
    main()