import cv2
import numpy as np
import time
import RPi.GPIO as GPIO

# =========================
# GPIO AYARLARI
# =========================

TRIG = 5
ECHO = 6

SERVO_PIN = 18

ENA = 12
IN1 = 17
IN2 = 27

GPIO.setmode(GPIO.BCM)

GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.setup(SERVO_PIN, GPIO.OUT)

GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)

servo_pwm = GPIO.PWM(SERVO_PIN, 50)
servo_pwm.start(0)

motor_pwm = GPIO.PWM(ENA, 1000)
motor_pwm.start(0)

# =========================
# AYARLAR
# =========================

CENTER_ANGLE = 90
MIN_ANGLE = 60
MAX_ANGLE = 120

KP = 0.15

SAFE_DISTANCE = 15

MOTOR_SPEED = 65

ROI_WIDTH = 300
ROI_HEIGHT = 150

OFFSET = 60

# =========================
# SERVO
# =========================

def set_servo(angle):
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))

    duty = 2 + angle / 18

    servo_pwm.ChangeDutyCycle(duty)
    time.sleep(0.02)
    servo_pwm.ChangeDutyCycle(0)

# =========================
# MOTOR
# =========================

def motor_forward(speed):
    GPIO.output(IN1, True)
    GPIO.output(IN2, False)
    motor_pwm.ChangeDutyCycle(speed)

def motor_stop():
    motor_pwm.ChangeDutyCycle(0)

# =========================
# HC-SR04
# =========================

def get_distance():

    GPIO.output(TRIG, False)
    time.sleep(0.0002)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time()

    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()

        if time.time() - timeout > 0.05:
            return 999

    timeout = time.time()

    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()

        if time.time() - timeout > 0.05:
            return 999

    duration = pulse_end - pulse_start

    distance = duration * 17150

    return distance

# =========================
# KAMERA
# =========================

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

motor_forward(MOTOR_SPEED)

try:

    while True:

        distance = get_distance()

        if distance < SAFE_DISTANCE:
            motor_stop()
            set_servo(CENTER_ANGLE)
            continue

        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]

        x1 = (w - ROI_WIDTH) // 2
        y1 = h - ROI_HEIGHT

        roi = frame[y1:h, x1:x1+ROI_WIDTH]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # =====================
        # KIRMIZI
        # =====================

        lower_red1 = np.array([0, 120, 80])
        upper_red1 = np.array([10, 255, 255])

        lower_red2 = np.array([170, 120, 80])
        upper_red2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        red_mask = red_mask1 | red_mask2

        # =====================
        # YEŞİL
        # =====================

        lower_green = np.array([35, 80, 80])
        upper_green = np.array([90, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        red_contours, _ = cv2.findContours(
            red_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        green_contours, _ = cv2.findContours(
            green_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        target_x = None

        # =====================
        # KIRMIZI ÖNCELİK
        # =====================

        if len(red_contours) > 0:

            c = max(red_contours, key=cv2.contourArea)

            if cv2.contourArea(c) > 400:

                x, y, ww, hh = cv2.boundingRect(c)

                center_x = x + ww // 2

                target_x = center_x + OFFSET

        elif len(green_contours) > 0:

            c = max(green_contours, key=cv2.contourArea)

            if cv2.contourArea(c) > 400:

                x, y, ww, hh = cv2.boundingRect(c)

                center_x = x + ww // 2

                target_x = center_x - OFFSET

        if target_x is not None:

            roi_center = ROI_WIDTH // 2

            error = target_x - roi_center

            angle = CENTER_ANGLE + (KP * error)

            set_servo(angle)

        else:
            set_servo(CENTER_ANGLE)

        cv2.imshow("ROI", roi)

        if cv2.waitKey(1) & 0xFF == 27:
            break

except KeyboardInterrupt:
    pass

finally:

    motor_stop()

    servo_pwm.stop()
    motor_pwm.stop()

    GPIO.cleanup()

    cap.release()
    cv2.destroyAllWindows()
