import cv2
import numpy as np

RED_LOWER = np.array([0, 100, 100])
RED_UPPER = np.array([10, 255, 255])

GREEN_LOWER = np.array([40, 100, 100])
GREEN_UPPER = np.array([80, 255, 255])

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Kamera acilamadi!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_r = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
        mask_g = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

        red_pixels = cv2.countNonZero(mask_r)
        green_pixels = cv2.countNonZero(mask_g)

        if green_pixels > 500 and red_pixels <= 500:
            cmd = "SAGA DON"
            color = (0, 255, 0)
        elif red_pixels > 500 and green_pixels <= 500:
            cmd = "SOLA DON"
            color = (0, 0, 255)
        elif red_pixels > 500 and green_pixels > 500:
            cmd = "DUZ GIT"
            color = (0, 255, 255)
        else:
            cmd = "DUZ GIT"
            color = (255, 255, 255)

        cv2.putText(frame, cmd, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("TEST V1", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()