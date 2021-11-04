import cv2
import cv2.aruco as aruco
import numpy as np

def get_rot_matrix(alpha, beta, gamma):
    R1 = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha), np.cos(alpha)],
        ])
    R2 = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)],
        ])
    R3 = np.array([
        [np.cos(gamma), -np.sin(gamma), 0],
        [np.sin(gamma), np.cos(gamma), 0],
        [0, 0, 1],
        ])
    return R3 @ R2 @ R1

cap = cv2.VideoCapture(2)
cap.set(3, 1280)
cap.set(4, 720)
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

camera_matrix = np.array([
    [1430, 0, 480],
    [0,1430,620],
    [0,0,1]
], dtype=float)

mouse_active_x = None
mouse_active_y = None
def callback(event, x, y, flags, param):
    global mouse_active_x, mouse_active_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_active_x = x
        mouse_active_y = y
        # print("Here")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",callback)

R = np.eye(3)
tl = [0,0,0]
while True:
    ret, frame = cap.read()
    # print(frame.shape)
    corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict,
        parameters=aruco_params)
    aruco.drawDetectedMarkers(frame, corners, ids)
    rvecs, tvecs,_ = aruco.estimatePoseSingleMarkers(corners, 1, camera_matrix, None)
    if corners:
        for rvec, tvec in zip(rvecs, tvecs):
            aruco.drawAxis(frame, camera_matrix, np.array([[0,0.,0,0]]), rvec, tvec, 1)
        # print(len(corners), tvecs)
        r = rvecs[0][0]
        t0 = tvecs[0]
        for t, p, r in zip(tvecs, corners, rvecs):
            # print(len(p[0][0]),p[0][0])
            r = r[0]
            p = p[0]
            cx = int(p[0][0]+p[1][0]+p[2][0]+p[3][0])//4
            cy = int(p[0][1]+p[1][1]+p[2][1]+p[3][1])//4
            t = t[0]
            if mouse_active_y:
                # print("Here")
                if  (cx-mouse_active_x)**2 + (cy-mouse_active_y)**2< 20*20:
                    # R = get_rot_matrix(r[0],r[1],r[2])
                    print("r", r)
                    R,_ = cv2.Rodrigues(r)
                    print(R)
                    tl = -R.T @ t
                    mouse_active_y = None
                    mouse_active_x = None
            t2 = R.T @ t + tl
            # print(r)
            # cv2.putText(frame, f"{round(t[0]), round(t[1]),round(t[2])}",(cx+10,cy+10),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, f"{round(t2[0]), round(t2[1]),round(t2[2])}",(cx+10,cy+10),cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        R = np.eye(3)
        tl = [0,0,0]


cap.release()

