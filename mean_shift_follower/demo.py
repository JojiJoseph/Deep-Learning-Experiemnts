import cv2
import numpy as np
import time
from math import ceil

def main():
    points = []
    arrow_location = np.array([400, 400])
    arrow_direction = np.array([0, 0])
    width, height = 800, 800
    window_name = 'Drag mouse'
    velocity = 250 # pixels per second
    effective_arrow_direction = np.array([0, 0])
    arrow_trail = []

    lifetime = 2

    mouse_down = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal mouse_down
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_down = False
        if mouse_down:
            offset_x = np.random.randint(-10, 10)
            offset_y = np.random.randint(-10, 10)
            time_offset = 0#np.random.uniform(-1.5, 1.5)
            points.append((x + offset_x, y + offset_y, time.time() + time_offset))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        image = np.zeros((width, height, 3), dtype=np.uint8)
        for point in points:
            x, y, t = point
            if time.time() - t < lifetime:
                radius = ceil((lifetime - (time.time()-t)) / lifetime * 5)
                cv2.circle(image, (x, y), radius, (0, 255, 0), -1)
        # Update points
        points = [point for point in points if time.time() - point[2] < lifetime]
        arrow_trail = [point for point in arrow_trail if time.time() - point[1] < lifetime]

        cv2.arrowedLine(image, (int(arrow_location[0]), int(arrow_location[1])), (int(arrow_location[0] + effective_arrow_direction[0]*50), int(arrow_location[1] + effective_arrow_direction[1]*50)), (0, 0, 255), 5)
        arrow_trail_points = [[x,y ] for (x, y), t in arrow_trail]
        for i in range(1, len(arrow_trail_points)):
            cv2.line(image, (int(arrow_trail_points[i-1][0]), int(arrow_trail_points[i-1][1])), (int(arrow_trail_points[i][0]), int(arrow_trail_points[i][1])), (0, 0, 255), 5)
        cv2.imshow(window_name, image)
        last_time = time.time()
        key = cv2.waitKey(10) & 0xFF
        # Draw arrow
        if key == ord('q') or key == 27:
            break

        # Calculate direction of arrow
        points_np = [[x, y] for x, y, t in points]
        points_np = np.array(points_np)
        if len(points_np) > 0:
            # print((np.linalg.norm(points_np-np.array(arrow_location), axis=1)[...,None]*points_np).shape)
            raddi = np.array([ceil((lifetime - (time.time()-t)) / lifetime * 5) for x, y, t in points])
            numerator = np.sum(np.linalg.norm(points_np-np.array(arrow_location), axis=1)[...,None]*points_np*raddi[:,None], axis=0)
            denominator = np.sum(np.linalg.norm(points_np-np.array(arrow_location), axis=1)*raddi)
            arrow_direction = numerator / denominator - arrow_location
            arrow_direction = arrow_direction / np.linalg.norm(arrow_direction)
        delta_time = time.time() - last_time
        # check if arrow_direction is nan
        if np.isnan(arrow_direction).any():
            arrow_direction = np.array([0, 0])
        effective_arrow_direction = 0.01 * arrow_direction + 0.99 * effective_arrow_direction
        arrow_location = arrow_location + effective_arrow_direction * velocity * delta_time
        arrow_trail.append([arrow_location, time.time()])

if __name__ == '__main__':
    main()