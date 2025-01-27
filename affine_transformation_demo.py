import numpy as np
import cv2

def construct_affine(rotation_angle, scale, shear, translation):
    """
    Construct an affine transformation matrix from rotation, scale, shear, and translation.
    
    Args:
        rotation_angle (float): Rotation angle in radians.
        scale (tuple): Scale factors (s_x, s_y).
        shear (float): Shear factor.
        translation (tuple): Translation (t_x, t_y).
    
    Returns:
        np.ndarray: 2x3 affine transformation matrix.
    """
    # Unpack components
    s_x, s_y = scale
    t_x, t_y = translation
    
    # Rotation matrix
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    R = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Apply scale
    R_scaled = R * [s_x, s_y]  # Element-wise scaling
    
    # Add shear
    A = np.array([
        [R_scaled[0, 0], R_scaled[0, 1] + shear],
        [R_scaled[1, 0], R_scaled[1, 1]]
    ])
    
    # Add translation
    M = np.hstack([A, np.array([[t_x], [t_y]])])
    return M

# Example usage
rotation_angle = np.radians(30)  # 30 degrees
scale = (1.5, 2.0)              # Scale factors
shear = 0.2                     # Shear factor
translation = (10, 20)          # Translation

affine_matrix = construct_affine(rotation_angle, scale, shear, translation)
print("Affine Transformation Matrix:\n", affine_matrix)


initial_points = np.random.rand(40, 2)
bb_points = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Rotation", "Image", 0, 360, lambda x: None)
cv2.createTrackbar("Scale X", "Image", 100, 200, lambda x: None)
cv2.createTrackbar("Scale Y", "Image", 100, 200, lambda x: None)
cv2.createTrackbar("Shear", "Image", 0, 200, lambda x: None)
cv2.createTrackbar("Translation X", "Image", 256, 512, lambda x: None)
cv2.createTrackbar("Translation Y", "Image", 256, 512, lambda x: None)
cv2.setTrackbarMin("Scale X", "Image", 1)
cv2.setTrackbarMin("Scale Y", "Image", 1)
cv2.setTrackbarMin("Shear", "Image", -200)
cv2.setTrackbarMin("Translation X", "Image", 0)
cv2.setTrackbarMin("Translation Y", "Image", 0)

# noise = 

while True:
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    initial_points_in_pixels = (initial_points * 100 + 256).astype(np.int32)
    bb_points_in_pixels = (bb_points * 100 + 256).astype(np.int32)
    for point in initial_points_in_pixels:
        cv2.circle(img, tuple(point), 5, (255, 255, 255), -1)
    cv2.polylines(img, [bb_points_in_pixels], isClosed=True, color=(255, 255, 255), thickness=2)

    rotation_angle = np.radians(cv2.getTrackbarPos("Rotation", "Image"))
    scale_x = cv2.getTrackbarPos("Scale X", "Image") / 100
    scale_y = cv2.getTrackbarPos("Scale Y", "Image") / 100
    shear = cv2.getTrackbarPos("Shear", "Image") / 100
    translation_x = cv2.getTrackbarPos("Translation X", "Image")- 256
    translation_y = cv2.getTrackbarPos("Translation Y", "Image") - 256
    translation_x /= 100
    translation_y /= 100

    A_gt = construct_affine(rotation_angle, (scale_x, scale_y), shear, (translation_x, translation_y))
    transformed_points = np.dot(np.hstack([initial_points, np.ones((initial_points.shape[0], 1))]), A_gt.T)
    print(transformed_points.shape)
    # transformed_points /= transformed_points[:, 2]#[:, None]
    transformed_points = transformed_points[:, :2]
    transformed_bb = np.dot(np.hstack([bb_points, np.ones((bb_points.shape[0], 1))]), A_gt.T)
    transformed_points_in_pixels = (transformed_points * 100 + 256).astype(np.int32)
    transformed_bb_in_pixels = (transformed_bb * 100 + 256).astype(np.int32)

    for point in transformed_points_in_pixels:
        cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)
    cv2.polylines(img, [transformed_bb_in_pixels], isClosed=True, color=(0, 255, 0), thickness=2)

    estimated_A, _ = cv2.estimateAffine2D(initial_points, transformed_points, method=cv2.RANSAC,maxIters=10000, ransacReprojThreshold=0)# + np.random.randn(*initial_points.shape) * 0.1)
    estimated_transformed_points = cv2.transform(np.array([initial_points]), estimated_A)
    estimated_transformed_points = estimated_transformed_points[0]
    estimated_transformed_points_in_pixels = (estimated_transformed_points * 100 + 256).astype(np.int32)
    estimated_bb = cv2.transform(np.array([bb_points]), estimated_A)
    estimated_bb = estimated_bb[0]
    estimated_bb_in_pixels = (estimated_bb * 100 + 256).astype(np.int32)
    for point in estimated_transformed_points_in_pixels:
        cv2.circle(img, tuple(point), 5, (0, 0, 255), -1)
    cv2.polylines(img, [estimated_bb_in_pixels], isClosed=True, color=(0, 0, 255), thickness=2)
    key = cv2.waitKey(1)
    cv2.imshow("Image", img)
    if key == ord('q'):
        break
    # elif key == ord('r'):
    #     initial_points = np.random.rand((10, 2))
    