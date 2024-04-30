import numpy as np
import cv2
import math

def wrap_rectangle_around_circle_top_half(image_size, circle_center, circle_radius, rectangle_width, rectangle_height):
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    cv2.circle(image, circle_center, circle_radius, (255, 255, 255), -1)

    # Calculate the angle range for the rectangle
    # width of the rectangle in relation to the circumference of the circle.
    angle_range = math.pi * (rectangle_width / (2 * math.pi * circle_radius))

    # Create a mask for the wrapped rectangle
    rectangle_mask = np.zeros_like(image[:, :, 0])
    # This is how much of a UNIT CIRCLE you wanna cover,
    # but inverse for some reason
    for angle in np.linspace(0, -math.pi, num=int(180 * angle_range), endpoint=False):
    # for angle in np.linspace(math.pi / 2, 3 * math.pi / 2, num=int(180 * angle_range), endpoint=False):
    # for angle in np.linspace(math.pi, 0, num=int(180 * angle_range), endpoint=False):
        x = int(circle_center[0] + circle_radius * math.cos(angle))
        y = int(circle_center[1] + circle_radius * math.sin(angle))
        cv2.line(rectangle_mask, (x, y), (x, y), 255, rectangle_height // 2)

    # Apply Gaussian smoothing to the rectangle mask
    gaussian_kernel = cv2.getGaussianKernel(rectangle_height, 0)
    remapped_mask = cv2.sepFilter2D(rectangle_mask.astype(np.float32), -1, gaussian_kernel, gaussian_kernel)

    # Create a red rectangle mask
    red_rectangle_mask = np.zeros_like(image)
    red_rectangle_mask[:, :, 2] = remapped_mask.astype(np.uint8)
    wrapped_image = cv2.addWeighted(image, 1, red_rectangle_mask, 1, 0)

    return wrapped_image





# ---------------------------------------------------------------
image_size = 400
circle_center = (200, 200)
circle_radius = 100
rectangle_width = 100
rectangle_height = 50

result = wrap_rectangle_around_circle_top_half(image_size, circle_center, circle_radius, rectangle_width, rectangle_height)

cv2.imshow('Wrapped Rectangle (Top Half)', result)
cv2.waitKey(0)
cv2.destroyAllWindows()