import flet as ft
import cv2
import numpy as np
import base64
import pytesseract
import math
import time
from scipy.ndimage import map_coordinates

# initialize camera
cap = cv2.VideoCapture(0)

# New libraries make camera sizes wonky for Yan and I. Forcing a standard size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

# FOR TESTING, ON RUNTIME SHOULD BE SET TO 1 (0-1)
blend_amount = 1

if not cap.isOpened():
    print("Error: Unable to open camera. Camera didn't open successfully. Did you make sure it's not connected to ur phone and ur phone sync isn't being slow?")
    exit()

# Control the size of the scotoma
scotoma_radius = 100

def remap_image_around_scotoma(frame, scotoma_radius):
    try:
        image_dims = frame.shape
        image_height, image_width = image_dims[0], image_dims[1]
        # print(image_height, image_width )
        scotoma_center = (image_width // 2, image_height // 2)

        # Calculate displacement of pixels from the scotoma center
        dy, dx = np.indices((image_height, image_width)) - np.array(scotoma_center)[:, None, None]
        distance = np.sqrt(dx**2 + dy**2)

        # area outside the scotoma
        outside_scotoma_mask = distance > scotoma_radius

        dx_normalized = np.where(outside_scotoma_mask, dx / distance, 0)
        dy_normalized = np.where(outside_scotoma_mask, dy / distance, 0)

        # sscale the displacement
        displacement_x = dx_normalized * scotoma_radius
        displacement_y = dy_normalized * scotoma_radius

        # new coordinates after remapping
        new_x = (np.indices((image_height, image_width))[1] - displacement_x).astype(int)
        new_y = (np.indices((image_height, image_width))[0] - displacement_y).astype(int)

        # ensure ur within the image bounds
        new_x = np.clip(new_x, 0, image_width - 1)
        new_y = np.clip(new_y, 0, image_height - 1)

        remapped_image = np.zeros_like(frame)
        for channel in range(frame.shape[2]):
            remapped_image[:,:,channel] = map_coordinates(frame[:,:,channel], [new_y, new_x], order=1, mode='reflect')
        
        # Blend the remapped image with the original image
        blended_image = frame.copy()
        blended_image[outside_scotoma_mask] = blend_amount * remapped_image[outside_scotoma_mask] + \
                                              (1 - blend_amount) * frame[outside_scotoma_mask]
        blended_image[~outside_scotoma_mask] = [0, 0, 0]
        return blended_image
        
        return remapped_image

    except Exception as e:
        print("Error:", e)
        return frame




class ScotomaWrapper(ft.UserControl):

    def __init__(self):
        super().__init__()
        self.last_move_time = time.time()
        self.overlay_text = None
        self.prev_frame = None

    def did_mount(self):
        self.update_timer()

    # This will be the entry point to rendering the camera image on the screen.
    def update_timer(self):
        print("Updating timer...")
        while True:
            _, frame = cap.read()
            # Draw scotoma circle
            # circle_center = (image_width // 2, image_height // 2)
            # cv2.circle(frame, circle_center, scotoma_radius, (0, 0, 0), -1)  # -1 means filled circle

            # Remap the image around the scotoma
            frame = remap_image_around_scotoma(frame, scotoma_radius)



            _, im_arr = cv2.imencode(".png", frame)
            im_b64 = base64.b64encode(im_arr)
            self.img.src_base64 = im_b64.decode('utf-8')
            self.update()   


    def build(self):
        self.img = ft.Image(border_radius=ft.border_radius.all(20))
        return ft.Column([
            self.img
        ])


camera_container = ft.Container(
    margin = ft.margin.only(bottom=40),
    content=ft.Row([
        ft.Card(
            elevation=30,
            content=ft.Container(
                bgcolor='blue',
                padding=10,
                border_radius=ft.border_radius.all(20),
                content=ft.Column([
                    ft.Text("Camera", 
                            size=30, 
                            weight="bold",
                            color="white"
                            ),
                        ScotomaWrapper()
                ])
            )
        )
    ], alignment='center')
)


def main(page: ft.Page):
    page.title = "Visualize Scotoma"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.padding=50
    page.window_left=page.window_left+100
    page.theme_mode='light'

    # Allow the user to change the size of the scotoma
    txt_number = ft.TextField(value=str(scotoma_radius), text_align=ft.TextAlign.RIGHT, width=100)
    def decrement_click(e):
        global scotoma_radius
        txt_number.value = str(int(txt_number.value) - 1)
        scotoma_radius = int(txt_number.value)
        page.update()

    def increment_click(e):
        global scotoma_radius
        txt_number.value = str(int(txt_number.value) + 1)
        scotoma_radius = int(txt_number.value)
        page.update()
    
    # Increment Text Field
    page.add(
        ft.Row(
            [
                ft.IconButton(ft.icons.REMOVE, on_click=decrement_click),
                txt_number,
                ft.IconButton(ft.icons.ADD, on_click=increment_click),
            ]
        )
    )
    # Main camera frame
    page.add(camera_container)
    
if __name__ == "__main__":
    ft.app(target=main) # <--- Visualize in native OS window
    # ft.app(target=main, view=ft.AppView.WEB_BROWSER) # <--- Visualize in a web browser
    cap.release()
    cv2.destroyAllWindows()
