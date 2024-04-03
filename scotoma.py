import flet as ft
import cv2
import numpy as np
import base64


# initialize camera
cap = cv2.VideoCapture(0)
# Control the size of the scotoma
scotoma_radius = 180

class ScotomaWrapper(ft.UserControl):

    def __init__(self):
        super().__init__()
        self.tangent_color = (0, 0, 255)  # red


    def did_mount(self):
        self.update_timer()

    # This will the the entry point to rendering the camera image on the screen.
    def update_timer(self):
        while True:
            _, frame = cap.read()
            image_dims = frame.shape
            image_height, image_width = image_dims[0], image_dims[1]

            # DRAW SCOTOMA
            circle_center = (image_width // 2, image_height // 2)
            radius = scotoma_radius
            cv2.circle(frame, circle_center, radius, (0, 0, 0), -1)  # -1 means filled circle

            # DRAW ADJACANT TO SCOTOMA
            tangent_radius = scotoma_radius / 4  # Adjust this as you want
            tangent_center = (int(circle_center[0] + scotoma_radius + tangent_radius), circle_center[1])
            cv2.circle(frame, tangent_center, int(tangent_radius), self.tangent_color, -1)


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
    txt_number = ft.TextField(value="180", text_align=ft.TextAlign.RIGHT, width=100)
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
