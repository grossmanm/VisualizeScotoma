import flet as ft
import cv2
import base64


# initialize camera
cap = cv2.VideoCapture(0)

class ScotomaWrapper(ft.UserControl):

    def __init__(self):
        super().__init__()


    def did_mount(self):
        self.update_timer()

    # This will the the entry point to rendering the camera image on the screen.
    def update_timer(self):
        image_path = "pit.png" 
        image = cv2.imread(image_path) # cv2.imread("pit.png", cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        if image is None:
            print("Error: Could not load image")
            return
        print(f"Height:\t\t {image.shape[0]}") 
        print(f"Width:\t\t {image.shape[1]}") 
        print(f"Channel:\t {image.shape[2]}")

        while True:
            _,frame=cap.read()

            h, w, _ = image.shape
            frame[10:10+h, 10:10+w] = image  # Adjust position as needed
            

            _,im_arr = cv2.imencode(".png", frame)
            im_b64 = base64.b64encode(im_arr)
            self.img.src_base64 = im_b64.decode('utf-8')
            self.update()

    def build(self):
        self.img = ft.Image(border_radius=ft.border_radius.all(20))
        return ft.Column([
            self.img
        ])
    
section = ft.Container(
    margin = ft.margin.only(bottom=40),
    content=ft.Row([
        ft.Card(
            elevation=30,
            content=ft.Container(
                bgcolor='blue',
                padding=10,
                border_radius=ft.border_radius.all(20),
                content=ft.Column([
                    ft.Text("Camera", size=30, weight="bold",
                            color="white"
                            ),
                        ScotomaWrapper()
                ])
            )
        )
    ], alignment='center')
)
def main(page: ft.Page):
    page.padding=50
    page.window_left=page.window_left+100
    page.theme_mode='light'
    page.add(section)
if __name__ == "__main__":

    ft.app(target=main)
    cap.release()
    cv2.destroyAllWindows()
