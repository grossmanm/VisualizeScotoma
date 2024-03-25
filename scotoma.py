import flet as ft
import cv2
import base64


# initailze camera
cap = cv2.VideoCapture(0)

class SarcomaWarper(ft.UserControl):

    def __init__(self):
        super().__init__()


    def did_mount(self):
        self.update_timer()

    def update_timer(self):
        while True:
            _,frame=cap.read()
        

            _,im_arr = cv2.imencode(".png",frame)
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
                        SarcomaWarper()
                ])
            )
        )
    ],alignment='center')
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
