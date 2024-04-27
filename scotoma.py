import flet as ft
import cv2
import numpy as np
import base64
import pytesseract
import time


# initialize camera
cap = cv2.VideoCapture(0)
# Control the size of the scotoma
scotoma_radius = 100

class ScotomaWrapper(ft.UserControl):

    def __init__(self):
        super().__init__()
        self.tangent_color = (0, 0, 255)  # red
        self.last_move_time = time.time()
        self.just_drew = False
        self.last_draw_time = time.time()
        self.overlay_text = None
        self.curve_text = None



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

            # cut image to within the scotoma for OCR analysis
            mask = np.zeros_like(frame)
            cv2.circle(mask, circle_center, radius, (255,255,255),-1)

            ocr_frame = cv2.bitwise_and(frame,mask)     


            #ocr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            #ocr_frame[:, :, 3] = mask[:, :, 0]
            read_frame = np.copy(frame)
            cv2.circle(frame, circle_center, radius, (0, 0, 0), -1)  # -1 means filled circle

            # DRAW ADJACANT TO SCOTOMA
            tangent_radius = scotoma_radius / 4  # Adjust this as you want
            tangent_center = (int(circle_center[0] + scotoma_radius + tangent_radius), circle_center[1])
            # cv2.circle(frame, tangent_center, int(tangent_radius), self.tangent_color, -1)

            # OCR
            # we need to convert from BGR to RGB format/mode:
            # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
            # we need to convert from BGR to RGB format/mode:
            ocr_frame = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2RGB)
            # Convert to greyscale 
            ocr_frame = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2GRAY)
            # ocr_frame = cv2.Canny(ocr_frame, threshold1=30, threshold2=100)
            ocr_frame = cv2.Canny(ocr_frame, threshold1=255/3, threshold2=255)
            # thresholding
            # ocr_frame = cv2.threshold(ocr_frame, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            # https://stackoverflow.com/questions/60009533/drawing-bounding-boxes-with-pytesseract-opencv
            # --psm 6: This parameter sets the page segmentation mode (PSM) to 6. Page segmentation mode determines how 
            #          Tesseract should interpret the structure of the input image during OCR. PSM 6, also known as 
            #          "Assume a single uniform block of text," is suitable for cases where the image contains a single block of 
            #           text without any column, paragraph, or line structure. It's commonly used for scenarios like reading a single 
            #           word or a small text region.
            # custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6'
            # d = pytesseract.image_to_data(ocr_frame, output_type=pytesseract.Output.DICT, lang='eng', config=custom_config)
            d = pytesseract.image_to_data(ocr_frame, output_type=pytesseract.Output.DICT)
            n_boxes = len(d['level'])

            # Check if it's time to trigger OCR again (15 secs since u last drew)
            if self.just_drew and time.time() - self.last_draw_time >= 15:
                self.just_drew = False
                self.overlay_text = None

            # for i in range(n_boxes):
            #   if d['level'][i] == 5 and d['conf'][i] >= 40:
            #       (x,y,w,h) = (d['left'][i], d['top'][i], d['width'][i],d['height'][i])
            #        cv2.rectangle(ocr_frame, (x,y), (x+w, y+h), (255,255,255),2)

            # redraw text outside of scotoma
            for i in range(n_boxes):
                # LEVELS:
                # 1. page
                # 2. block
                # 3. paragraph
                # 4. line
                # 5. word

                if d['level'][i] == 5 and d['width'][i]<200 and d['height'][i]<200 and d["conf"][i] > 80 and (d['width'][i] > 20 and d['height'][i] > 10):
                    print(".")
                    text_left = d['left'][i]
                    text_top = d['top'][i]
                    text_width = d['width'][i] + 20
                    text_height = d['height'][i] + 5
                    text_test = d['text'][i]
                    confidence = d["conf"][i]
                    print(confidence)

                    # for now just move the text above the circle
                    new_text_left = text_left
                    new_text_top = circle_center[1]-radius-text_height

                    # unless its too big, then move it to the right
                    if new_text_top < 0:
                        print("Scotoma too big!")
                        new_text_top = circle_center[1] - text_height // 2
                        new_text_left = circle_center[0] + radius + 10

                    # Store the drawn text for the overla
    
                    self.overlay_text = read_frame[
                            text_top:text_top+text_height,
                            text_left:text_left+text_width,:]
                    

                    self.curve_text = frame[new_text_top:new_text_top+text_height,
                         new_text_left:new_text_left+text_width,:]
                    
                    self.last_draw_time = time.time()
                    self.just_drew = True
                    # frame[new_text_top:new_text_top+text_height,
                    #     new_text_left:new_text_left+text_width,] = read_frame[
                    #     text_top:text_top+text_height,
                    #     text_left:text_left+text_width,]
                    # time.sleep(5)

            if self.overlay_text is not None:
                self.overlay_text = self.overlay_text.astype(np.float32)
                self.curve_text = self.curve_text.astype(np.float32)
               # M = cv2.getPerspectiveTransform(self.overlay_text,self.curve_text)
                #frame = cv2.warpPerspective(frame, M, frame.shape)
                frame[new_text_top:new_text_top+text_height,
                         new_text_left:new_text_left+text_width,:] = self.overlay_text

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