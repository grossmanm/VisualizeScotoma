import flet as ft
import cv2
import numpy as np
import base64
import pytesseract
import time


def check_overlap(x1,y1,w1,h1,x2,y2,w2,h2):
    return not(x1+w1+5<=x2 or x1>=x2+w2+5 or y1+h1+5<=y2 or y1 >=y2+h2+5)


# initialize camera
cap = cv2.VideoCapture(0)

# New libaries make camera sizes wonky for Yan and I. Forcing a standard size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Unable to open camera. Camera didn't open successfully. Did you make sure it's not connected to ur phone and ur phone sync isn't being slow?")
    exit()

# Control the size of the scotoma
scotoma_radius = 100

class ScotomaWrapper(ft.UserControl):

    def __init__(self):
        super().__init__()
        self.last_move_time = time.time()
        self.just_drew = False
        self.last_draw_time = time.time()
        self.overlay_text = None
        self.curve_text = None
        self.prev_ocr_frame = None
        self.prev_scotoma_frame = None
        self.prev_scotoma_radius = scotoma_radius

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

            # cut image to within the scotoma for OCR analysis
            mask = np.zeros_like(frame)
            cv2.circle(mask, circle_center, scotoma_radius, (255,255,255),-1)

            ocr_frame = cv2.bitwise_and(frame,mask)     

            #ocr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            #ocr_frame[:, :, 3] = mask[:, :, 0]
            read_frame = np.copy(frame)
            cv2.circle(frame, circle_center, scotoma_radius, (0, 0, 0), -1)  # -1 means filled circle

            # OCR
            # we need to convert from BGR to RGB format/mode:
            # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
            # we need to convert from BGR to RGB format/mode:
            ocr_frame = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2RGB)

            # normalize frame
            norm_img = np.zeros((ocr_frame.shape[0], ocr_frame.shape[1]))
            ocr_frame = cv2.normalize(ocr_frame, norm_img, 0, 255, cv2.NORM_MINMAX)

            # Convert to greyscale 
            ocr_frame = cv2.cvtColor(ocr_frame, cv2.COLOR_BGR2GRAY)
            #ocr_frame = cv2.Canny(ocr_frame, threshold1=30, threshold2=100)
            #ocr_frame = cv2.Canny(ocr_frame, threshold1=255/3, threshold2=255)
            # thresholding
            ocr_frame = cv2.adaptiveThreshold(ocr_frame, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 6)
            #ocr_frame = cv2.threshold(ocr_frame, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

            # to reduce noise we only consider the area around the scotoma
            scotoma_frame = ocr_frame[circle_center[1]-scotoma_radius:circle_center[1]+scotoma_radius,
                                      circle_center[0]-scotoma_radius:circle_center[0]+scotoma_radius]
            if type(self.prev_ocr_frame) is not type(None) and scotoma_radius == self.prev_scotoma_radius:
                diff = cv2.absdiff(scotoma_frame, self.prev_scotoma_frame)
                total_size = self.prev_scotoma_frame.size
                num_diff_pixels = np.count_nonzero(diff)
                percent_diff = (num_diff_pixels/total_size)
                if percent_diff < 0.05:
                   ocr_frame = self.prev_ocr_frame
                   scotoma_frame = self.prev_scotoma_frame
                else:
                    self.prev_ocr_frame = ocr_frame
                    self.prev_scotoma_frame = scotoma_frame
                    self.prev_scotoma_radius = scotoma_radius
            else:   
                self.prev_ocr_frame = ocr_frame
                self.prev_scotoma_frame = scotoma_frame
                self.prev_scotoma_radius = scotoma_radius

            
            # https://stackoverflow.com/questions/60009533/drawing-bounding-boxes-with-pytesseract-opencv
            # --psm 6: This parameter sets the page segmentation mode (PSM) to 6. Page segmentation mode determines how 
            #          Tesseract should interpret the structure of the input image during OCR. PSM 6, also known as 
            #          "Assume a single uniform block of text," is suitable for cases where the image contains a single block of 
            #           text without any column, paragraph, or line structure. It's commonly used for scenarios like reading a single 
            #           word or a small text region.
            # custom_config = r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6'
            # d = pytesseract.image_to_data(ocr_frame, output_type=pytesseract.Output.DICT, lang='eng', config=custom_config)
            d = pytesseract.image_to_data(scotoma_frame, output_type=pytesseract.Output.DICT)
           # print(d)
            n_boxes = len(d['level'])

            # Check if it's time to trigger OCR again (15 secs since u last drew)
            if self.just_drew and time.time() - self.last_draw_time >= 15:
                self.just_drew = False
                self.overlay_text = None

            

            clean_output = {'left':[], 'top':[], 'width':[], 'height':[], 'text':[]}
            # clean up boxes to remove overlaps and certain levels
            #print(d)
            remove_idx = 0
            for i in range(len(d['left'])):
                if d['level'][i-remove_idx] < 2 or (d['width'][i-remove_idx] >=scotoma_radius*2 and d['height'][i-remove_idx] >= scotoma_radius*2):
                    d['left'].pop(i-remove_idx)
                    d['top'].pop(i-remove_idx)
                    d['width'].pop(i-remove_idx)
                    d['height'].pop(i-remove_idx)
                    d['level'].pop(i-remove_idx)
                    remove_idx+=1
            while len(d['left']) > 0:
                cur_left = d['left'].pop(0)+(circle_center[0]-scotoma_radius)
                cur_top = d['top'].pop(0)+(circle_center[1]-scotoma_radius)
                cur_width = d['width'].pop(0)
                cur_height = d['height'].pop(0)

                merged_left = cur_left
                merged_top = cur_top
                merged_width = cur_width
                merged_height = cur_height

                i = 0
                while i < len(d['left']):
                    if check_overlap(merged_left,merged_top,merged_width,merged_height,
                               d['left'][i]+(circle_center[0]-scotoma_radius),d['top'][i]+(circle_center[1]-scotoma_radius),
                               d['width'][i],d['height'][i]):
                        merged_left = min(merged_left,d['left'][i]+(circle_center[0]-scotoma_radius))
                        merged_top = min(merged_top,d['top'][i]+(circle_center[1]-scotoma_radius))
                        if merged_left+merged_width > d['left'][i]+(circle_center[0]-scotoma_radius)+d['width'][i]:
                            merged_width=merged_width
                        else:
                            merged_width=d['width'][i]
                        if merged_top+merged_height > d['top'][i]+(circle_center[1]-scotoma_radius)+d['height'][i]:
                            merged_height=merged_height
                        else:
                            merged_height=d['height'][i]
                        d['left'].pop(i)
                        d['top'].pop(i)
                        d['width'].pop(i)
                        d['height'].pop(i)
                    else:
                        i+=1
                clean_output['left'].append(merged_left)
                clean_output['top'].append(merged_top)
                clean_output['width'].append(merged_width)
                clean_output['height'].append(merged_height)
            '''
            for i in range(n_boxes):
                # Filter out small, insignificant bounding box
                # LEVELS:
                # 1. page
                # 2. block
                # 3. paragraph
                # 4. line
                # 5. word
                if d['level'][i] >= 2 and d['width'][i] < 200 and d['height'][i] < 200:
                    cur_left = d['left'][i]+(circle_center[0]-scotoma_radius)
                    cur_top = d['top'][i]+(circle_center[1]-scotoma_radius)
                    cur_width = d['width'][i]
                    cur_height = d['height'][i]
                    area_cur = cur_width*cur_height

                    overlap_indices = []
                    did_overlap = False
                    largest = False

                    # check for overlaps
                    for j in range(n_boxes):
                        if j == i: 
                            continue
                        old_left = d['left'][j]+(circle_center[0]-scotoma_radius)
                        old_top = d['top'][j]+(circle_center[1]-scotoma_radius)
                        old_width = d['width'][j]
                        old_height = d['height'][j]

                        dx = min(cur_left+cur_width, old_left+old_width)-max(cur_left, old_left)
                        dy = min(cur_top+cur_height, old_top+old_height)-max(cur_top, old_top)
                        if (dx<0) or (dy<0):
                            continue
                        did_overlap = True
                        intersection_area = dx*dy
                        area_old = old_width*old_height
                        overlap=(intersection_area)/(min(area_cur,area_old))
                        print(overlap)
                        print(area_cur)
                        print(area_old)
                        if overlap >=0.1 and area_cur > area_old:
                            largest=True
                    if not did_overlap or largest:
                        clean_output['left'].append(cur_left)
                        clean_output['top'].append(cur_top)
                        clean_output['width'].append(cur_width)
                        clean_output['height'].append(cur_height)
                        clean_output['text'].append(d['text'][i])
            '''
            #for i in range(len(clean_output['left'])):
            #   (x,y,w,h) = (clean_output['left'][i], clean_output['top'][i], clean_output['width'][i],clean_output['height'][i])
            #   cv2.rectangle(frame, (x,y), (x+w, y+h), (255,255,255),2)

            # redraw text outside of scotoma
            #print(1)
            for i in range(len(clean_output['left'])):
                text_left = clean_output['left'][i]
                text_top = clean_output['top'][i]
                text_width = clean_output['width'][i] 
                text_height = clean_output['height'][i]
               # print(clean_output['text'][i])
                # text_test = clean_output['text'][i]

                # for now just move the text above the circle
                new_text_left = text_left
                # check if above or below center of scotoma
                if text_top <= circle_center[1]:
                    new_text_top = circle_center[1]-scotoma_radius-text_height-(circle_center[1]-text_top)
                else:
                    new_text_top = circle_center[1]+scotoma_radius+(text_top-circle_center[1])

                #new_text_top = circle_center[1]-scotoma_radius-text_height

                # unless its too big, then move it to the right
                if new_text_top < 0:
                    print("Scotoma too big!")
                    new_text_top = circle_center[1] - text_height // 2
                    new_text_left = circle_center[0] + scotoma_radius + 10

                # Store the drawn text for the overla
                self.overlay_text = read_frame[
                        text_top:text_top+text_height,
                        text_left:text_left+text_width,:]
                
                # self.curve_text = frame[new_text_top:new_text_top+text_height,
                #         new_text_left:new_text_left+text_width,:]
                
                self.last_draw_time = time.time()
                self.just_drew = True

               # if self.overlay_text is not None:
                # print(i)
                self.overlay_text = self.overlay_text.astype(np.float32)
                # self.curve_text = self.curve_text.astype(np.float32)
                # M = cv2.getPerspectiveTransform(self.overlay_text,self.curve_text)
                # frame = cv2.warpPerspective(frame, M, frame.shape)
                frame[new_text_top: new_text_top+text_height,
                    new_text_left: new_text_left+text_width,:] = self.overlay_text
                
           

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