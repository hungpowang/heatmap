import cv2
import numpy as np
import time

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# --------------------------
# timeing control
# --------------------------
# interval in mimutes
INTERVAL_MINUTES = 1  # integer: 1~30
HOUR_IN_MINUTES = 60
# initialization
mm_now = time.localtime(time.time()).tm_min
mm_nxt = (mm_now + INTERVAL_MINUTES) % HOUR_IN_MINUTES
print("mm_now:", mm_now)
print("mm_nxt:", mm_nxt)

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture("crowd.mp4")

net = cv2.dnn.readNetFromDarknet("yolov4-person.cfg", "yolov4-person_best.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/256)


def heat_amplifier(image, TH, img_w, img_h, center_x, center_y):
    '''
    用來畫更大的熱區
    '''
    sx = 0 if center_x - TH <= 0 else center_x - TH  # 要累加的x起點
    sy = 0 if center_y - TH <= 0 else center_y - TH  # 要累加的y起點
    range_x = TH * 2 + 1  # 範圍x
    range_y = TH * 2 + 1  # 範圍y
    itr_x = img_w - sx if range_x + sx >= img_w else range_x  # 範圍轉有修正x軸邊界的迭代次數
    itr_y = img_h - sy if range_y + sy >= img_h else range_y  # 範圍轉有修正y軸邊界的迭代次數
    for i in range(itr_x):
        for j in range(itr_y):
            image[sy+j][sx+i] += 1


frame_count = 0
NOT_GET_FRAME_YET = True
while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        break

    if NOT_GET_FRAME_YET:
        # 以第一個frame作為原圖暫存
        original_img = frame
        cv2.imwrite("original_img.jpg", original_img)
        NOT_GET_FRAME_YET = False
        # 新增一張類灰階底圖(正常圖片為np.unit8)
        base = np.full((frame.shape[0], frame.shape[1]), 0, np.uint16)
        print("底圖尺寸為:", base.shape)

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for i, (classid, score, box) in enumerate(zip(classes, scores, boxes)):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)

        # 找BBox中心點(先使用底部中心點)
        # box: [x y w h]
        cx = box[0] + box[2] // 2
        cy = box[1] + box[-1] - 1

        # ------------------------------------------
        # 把中心點加至底圖
        # ------------------------------------------
        # 基本款
        base[cy][cx] += 2  
        # 放大款
        img_w = base.shape[1]  # 底圖寬
        img_h = base.shape[0]  # 底圖長
        heat_amplifier(base, 9, img_w, img_h, cx, cy)
        heat_amplifier(base, 6, img_w, img_h, cx, cy)
        heat_amplifier(base, 4, img_w, img_h, cx, cy)

        color = (255, 0, 0)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()

    frame_count += 1
    # print("frame:", frame_count)
    mm_now = time.localtime(time.time()).tm_min
    hr_now = time.localtime(time.time()).tm_hour
    if mm_now % INTERVAL_MINUTES == 0:
        print("mm_now % INTERVAL_MINUTES == 0")
        if mm_nxt == time.localtime(time.time()).tm_min:
            print("中interval~~  mm_nxt={}, mm_now={}" .format(mm_nxt, mm_now))
            np.save("base_{}_{}" .format(hr_now, mm_now), base)  # 存成np array 格式
            base = np.full((frame.shape[0], frame.shape[1]), 0, np.uint16)  # 清空base
            mm_nxt = (mm_now + INTERVAL_MINUTES) % HOUR_IN_MINUTES

    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)
np.save("base_last", base)  # 存成np array 格式
