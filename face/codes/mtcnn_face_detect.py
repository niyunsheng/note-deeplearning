import cv2,time
from facenet_pytorch.models.mtcnn import MTCNN

device = "cpu:0"
detector = MTCNN(margin=0, thresholds=[0.7, 0.8, 0.8], device=device)
# detector = MTCNN(margin=0, thresholds=[0.85, 0.95, 0.95], device=device)
video_capture = cv2.VideoCapture(0)
mode = 'landmark'
WIDTH = 224
while True:
    # Capture frame-by-frame
    start = time.time()
    ret, frame = video_capture.read()
    if ret:
        rate = WIDTH/frame.shape[0]
        frame = cv2.flip(frame, 1)
        resized = cv2.resize(frame, None, fx=rate, fy=rate)
        # print('resize time:',time.time()-start)
        if mode == 'face':
            face_boxes, pred = detector.detect(resized, landmarks=False)
            # print('detect face:', time.time() - start)
            if face_boxes is not None:
                for face_box in face_boxes:
                    x0, y0, x1, y1 = [int(t/rate) for t in face_box]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        elif mode == 'landmark':
            face_boxes, pred, landmarks = detector.detect(resized, landmarks=True)
            if face_boxes is not None:
                for face_box in face_boxes:
                    x0, y0, x1, y1 = [int(t/rate) for t in face_box]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
            if landmarks is not None:
                for landmark in landmarks:
                    for circles in landmark:
                        x0, y0 = [int(t/rate) for t in circles]
                        cv2.circle(frame, (x0, y0), 2, (255, 0, 0), 0)
        cv2.imshow("Image", frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        # print('show face:', time.time() - start)

video_capture.release()
cv2.destroyAllWindows()