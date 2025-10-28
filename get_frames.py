import cv2

video = cv2.VideoCapture(r"/object counting/org_video.mp4")


counter = 0
frame_count = 0
while video.isOpened():

    if counter == 8:
        break

    flag, frame = video.read()
    if frame_count % 100 ==0:
        cv2.imwrite("cars/frame{}.jpg".format(frame_count), frame)
        counter +=1
    frame_count += 1



video.release()