import cv2
import re
import os
import time

URL = "http://vrl-shrimp.cv:5000/video_feed"
video = cv2.VideoCapture(URL)

# 保存先フォルダ内の既存の画像枚数をカウントし，開始番号を決定
save_folder = "/Users/kubo/ex/python/shrimp_pool/JPEGImages/"
dir = os.chdir(save_folder)
files = os.listdir(dir)
cnt = 0
for file in files:
    index = re.search('.jpg', file)
    if index:
        cnt = cnt + 1

while True:
    ret, img = video.read()
    if not img == None:
        cv2.imshow("Stream Video",img)
    key = cv2.waitKey(1) & 0xff
    if key == ord('q'): break
    if key == ord('s') and not img == None:
        cnt_zero = str(cnt).zfill(5) #0うめ
        print(save_folder + cnt_zero + ".jpg")
        cv2.imwrite(save_folder + cnt_zero + ".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cnt = cnt + 1