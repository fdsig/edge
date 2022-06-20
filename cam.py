import cv2
import subprocess as sp
import numpy
import time
IMG_W = 64
IMG_H = 64

FFMPEG_BIN = "/usr/bin/ffmpeg"
ffmpeg_cmd = ['sudo', FFMPEG_BIN,'-i', '/dev/video0','-r', '10','-pix_fmt', 'bgr24','-vcodec', 'rawvideo','-an','-sn','-f', 'image2pipe', '-']    
pipe = sp.Popen(ffmpeg_cmd, stdout = sp.PIPE, bufsize=1)
while True:
    time.sleep(1)
    raw_image = pipe.stdout.read(IMG_W*IMG_H*3)
    image =  numpy.frombuffer(raw_image, dtype='uint8')
    image = image.reshape((IMG_H,IMG_W,3))
    print(image)
    cv2.imwrite('vid.jpg', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pipe.stdout.flush()
