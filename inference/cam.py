import cv2
import subprocess as sp
import numpy as np

import platform
import time
from pathlib import Path
pwd = Path('.')
IN_IMG = pwd/'cv_out.jpg'

IMG_W = 1280
IMG_H = 720
#sudo ffmpeg -y -f v4l2 -video_size 1280x720 -i /dev/video0      -r 5 -qscale:v 2 -update 1 out.jpg
i = 0 
if platform.system()=='Linux':
    FFMPEG_BIN = "/usr/bin/ffmpeg"
    ffmpeg_cmd = ['sudo', FFMPEG_BIN,'-video_size',f'1280x72{i}', '-i', '/dev/video0','-r', '10','-pix_fmt', 'bgr24','-vcodec', 'rawvideo','-an','-sn','-f', 'image2pipe', '-']    
#ffmpeg_cmd = ['sudo',FFMPEG_BIN, '-y','-f','v4l2','-video_size','1280x720','-i','/dev/video0','-r','1','-qscale:v','2','-update','1','out.jpg']
#sp.run(ffmpeg_cmd, capture_output=True)
with sp.Popen(ffmpeg_cmd, stdout = sp.PIPE,stderr = sp.PIPE,bufsize=10000) as pipe:
    while True:
        # Capture frame-by-frame
        out  = pipe.stdout.read(IMG_H*IMG_W*3)
        #out, err =  pipe.communicate()
        img  = np.frombuffer(out, dtype='uint8').reshape((IMG_H,IMG_W,3))
        i+=1
        #time.sleep(1)
        # transform the byte read into a numpy array
        #image = numpy.frombuffer(raw_image, dtype='uint8')
        img  = img.reshape((IMG_H,IMG_W,3))
        # Notice how height is specified first and then width
        cv2.imwrite(f'ims/cv_{i}out.jpg',img)
        if IN_IMG.exists():
            #print(image)
            # img = cv2.imread('out.jpg')
            #print(type(img))
            try:
                #print(img)
                img = cv2.resize(img,(IMG_H,IMG_W),interpolation = cv2.INTER_AREA)
                cv2.imwrite('cv_out.jpg',img)
            except:
                print('none')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        pipe.poll()
        pipe.stdout.flush()

#while True:
#        cv2_stream = cv2.VideoCapture(f'pipe:{child_proc.stdout.fileno()}')
        #raw_image = pipe.stdout.read(IMG_W*IMG_H*3)
        #image =  numpy.frombuffer(raw_image, dtype='uint8')
        #image = image.reshape((IMG_H,IMG_W,3))
        #print(image)
        #cv2.imwrite('vid.jpg', image)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break
#        pipe.stdout.flush()
#else:
#    print('not Linux')
