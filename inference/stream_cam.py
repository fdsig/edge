
import numpy as np
import subprocess
import platform 

t_w,t_h = 224, 224 
num_frame = 1 
i = 0
THREAD_NUM = 4 
fileloc = '/dev/camera0'

if platform.system()=='Linux':
    FFMPEG_BIN = "/usr/bin/ffmpeg"
ffmpeg_cmd = ['sudo',
        FFMPEG_BIN,
        '-video_size', f'1280x72{i}',
        '-i', '/dev/video0',
        '-r', '10',
        '-pix_fmt', 
        'bgr24',
        '-vcodec', 'rawvideo',
        '-an','-sn',
        '-f', 
        'image2pipe', '-']  
command = ['ffmpeg',
               '-loglevel', 'fatal',
               #'-ss', str(datetime.timedelta(seconds=frame/fps)),
               '-i', fileloc,
               #'-vf', '"select=gte(n,%d)"'%(frame),
               '-threads', str(THREAD_NUM),
               '-vf', 'scale=%d:%d'%(t_w,t_h),
               '-vframes', str(num_frame),
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']
    #print(command)
ffmpeg = subprocess.Popen(ffmpeg_cmd, stderr=subprocess.PIPE ,stdout = subprocess.PIPE )
out, err = ffmpeg.communicate()
print(out)
if(err) : print('error',err)
video = np.frombuffer(out, dtype='uint8').reshape((t_h,t_w,3))
print(video)
