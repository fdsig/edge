import ffmpeg
stream = ffmpeg.input('/dev/camera0')
stream = ffmpeg.hflip(stream)
stream = ffmpeg.output(stream, 'output.mp4')
ffmpeg.run(stream)
