import os
import sys
from time import sleep 
from sys import stdout
import gdown
import tqdm 
def parse_google_url(fid=None):
    with open(fid,'r') as handle:
        fid_list = handle.readlines()
    return fid_list
parsed = parse_google_url(fid='files.txt')

data = sys.stdin.readlines()
outputs = [f'batch_{b_num}.zip' for b_num in range(len(data))]
for line,output in zip(data,outputs):
    line = line.strip(',')
    line = line.split('/')
    id = line[5]
    sys.stdout.write(str(line[5]))
    gdown.download(id=id, output=output, quiet=False)


 