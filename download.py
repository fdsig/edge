import zipfile 
from pathlib import Path

import sys
from time import sleep 
from sys import stdout
from urllib import response
import requests 
import gdown
import tqdm 

class get_dataset: 

    def __init__(self,fid:str, url:str):
        self.fid = fid
        self.url = url

    def parse_google_url(self,fid=None):
        with open(fid,'r') as handle:
            fid_list = handle.readlines()
        self.fid_list =  fid_list

    def download_from_txt(self):
        parsed = self.parse_google_url(self.fid)
        data = sys.stdin.readlines()
        outputs = [f'batch_{b_num}.zip' for b_num in range(len(data))]
        for line,output in zip(data,outputs):
            line = line.strip(',')
            line = line.split('/')
            id = line[5]
            sys.stdout.write(str(line[5]))
            gdown.download(id=id, output=output, quiet=False)

    def get_zip(self):
        '''gets zipfile of data from url'''
        resp = requests.get(self.url,stream=True)
        file = self.url.split('/')[-1]
        with open(file, 'wb') as hndl:
            for data in tqdm(resp.iter_content()):
                hndl.write(data)

    def unzip(self,out_dir:str):
        '''checks if images dir exists
        makes if not, then infaltes/decompreses to directory'''
        out, fid = Path(out_dir), Path(self.fid)
        if not out.is_dir():
            out.mkdir()
        with zipfile.ZipExtFile(out,'r') as hndl:
            hndl.extractall(out)

    




 