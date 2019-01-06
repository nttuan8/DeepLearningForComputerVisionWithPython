# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 11:06:45 2019

@author: DELL
"""

import requests
import os
import time

#url = 'http://webproxy.to/browse.php/gBqQf2ab/8rvz7_2B/YDcBkCGu/14iKivOv/lEvqV1yy/AV8_2BwI/i3bi6im2/Rw_3D_3D/b5/fnorefer'
url = 'https://www.e-zpassny.com/vector/jcaptcha.do'
numOfImage = 1
total = 0

for i in range(0, numOfImage):
    try:
        r = requests.get(url, timeout=60)
        p = os.path.join('Download', '{}.jpg'.format(str(total).zfill(5)))
        f = open(p, 'wb')
        print(r)
        f.write(r.content)
        f.close()
        
        print('[INFO] Dowloaded {}'.format(p))
        total += 1
    except:
        print('[ERROR] Download image')
    time.sleep(0.1)
