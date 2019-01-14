import numpy as np
import re
import os

def ERR(path):
    o_img = "TL"
    Ave_err = 0
    img_num = 10
    for i in range(img_num):
        disp = readPFM(os.path.join(path, o_img + str(i) + ".pfm"))
        GT = readPFM(os.path.join(path, o_img + "D" + str(i) + ".pfm"))
        err = cal_avgerr(GT, disp)
        print( (o_img + str(i) + " err: ") ,err)
        Ave_err += err
    print("Average err: ", Ave_err/img_num)
        
    return Ave_err/img_num


def cal_avgerr(GT, disp):
    return np.sum(np.multiply(np.abs(GT - disp), GT[GT != np.inf].reshape(GT.shape))) / np.sum(GT[GT != np.inf])

def readPFM(file):
    file = open(file, 'rb')

    header = file.readline().rstrip()
    header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data