import numpy as np
import cv2
import time

def GaussainKernel(k, sigma):
    
    if k % 2 == 0:
        raise Exception('k must be an odd number')

    K = np.indices((k, k))
    
    center = (k-1) / 2
    center = np.array([[[center]], [[center]]])
    center = np.repeat(center, k, 1)
    center = np.repeat(center, k, 2)
    
    K = K - center
    K = K**2
    K = np.sum(K, axis=0)
    K = -K
    K = K / (sigma**2)
    K = np.exp(K)
    
    K = np.expand_dims(K, axis=2)
    
    return K

def RangeKernel(K, sigma):
    
    k = K.shape[0]
    center = int((k-1) / 2)
    
    if len(K.shape) == 2:
        
        center = K[center, center]
        
        K = K - center   
        K = K**2
        K = -K
        K = K / (sigma**2)
        K = np.exp(K)
    
    else:
        
        center = K[center:center+1, center:center+1]
        center = np.repeat(center, k, 1)
        center = np.repeat(center, k, 0)
        
        K = K - center
        K = K**2
        K = np.sum(K, axis=2)
        K = -K
        K = K / (sigma**2)
        K = np.exp(K)
        
    K = np.expand_dims(K, axis=2)
    
    return K

def JBF(costvolume, I, k, sigma_s, sigma_r):
    
    H, W, C = costvolume.shape
    costvolume_filtered = np.zeros(costvolume.shape)
    
    r = int((k-1) / 2)
    costvolume = np.pad(costvolume, [(r, r), (r, r), (0, 0)], 'reflect')
    if len(I.shape) == 2:
        I = np.pad(I, [(r, r), (r, r)], 'reflect')
    else:    
        I = np.pad(I, [(r, r), (r, r), (0, 0)], 'reflect')

    Ks = GaussainKernel(k, sigma_s)
    
    for h in range(H):
        for w in range(W):
            
            h_ = h + r
            w_ = w + r
            
            Kr = RangeKernel(I[h_-r:h_+r+1, w_-r:w_+r+1], sigma_r)
            K = Ks * Kr
            K = np.repeat(K, C, axis=2)
            sum_ = np.sum(K, axis=1)
            sum_ = np.sum(sum_, axis=0)
                   
            cost = costvolume[h_-r:h_+r+1, w_-r:w_+r+1] * K
            cost = np.sum(cost, axis=1)
            cost = np.sum(cost, axis=0)
            cost = cost / sum_
            
            costvolume_filtered[h][w]  = cost
        
    return costvolume_filtered

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    
    Il = Il.astype('float32')
    Ir = Ir.astype('float32')
    Il = Il / 255
    Ir = Ir / 255
    
    Il_dx = Il.copy()
    Il_dx[:, 1:] =  Il_dx[:, 1:] - Il[:, :w-1]
    Ir_dx = Ir.copy()
    Ir_dx[:, 1:] =  Ir_dx[:, 1:] - Ir[:, :w-1]
    
    cost_ls = []
    
    # >>> Cost computation
    tic = time.time()
    
    # TODO: Compute matching cost from Il and Ir
    for d in range(1, max_disp+1):
        cost1 = Il.copy()
        Ir_ = Ir[:, :w-d, :]
        cost1[:, d:, :] = cost1[:, d:, :] - Ir_
        cost1 = cost1**2
        cost1 = np.sum(cost1, axis=2)
        
        cost2 = Il_dx.copy()
        cost2[:, d:, :] = cost2[:, d:, :] - Ir_dx[:, :w-d, :]
        cost2= cost2**2
        cost2 = np.sum(cost2, axis=2)
        
        cost_ls.append(0.1*cost1 + 0.*cost2)
        
    costvolume = np.stack(cost_ls, axis=2)
        
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    
    # TODO: Refine cost by aggregate nearby costs
    costvolume_filtered = JBF(costvolume, Il, 19, 9, 0.1)
    
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

    # >>> Disparity optimization
    tic = time.time()
    
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    labels = np.argmin(costvolume_filtered, axis=2)
    labels = labels + 1
     
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

    return labels


def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))

    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
