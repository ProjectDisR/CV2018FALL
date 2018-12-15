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
    
    
    # >>> Cost computation
    tic = time.time()
    
    # TODO: Compute matching cost from Il and Ir
    
    cost_ls = []
    
    for d in range(1, max_disp+1):
        cost1 = Il.copy()
        Ir_ = Ir[:, :w-d, :]
        cost1[:, d:, :] = cost1[:, d:, :] - Ir_
        cost1 = cost1**2
        cost1 = np.sum(cost1, axis=2)

        cost_ls.append(cost1)
        
    costvolume_l = np.stack(cost_ls, axis=2)
    
    cost_ls = []
    
    for d in range(1, max_disp+1):
        cost1 = Ir.copy()
        Il_ = Il[:, d:, :]
        cost1[:, :w-d, :] = cost1[:, :w-d, :] - Il_
        cost1 = cost1**2
        cost1 = np.sum(cost1, axis=2)
        
        cost_ls.append(cost1)
        
    costvolume_r = np.stack(cost_ls, axis=2)
        
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))






    # >>> Cost aggregation
    tic = time.time()
    
    # TODO: Refine cost by aggregate nearby costs
    
    costvolume_filtered_l = JBF(costvolume_l, Il, 19, 17, 0.12)
    costvolume_filtered_r = JBF(costvolume_r, Ir, 19, 17, 0.12)
    
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))






    # >>> Disparity optimization
    tic = time.time()
    
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    
    labels_l = np.argmin(costvolume_filtered_l, axis=2)
    labels_l = labels_l + 1
    labels_r = np.argmin(costvolume_filtered_r, axis=2)
    labels_r = labels_r + 1
     
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))







    # >>> Disparity refinement
    tic = time.time()
    
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    
    occluded = np.full((h, w), True)
    for i in range(h):
        for j in range(w):
            
            if int(j-labels_l[i, j]) < 0:
                continue
            
            if labels_l[i, j] == labels_r[i][int(j-labels_l[i, j])]:
                occluded[i, j] = False
    
    for i in range(h):
        not_occluded = np.where(occluded[i] == False)[0]
        for j in range(w):
            if occluded[i, j]:
                labels_l[i, j] = labels_l[i, not_occluded[np.argmin(np.abs(not_occluded-j))]]
    
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    
    
    labels_l = labels_l.astype('uint8')
    return cv2.medianBlur(labels_l, 7)


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
