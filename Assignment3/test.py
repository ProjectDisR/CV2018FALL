import numpy as np
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    A = np.zeros((2*N, 9))

    for i in range(0, len(u)):
        x = u[i][0]
        y = u[i][1]

        x_ = v[i][0]
        y_ = v[i][1]

        A[2*i] = [x, y, 1, 0, 0, 0, -x_*x, -x_*y, -x_]
        A[2*i + 1] = [0, 0, 0, x, y, 1, -y_*x, -y_*y, -y_]

    U, s, V = np.linalg.svd(A, full_matrices=True)
    H = V[-1, :].reshape(3, 3)
    
    return H


def transform(img, canvas, corners):
    h, w, ch = img.shape
    canvas = canvas.astype(np.float)
    
    img_t = np.zeros(canvas.shape)

    indices = np.indices((h, w))
    indices = indices.reshape(2, -1)
    indices = np.roll(indices, 1, axis=0)
    ones = np.ones((1, indices.shape[1]))
    indices = np.concatenate((indices, ones), axis=0)
    
    H = solve_homography(getcorners(img), corners)
    
    indices_t = np.matmul(H, indices)
    indices_t[0] = indices_t[0] / indices_t[2]
    indices_t[1] = indices_t[1] / indices_t[2]
    indices_t[0] = np.clip(indices_t[0], 0, canvas.shape[1]-1)
    indices_t[1] = np.clip(indices_t[1], 0, canvas.shape[0]-1)
    indices_t = indices_t.astype(np.int)
    indices_t = np.roll(indices_t[:2], 1, axis=0).reshape(2, h, w)
    
    img_t[indices_t[0], indices_t[1]] = img
    mask = np.sum(img_t, axis=2)>1
    mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, 3))
    
    canvas = mask*img_t + (1-mask)*canvas
    
    return canvas

def inversetransform(img, corners, shape):
    
    img = img.astype(np.float)
    h = shape[0]
    w = shape[1] 
    corners_t = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    
    indices_t = np.indices((h, w))
    indices_t = indices_t.reshape(2, -1)
    indices_t = np.roll(indices_t, 1, axis=0)
    ones = np.ones((1, indices_t.shape[1]))
    indices_t = np.concatenate((indices_t, ones), axis=0)
    
    H_inverse = solve_homography(corners_t, corners)
    
    indices = np.matmul(H_inverse, indices_t)
    indices[0] = indices[0] / indices[2]
    indices[1] = indices[1] / indices[2]
    indices[0] = np.clip(indices[0], 0, img.shape[1]-1)
    indices[1] = np.clip(indices[1], 0, img.shape[0]-1)
    indices = np.roll(indices[:2], 1, axis=0).reshape(2, h, w)
    
    
    ul = (np.ceil(indices[0])-indices[0])*(np.ceil(indices[1])-indices[1])
    ur = (np.ceil(indices[0])-indices[0])*(indices[1]-np.floor(indices[1]))
    bl = (indices[0]-np.floor(indices[0]))*(np.ceil(indices[1])-indices[1])
    br = (indices[0]-np.floor(indices[0]))*(indices[1]-np.floor(indices[1]))
    
    ul = np.tile(np.expand_dims(ul, axis=2), (1, 1, 3))
    ur = np.tile(np.expand_dims(ur, axis=2), (1, 1, 3))
    bl = np.tile(np.expand_dims(bl, axis=2), (1, 1, 3))
    br = np.tile(np.expand_dims(br, axis=2), (1, 1, 3))
    
    
    img_ul = img[np.floor(indices[0]).astype(np.int), np.floor(indices[1]).astype(np.int)]
    img_ur = img[np.floor(indices[0]).astype(np.int), np.ceil(indices[1]).astype(np.int)]
    img_bl = img[np.ceil(indices[0]).astype(np.int), np.floor(indices[1]).astype(np.int)]
    img_br = img[np.ceil(indices[0]).astype(np.int), np.ceil(indices[1]).astype(np.int)]
    
    return img_ul*ul+img_ur*ur+img_bl*bl+img_br*br

def getcorners(img):
    
    m = img.shape[0]
    n = img.shape[1]
    
    return np.array([[0, 0], [n-1, 0], [0, m-1], [n-1, m-1]])


# Part 1
canvas = cv2.imread('./input/times_square.jpg')
img1 = cv2.imread('./input/wu.jpg')
img2 = cv2.imread('./input/ding.jpg')
img3 = cv2.imread('./input/yao.jpg')
img4 = cv2.imread('./input/kp.jpg')
img5 = cv2.imread('./input/lee.jpg')

corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]])
corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])

canvas_t = transform(img1, canvas, corners1)
canvas_t = transform(img2, canvas_t, corners2)
canvas_t = transform(img3, canvas_t, corners3)
canvas_t = transform(img4, canvas_t, corners4)
canvas_t = transform(img5, canvas_t, corners5)

cv2.imwrite('canvas_t.jpg', canvas_t)



#    # TODO: some magic
#    cv2.imwrite('part1.png', canvas)
#
    # Part 2
img = cv2.imread('./input/screen.jpg')

corners = np.array([[1038, 368], [1100, 395], [982, 553], [1036, 600]])
qrcode = inversetransform(img, corners, (150, 150))
cv2.imwrite('qrcode.jpg', qrcode)

#    # Part 3
#    img_front = cv2.imread('./input/crosswalk_front.jpg')
#    # TODO: some magic
#    # cv2.imwrite('part3.png', output3)
