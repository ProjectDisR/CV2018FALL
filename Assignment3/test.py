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
#    A = np.zeros((2*N, 8))
#	# if you take solution 2:
#	# A = np.zeros((2*N, 9))
#    b = np.zeros((2*N, 1))
#    H = np.zeros((3, 3))
#    # TODO: compute H from A and b
        
    H = cv2.findHomography(u, v)
    
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    # TODO: some magic

def corners(img):
    m = img.shape[0]
    n = img.shape[1]
    
    return np.array([[0, 0], [m-1, 0], [0, n-1], [m-1, n-1]])


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

H, _ = cv2.findHomography(corners(img1), corners1)
a = cv2.warpPerspective(img1, H, (canvas.shape[1], canvas.shape[0]))
cv2.imwrite('a.jpg', a)
mask = np.array(a) > 1
mask = mask.astype(np.float32)
print(np.sum(mask))
b = mask*a + (1-mask)*canvas
cv2.imwrite('b.jpg', b)



#    # TODO: some magic
#    cv2.imwrite('part1.png', canvas)
#
    # Part 2
img = cv2.imread('./input/screen.jpg')

cornerss = np.array([[1039, 368], [982, 552], [1102, 395], [1034, 559]])
H, _ = cv2.findHomography(cornerss, corners(img))
a = cv2.warpPerspective(img, H, (img.shape[1], img.shape[0]))
cv2.imwrite('a.jpg', a)

#    # Part 3
#    img_front = cv2.imread('./input/crosswalk_front.jpg')
#    # TODO: some magic
#    # cv2.imwrite('part3.png', output3)
