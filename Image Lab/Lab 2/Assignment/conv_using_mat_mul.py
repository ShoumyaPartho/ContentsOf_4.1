
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz


def conv_with_mat_mul(img, kernel2):
    img = cv2.resize(img, (150,150))
    img_height = img.shape[0]
    img_width = img.shape[1]
    ker_height = kernel2.shape[0]
    ker_width = kernel2.shape[1]
    #output=np.zeros((img.shape),np.float32)
    out_height = img_height+ker_height-1
    out_width = img_width+ker_width-1
    
    pad_ker = np.pad(kernel2, ((out_height-ker_height,0), (0, out_width-ker_width)), 'constant', constant_values = 0)
    
    toeplitz_list = []
    for i in range(pad_ker.shape[0]-1, -1, -1):
        col = pad_ker[i,:]
        row = np.r_[col[0], np.zeros(img_width-1, np.float32)]
        temp_toeplitz = toeplitz(col, row)
        toeplitz_list.append(temp_toeplitz)
    
    col = range(1,pad_ker.shape[0]+1)
    row = np.r_[col[0], np.zeros(img_height-1, dtype = int)]
    doublyblocked_indices = toeplitz(col, row)
    
    t_height = pad_ker.shape[1]
    t_width = img_width
    t_block_height = t_height*doublyblocked_indices.shape[0]
    t_block_width = t_width*doublyblocked_indices.shape[1]
    t_block = np.zeros((t_block_height, t_block_width), np.float32)
    
    for i in range(doublyblocked_indices.shape[0]):
        for j in range(doublyblocked_indices.shape[1]):
            start_height = t_height*i
            start_width = t_width*j
            end_height = start_height + t_height
            end_width = start_width + t_width
            t_block[start_height:end_height, start_width:end_width] = toeplitz_list[doublyblocked_indices[i][j]-1]

    out_vector = np.zeros(img.shape[0]*img.shape[1], np.float32)
    
    flipped_img = img[::-1]

    out_vector = flipped_img.reshape(-1,1)
    
    result_vector = np.matmul(t_block, out_vector)
    
    result = np.zeros((out_height, out_width), np.float32)
    for i in range(result.shape[0]):
        start = out_width*i
        end = start + out_width
        result[i,:] = result_vector[start:end,0]
    result = result[::-1]

    cv2.imshow("Input", img)
    cv2.imshow("Result",result)
    
    plt.imshow(img, 'gray')
    plt.title("Input")
    plt.show()
    
    plt.imshow(result, 'gray')
    plt.title("Output")
    plt.show()


    
    
    

img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
ker_width = int(input('Width of kernel: '))
ker_height = int(input('Height of kernel: '))

kernel2 = np.ones((ker_width, ker_height), dtype = np.float32)
kernel2 /= ker_height*ker_width

conv_with_mat_mul(img, kernel2)

cv2.waitKey(0)
cv2.destroyAllWindows()

