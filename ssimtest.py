import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt

video_name = 'soccer'

# jpg video frames to be analyzed - ordered frame0.jpg, frame1.jpg, etc.
frames_jpg_path = '../project_files/project_dataset/frames/'+video_name+'/'

frame_a = cv2.imread(frames_jpg_path+'frame10258.jpg')
frame_b = cv2.imread(frames_jpg_path+'frame10259.jpg')
frame_c = cv2.imread(frames_jpg_path+'frame10260.jpg')
frame_d = cv2.imread(frames_jpg_path+'frame10261.jpg')

hist1 = []
color = ('b','g','r')
for j in range(3):
    histr = cv2.calcHist([frame_b],[j],None,[256],[0,256])
    hist1.append(histr)

hist2 = []
for j in range(3):
    histr = cv2.calcHist([frame_c],[j],None,[256],[0,256])
    hist2.append(histr)

hist1a = np.asarray(hist1)
hist2a = np.asarray(hist2)
average_dist = 0

for j in range(3):
    dist = cv2.compareHist(hist1a[j], hist2a[j], 0)
    average_dist = average_dist + dist

average_dist = average_dist / 3.0

print(average_dist)

frame_a_bw = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
frame_b_bw = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
frame_c_bw = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
frame_d_bw = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)

# hist1 = cv.calcHist([frame_a_bw],[0],None,[256],[0,256])
# hist2 = cv.calcHist([frame_b_bw],[0],None,[256],[0,256])

# # frame_a_bw = frame_a
# # frame_b_bw = frame_b
# # frame_c_bw = frame_c
# # frame_d_bw = frame_d

# cv2.imshow('RGB Image',frame_c_bw )
# cv2.waitKey(0)

# # print(np.min(frame_c_bw))

# ssim_ab = ssim(frame_a_bw, frame_b_bw, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
# ssim_bc = ssim(frame_b_bw, frame_c_bw, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
# ssim_cd = ssim(frame_c_bw, frame_d_bw, multichannel=False, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)

ssim_ab = ssim(frame_a_bw, frame_b_bw)
ssim_bc = ssim(frame_b_bw, frame_c_bw)
ssim_cd = ssim(frame_c_bw, frame_d_bw)

ssim_ab = round(ssim_ab, 3)
ssim_bc = round(ssim_bc, 3)
ssim_cd = round(ssim_cd, 3)

# # print(ssim_ab)
# print(ssim_bc)
# # print(ssim_cd)

print(ssim_bc/ssim_ab)
print(ssim_bc/ssim_cd)