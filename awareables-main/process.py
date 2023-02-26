import numpy as np
import cv2
from imutils.object_detection import non_max_suppression 
import time

def preProcess(img, outdir, setting):
    # original image 
    # braille.jpg should be replaced with the file path from camera capture
    img = cv2.imread(img)
    img1 = img 
    #cv2.imshow('img',img)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    ##cv2.imshow('gray',gray)

    # Compute the background (use a large filter radius for excluding the dots)
    bg = cv2.medianBlur(gray, 151)  
    ##cv2.imshow('bg', bg)

    # Compute absolute difference 
    fg = cv2.absdiff(gray, bg)  
    ##cv2.imshow('fg',fg)

    #can use either "blur" or "fg" for threshold input parameter 
    blur = cv2.medianBlur(img, 11)
    blur2 = cv2.GaussianBlur(gray, (7,7), 0)
    ##cv2.imshow('blur',blur)
    ##cv2.imshow('Gaussian blur', blur2)

    # Attempting to find best-fitting parameters for Gaussian Adaptive Threshold
    # last two input parameters are :
    # 1) Size of a pixel neighborhood that is used to calculate a threshold value 
    # for the pixel: 3, 5, 7, and so on.
    # 2) Constant subtracted from the mean or weighted mean

    thresh = cv2.adaptiveThreshold(blur2, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)

    thresh2 = cv2.adaptiveThreshold(blur2, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 4)

    thresh3 = cv2.adaptiveThreshold(blur2, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)

    thresh4 = cv2.adaptiveThreshold(blur2, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 8)

    thresh5 = cv2.adaptiveThreshold(blur2, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 6)

    thresh6 = cv2.adaptiveThreshold(blur2, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)

    thresh7 = cv2.adaptiveThreshold(blur2, 255,
	    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 17, 6)

    #cv2.imshow("Gaussian Adaptive Thresholding 1", thresh)
    #cv2.imshow("test2", thresh2)
    #cv2.imshow("test3", thresh3)
    #cv2.imshow("test4", thresh4)
    #cv2.imshow("test5", thresh5)
    #cv2.imshow("test6", thresh6)
    #cv2.imshow("test7", thresh7)

    # Apply Canny edge detection.
    edges = cv2.Canny(gray, threshold1=50, threshold2=100) 
    edges2 = cv2.Canny(gray, threshold1=100, threshold2=200)
    edges3 = cv2.Canny(gray, threshold1=50, threshold2=200) 

    #cv2.imshow('edges',edges)
    # Merge edges with thresh
    res = cv2.bitwise_or(thresh, edges)  
    res2 = cv2.bitwise_or(thresh, edges)  
    res3 = cv2.bitwise_or(thresh, edges)  

    #cv2.imshow('res',res)
    #cv2.imshow('res2',res2)
    #cv2.imshow('res3',res3)

    # kernel matrix that includes size of the mask (2,2), for erosion and dilation
    kernel = np.ones((2,2), np.uint8)
    dkernel = np.ones((3,3),np.uint8)
    # Erosion to get rid of small dots noises 
    erosion = cv2.erode(thresh, kernel, iterations=1)
    erosion2 = cv2.erode(thresh, kernel, iterations=2)
    erosion_canny = cv2.erode(res, kernel, iterations=1)
    erosion2_canny = cv2.erode(res, kernel, iterations=2)
    #cv2.imshow('erosion-thresh',erosion)
    #cv2.imshow('erosion2-thresh',erosion2)
    #cv2.imshow('erosion-canny',erosion_canny)
    #cv2.imshow('erosion2-canny',erosion2_canny)

    # Dilation to expand the eroded dots to reasonable sizes 
    dilation = cv2.dilate(erosion, dkernel, iterations = 1)
    dilation2 = cv2.dilate(erosion2, dkernel, iterations = 2)
    dilation_canny = cv2.erode(erosion_canny, dkernel, iterations=1)
    dilation2_canny = cv2.erode(erosion2_canny, dkernel, iterations=2)

    k_dilation = cv2.dilate(erosion, kernel, iterations = 1)
    k_dilation2 = cv2.dilate(erosion2, kernel, iterations = 2)
    k_dilation_canny = cv2.erode(erosion_canny, kernel, iterations=1)
    k_dilation2_canny = cv2.erode(erosion2_canny, kernel, iterations=2)

    #cv2.imshow('dilation-thresh',dilation)
    #cv2.imshow('dilation2-thresh',dilation2)
    #cv2.imshow('dilation-canny',dilation_canny)
    #cv2.imshow('dilation2-canny',dilation2_canny)

    #cv2.imshow('(2,2)dilation-thresh',k_dilation)
    #cv2.imshow('(2,2)dilation2-thresh',k_dilation2)
    #cv2.imshow('(2,2)dilation-canny',k_dilation_canny)
    #cv2.imshow('(2,2)dilation2-canny',k_dilation2_canny)

    # Invert the image for white background & block dots 
    final1 = cv2.bitwise_not(dilation)
    final2 = cv2.bitwise_not(k_dilation)
    #cv2.imshow('final1', final1)
    #cv2.imshow('final2', final2)
    temp_final = final1
    img2 = final1
    # non-max suppression for drawing colored circles on top of the final mask
    # specify connectivity (4 or 8, required for cv2.connectComponentWithStats())
    connectivity = 4 
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)

    # first cell represents # of labels
    number_label = output[0]
    # second cell represents label matrix
    # matrix_label is a matrix the size of the input image where each element 
    # has a value equal to its label. 
    matrix_label = output[1]
    # third cell represents stat matrix - so this information will be pulled 
    # matrix_stat is a matrix of the stats that the function calculates. It has 
    # a length equal to the number of labels and a width equal to the # of stats
    matrix_stat = output[2]
    #print("#printing matrix_stat")
    #print(matrix_stat)
    ##print(matrix_stat[0]) # this cell is useless data 
    stats = matrix_stat[1:]
    #print("#printing useless data-less stats")
    #print(stats)

    # fourth cell represents centroid matrix
    matrix_centroid = output[3]

    # stats is comprised of five component arrays: 1) cv2.CC_STAT_LEFT,
    # 2) cv2.CC_STAT_TOP, 3) cv2.CC_STAT_WIDTH, 4) cv2.CC_STAT_HEIGHT, and
    # 5) cv2.CC_STAT_AREA
    stats = matrix_stat[1:] # this gets rid of the first row of the matrix (dummy data)
    filtered_stats = []
    for x, y, w, h, pixels in stats:
        if pixels < 1000 and x > 10 and y > 10 and pixels > 0:
            filtered_stats.append((x, y, x+w, y+h))

    #print(stats[0])
    #print(stats[1])
    #print(stats[2])
    #print(stats[3])
    #print(stats[4])

    #print("#printing filtered_stats")
    #print(filtered_stats)

    filtered_stats = non_max_suppression(np.array(filtered_stats), overlapThresh= 0.2)
    ##print("#printing filtered_stats after nms")
    ##print(filtered_stats)

    temp2 = final2
    for x1, y1, x2, y2 in filtered_stats:
        # cv2.circle(image, center_coordinates, radius, color, thickness)
	    #(0,255,0)=green
        cv2.circle(img1, ((x1+x2)//2, (y1+y2)//2), 4, (0, 255, 0), -1)

	    #cv2.circle(temp_final, ((x1+x2)//2, (y1+y2)//2), 6, (0, 255, 0), -1)
	    #cv2.circle(final2, ((x1+x2)//2, (y1+y2)//2), 6, (0, 255, 0), -1)

    for x1, y1, x2, y2 in filtered_stats:
        cv2.circle(temp_final, ((x1+x2)//2, (y1+y2)//2), 4, (0, 255, 0), -1)

    #cv2.imshow('final',img1)
    #cv2.imshow('temp_final',temp_final)
    ##cv2.imshow('final3',final2)

    cv2.imwrite('final.jpg',img1)

    #getting the dimensions of image
    dimensions = img1.shape

    # getting height, width, number of channels in image
    height = img1.shape[0]
    width = img1.shape[1]
    # dont need #channels = img1.shape[2]

    #print("#printing new stuffs below")
    #print("dimensions,height,width,channel in order")
    #print(dimensions)
    #print(height)
    #print(width)
    ##print(channels)

    #getting the dimensions of image
    dimensions = img1.shape

    # getting height, width, number of channels in image
    height = img1.shape[0]
    width = img1.shape[1]
    channels = img1.shape[2]

    # these numbers need to be adaptive (needs change)
    horizontal_braille_count = 40
    vertical_braille_count = 14

    horizontal_margin = 23
    vertical_margin = 50

    crop_width = (width - (horizontal_margin * 2)) / horizontal_braille_count
    crop_height = (height - (vertical_margin * 2)) / vertical_braille_count
    #print("#printing crop w&h")
    #print(crop_width)
    #print(crop_height)

    #[height, width]
    #row1
    cropped_1 = temp_final[30:70, 28:53]
    cropped_2 = temp_final[30:70, 53:78]
    cropped_3 = temp_final[30:70, 78:103]
    cropped_4 = temp_final[30:70, 103:128]
    cropped_5 = temp_final[30:70, 128:153]
    cropped_6 = temp_final[30:70, 153:178]
    cropped_7 = temp_final[30:70, 175:200]
    cropped_8 = temp_final[30:70, 200:225]
    cropped_9 = temp_final[30:70, 223:248]
    cropped_10 = temp_final[30:70, 248:273]
    cropped_11 = temp_final[30:70, 272:296]
    cropped_12 = temp_final[30:70, 295:320]
    cropped_13 = temp_final[30:70, 320:344]
    cropped_14 = temp_final[30:70, 343:368]
    cropped_15 = temp_final[30:70, 367:391]
    cropped_16 = temp_final[30:70, 391:415]
    cropped_17 = temp_final[30:70, 415:439]
    cropped_18 = temp_final[30:70, 439:463]
    cropped_19 = temp_final[30:70, 462:486]
    cropped_20 = temp_final[30:70, 486:510]
    cropped_21 = temp_final[30:70, 510:533]
    cropped_22 = temp_final[30:70, 532:556]
    cropped_23 = temp_final[30:70, 557:579]
    cropped_24 = temp_final[30:70, 579:603]
    cropped_25 = temp_final[30:70, 603:625]
    cropped_26 = temp_final[30:70, 625:648]
    cropped_27 = temp_final[30:70, 648:672]
    cropped_28 = temp_final[30:70, 672:696]
    cropped_29 = temp_final[30:70, 695:719]
    cropped_30 = temp_final[30:70, 719:743]
    cropped_31 = temp_final[30:70, 742:766]
    cropped_32 = temp_final[30:70, 766:790]
    cropped_33 = temp_final[30:70, 789:813]
    cropped_34 = temp_final[30:70, 813:837]
    cropped_35 = temp_final[30:70, 837:861]
    cropped_36 = temp_final[30:70, 860:884]
    cropped_37 = temp_final[30:70, 884:908]
    cropped_38 = temp_final[30:70, 908:932]
    cropped_39 = temp_final[30:70, 932:956]
    cropped_40 = temp_final[30:70, 956:980]
    #row 2
    cropped_41 = temp_final[70:110, 28:53]
    cropped_42 = temp_final[70:110, 53:78]
    cropped_43 = temp_final[70:110, 78:103]
    cropped_44 = temp_final[70:110, 103:128]
    cropped_45 = temp_final[70:110, 128:153]
    cropped_46 = temp_final[70:110, 153:178]
    cropped_47 = temp_final[70:110, 175:200]
    cropped_48 = temp_final[70:110, 200:225]
    cropped_49 = temp_final[70:110, 223:248]
    cropped_50 = temp_final[70:110, 248:273]
    cropped_51 = temp_final[70:110, 272:296]
    cropped_52 = temp_final[70:110, 295:320]
    cropped_53 = temp_final[70:110, 320:344]
    cropped_54 = temp_final[70:110, 343:368]
    cropped_55 = temp_final[70:110, 367:391]
    cropped_56 = temp_final[70:110, 391:415]
    cropped_57 = temp_final[70:110, 415:439]
    cropped_58 = temp_final[70:110, 439:463]
    cropped_59 = temp_final[70:110, 462:486]
    cropped_60 = temp_final[70:110, 486:510]
    cropped_61 = temp_final[70:110, 510:533]
    cropped_62 = temp_final[70:110, 532:556]
    cropped_63 = temp_final[70:110, 557:579]
    cropped_64 = temp_final[70:110, 579:603]
    cropped_65 = temp_final[70:110, 603:625]
    cropped_66 = temp_final[70:110, 625:648]
    cropped_67 = temp_final[70:110, 648:672]
    cropped_68 = temp_final[70:110, 672:696]
    cropped_69 = temp_final[70:110, 695:719]
    cropped_70 = temp_final[70:110, 719:743]
    cropped_71 = temp_final[70:110, 742:766]
    cropped_72 = temp_final[70:110, 766:790]
    cropped_73 = temp_final[70:110, 789:813]
    cropped_74 = temp_final[70:110, 813:837]
    cropped_75 = temp_final[70:110, 837:861]
    cropped_76 = temp_final[70:110, 860:884]
    cropped_77 = temp_final[70:110, 884:908]
    cropped_78 = temp_final[70:110, 908:932]
    cropped_79 = temp_final[70:110, 932:956]
    cropped_80 = temp_final[70:110, 956:980]

    #row 3
    cropped_81 = temp_final[110:150, 28:53]
    cropped_82 = temp_final[110:150, 53:78]
    cropped_83 = temp_final[110:150, 78:103]
    cropped_84 = temp_final[110:150, 103:128]
    cropped_85 = temp_final[110:150, 128:153]
    cropped_86 = temp_final[110:150, 153:178]
    cropped_87 = temp_final[110:150, 175:200]
    cropped_88 = temp_final[110:150, 200:225]
    cropped_89 = temp_final[110:150, 223:248]
    cropped_90 = temp_final[110:150, 248:273]
    cropped_91 = temp_final[110:150, 272:296]
    cropped_92 = temp_final[110:150, 295:320]
    cropped_93 = temp_final[110:150, 320:344]
    cropped_94 = temp_final[110:150, 343:368]
    cropped_95 = temp_final[110:150, 367:391]
    cropped_96 = temp_final[110:150, 391:415]
    cropped_97 = temp_final[110:150, 415:439]
    cropped_98 = temp_final[110:150, 439:463]
    cropped_99 = temp_final[110:150, 462:486]
    cropped_100 = temp_final[110:150, 486:510]
    cropped_101 = temp_final[110:150, 510:533]
    cropped_102 = temp_final[110:150, 532:556]
    cropped_103 = temp_final[110:150, 557:579]
    cropped_104 = temp_final[110:150, 579:603]
    cropped_105 = temp_final[110:150, 603:625]
    cropped_106 = temp_final[110:150, 625:648]
    cropped_107 = temp_final[110:150, 648:672]
    cropped_108 = temp_final[110:150, 672:696]
    cropped_109 = temp_final[110:150, 695:719]
    cropped_110 = temp_final[110:150, 719:743]
    cropped_111 = temp_final[110:150, 742:766]
    cropped_112 = temp_final[110:150, 766:790]
    cropped_113 = temp_final[110:150, 789:813]
    cropped_114 = temp_final[110:150, 813:837]
    cropped_115 = temp_final[110:150, 837:861]
    cropped_116 = temp_final[110:150, 860:884]
    cropped_117 = temp_final[110:150, 884:908]
    cropped_118 = temp_final[110:150, 908:932]
    cropped_119 = temp_final[110:150, 932:956]
    cropped_120 = temp_final[110:150, 956:980]

    #row 4
    cropped_121 = temp_final[150:190, 28:53]
    cropped_122 = temp_final[150:190, 53:78]
    cropped_123 = temp_final[150:190, 78:103]
    cropped_124 = temp_final[150:190, 103:128]
    cropped_125 = temp_final[150:190, 128:153]
    cropped_126 = temp_final[150:190, 153:178]
    cropped_127 = temp_final[150:190, 175:200]
    cropped_128 = temp_final[150:190, 200:225]
    cropped_129 = temp_final[150:190, 223:248]
    cropped_130 = temp_final[150:190, 248:273]
    cropped_131 = temp_final[150:190, 272:296]
    cropped_132 = temp_final[150:190, 295:320]
    cropped_133 = temp_final[150:190, 320:344]
    cropped_134 = temp_final[150:190, 343:368]
    cropped_135 = temp_final[150:190, 367:391]
    cropped_136 = temp_final[150:190, 391:415]
    cropped_137 = temp_final[150:190, 415:439]
    cropped_138 = temp_final[150:190, 439:463]
    cropped_139 = temp_final[150:190, 462:486]
    cropped_140 = temp_final[150:190, 486:510]
    cropped_141 = temp_final[150:190, 510:533]
    cropped_142 = temp_final[150:190, 532:556]
    cropped_143 = temp_final[150:190, 557:579]
    cropped_144 = temp_final[150:190, 579:603]
    cropped_145 = temp_final[150:190, 603:625]
    cropped_146 = temp_final[150:190, 625:648]
    cropped_147 = temp_final[150:190, 648:672]
    cropped_148 = temp_final[150:190, 672:696]
    cropped_149 = temp_final[150:190, 695:719]
    cropped_150 = temp_final[150:190, 719:743]
    cropped_151 = temp_final[150:190, 742:766]
    cropped_152 = temp_final[150:190, 766:790]
    cropped_153 = temp_final[150:190, 789:813]
    cropped_154 = temp_final[150:190, 813:837]
    cropped_155 = temp_final[150:190, 837:861]
    cropped_156 = temp_final[150:190, 860:884]
    cropped_157 = temp_final[150:190, 884:908]
    cropped_158 = temp_final[150:190, 908:932]
    cropped_159 = temp_final[150:190, 932:956]
    cropped_160 = temp_final[150:190, 956:980]

    #row 5
    cropped_161 = temp_final[190:230, 28:53]
    cropped_162 = temp_final[190:230, 53:78]
    cropped_163 = temp_final[190:230, 78:103]
    cropped_164 = temp_final[190:230, 103:128]
    cropped_165 = temp_final[190:230, 128:153]
    cropped_166 = temp_final[190:230, 153:178]
    cropped_167 = temp_final[190:230, 175:200]
    cropped_168 = temp_final[190:230, 200:225]
    cropped_169 = temp_final[190:230, 223:248]
    cropped_170 = temp_final[190:230, 248:273]
    cropped_171 = temp_final[190:230, 272:296]
    cropped_172 = temp_final[190:230, 295:320]
    cropped_173 = temp_final[190:230, 320:344]
    cropped_174 = temp_final[190:230, 343:368]
    cropped_175 = temp_final[190:230, 367:391]
    cropped_176 = temp_final[190:230, 391:415]
    cropped_177 = temp_final[190:230, 415:439]
    cropped_178 = temp_final[190:230, 439:463]
    cropped_179 = temp_final[190:230, 462:486]
    cropped_180 = temp_final[190:230, 486:510]
    cropped_181 = temp_final[190:230, 510:533]
    cropped_182 = temp_final[190:230, 532:556]
    cropped_183 = temp_final[190:230, 557:579]
    cropped_184 = temp_final[190:230, 579:603]
    cropped_185 = temp_final[190:230, 603:625]
    cropped_186 = temp_final[190:230, 625:648]
    cropped_187 = temp_final[190:230, 648:672]
    cropped_188 = temp_final[190:230, 672:696]
    cropped_189 = temp_final[190:230, 695:719]
    cropped_190 = temp_final[190:230, 719:743]
    cropped_191 = temp_final[190:230, 742:766]
    cropped_192 = temp_final[190:230, 766:790]
    cropped_193 = temp_final[190:230, 789:813]
    cropped_194 = temp_final[190:230, 813:837]
    cropped_195 = temp_final[190:230, 837:861]
    cropped_196 = temp_final[190:230, 860:884]
    cropped_197 = temp_final[190:230, 884:908]
    cropped_198 = temp_final[190:230, 908:932]
    cropped_199 = temp_final[190:230, 932:956]
    cropped_200 = temp_final[190:230, 956:980]

    #row 6
    cropped_201 = temp_final[228:268, 28:53]
    cropped_202 = temp_final[228:268, 53:78]
    cropped_203 = temp_final[228:268, 78:103]
    cropped_204 = temp_final[228:268, 103:128]
    cropped_205 = temp_final[228:268, 128:153]
    cropped_206 = temp_final[228:268, 153:178]
    cropped_207 = temp_final[228:268, 175:200]
    cropped_208 = temp_final[228:268, 200:225]
    cropped_209 = temp_final[228:268, 223:248]
    cropped_210 = temp_final[228:268, 248:273]
    cropped_211 = temp_final[228:268, 272:296]
    cropped_212 = temp_final[228:268, 295:320]
    cropped_213 = temp_final[228:268, 320:344]
    cropped_214 = temp_final[228:268, 343:368]
    cropped_215 = temp_final[228:268, 367:391]
    cropped_216 = temp_final[228:268, 391:415]
    cropped_217 = temp_final[228:268, 415:439]
    cropped_218 = temp_final[228:268, 439:463]
    cropped_219 = temp_final[228:268, 462:486]
    cropped_220 = temp_final[228:268, 486:510]
    cropped_221 = temp_final[228:268, 510:533]
    cropped_222 = temp_final[228:268, 532:556]
    cropped_223 = temp_final[228:268, 557:579]
    cropped_224 = temp_final[228:268, 579:603]
    cropped_225 = temp_final[228:268, 603:625]
    cropped_226 = temp_final[228:268, 625:648]
    cropped_227 = temp_final[228:268, 648:672]
    cropped_228 = temp_final[228:268, 672:696]
    cropped_229 = temp_final[228:268, 695:719]
    cropped_230 = temp_final[228:268, 719:743]
    cropped_231 = temp_final[228:268, 742:766]
    cropped_232 = temp_final[228:268, 766:790]
    cropped_233 = temp_final[228:268, 789:813]
    cropped_234 = temp_final[228:268, 813:837]
    cropped_235 = temp_final[228:268, 837:861]
    cropped_236 = temp_final[228:268, 860:884]
    cropped_237 = temp_final[228:268, 884:908]
    cropped_238 = temp_final[228:268, 908:932]
    cropped_239 = temp_final[228:268, 932:956]
    cropped_240 = temp_final[228:268, 956:980]

    #row 7
    cropped_241 = temp_final[268:308, 28:53]
    cropped_242 = temp_final[268:308, 53:78]
    cropped_243 = temp_final[268:308, 78:103]
    cropped_244 = temp_final[268:308, 103:128]
    cropped_245 = temp_final[268:308, 128:153]
    cropped_246 = temp_final[268:308, 153:178]
    cropped_247 = temp_final[268:308, 175:200]
    cropped_248 = temp_final[268:308, 200:225]
    cropped_249 = temp_final[268:308, 223:248]
    cropped_250 = temp_final[268:308, 248:273]
    cropped_251 = temp_final[268:308, 272:296]
    cropped_252 = temp_final[268:308, 295:320]
    cropped_253 = temp_final[268:308, 320:344]
    cropped_254 = temp_final[268:308, 343:368]
    cropped_255 = temp_final[268:308, 367:391]
    cropped_256 = temp_final[268:308, 391:415]
    cropped_257 = temp_final[268:308, 415:439]
    cropped_258 = temp_final[268:308, 439:463]
    cropped_259 = temp_final[268:308, 462:486]
    cropped_260 = temp_final[268:308, 486:510]
    cropped_261 = temp_final[268:308, 510:533]
    cropped_262 = temp_final[268:308, 532:556]
    cropped_263 = temp_final[268:308, 557:579]
    cropped_264 = temp_final[268:308, 579:603]
    cropped_265 = temp_final[268:308, 603:625]
    cropped_266 = temp_final[268:308, 625:648]
    cropped_267 = temp_final[268:308, 648:672]
    cropped_268 = temp_final[268:308, 672:696]
    cropped_269 = temp_final[268:308, 695:719]
    cropped_270 = temp_final[268:308, 719:743]
    cropped_271 = temp_final[268:308, 742:766]
    cropped_272 = temp_final[268:308, 766:790]
    cropped_273 = temp_final[268:308, 789:813]
    cropped_274 = temp_final[268:308, 813:837]
    cropped_275 = temp_final[268:308, 837:861]
    cropped_276 = temp_final[268:308, 860:884]
    cropped_277 = temp_final[268:308, 884:908]
    cropped_278 = temp_final[268:308, 908:932]
    cropped_279 = temp_final[268:308, 932:956]
    cropped_280 = temp_final[268:308, 956:980]

    ###2
    cropped2_1 = img1[30:70, 28:53]
    cropped2_2 = img1[30:70, 53:78]
    cropped2_3 = img1[30:70, 78:103]
    cropped2_4 = img1[30:70, 103:128]
    cropped2_5 = img1[30:70, 128:153]
    cropped2_6 = img1[30:70, 153:178]
    cropped2_7 = img1[30:70, 175:200]
    cropped2_8 = img1[30:70, 200:225]
    cropped2_9 = img1[30:70, 223:248]
    cropped2_10 = img1[30:70, 248:273]
    cropped2_11 = img1[30:70, 272:296]
    cropped2_12 = img1[30:70, 295:320]
    cropped2_13 = img1[30:70, 320:344]
    cropped2_14 = img1[30:70, 343:368]
    cropped2_15 = img1[30:70, 367:391]
    cropped2_16 = img1[30:70, 391:415]
    cropped2_17 = img1[30:70, 415:439]
    cropped2_18 = img1[30:70, 439:463]
    cropped2_19 = img1[30:70, 462:486]
    cropped2_20 = img1[30:70, 486:510]
    cropped2_21 = img1[30:70, 510:533]
    cropped2_22 = img1[30:70, 532:556]
    cropped2_23 = img1[30:70, 557:579]
    cropped2_24 = img1[30:70, 579:603]
    cropped2_25 = img1[30:70, 603:625]
    cropped2_26 = img1[30:70, 625:648]
    cropped2_27 = img1[30:70, 648:672]
    cropped2_28 = img1[30:70, 672:696]
    cropped2_29 = img1[30:70, 695:719]
    cropped2_30 = img1[30:70, 719:743]
    cropped2_31 = img1[30:70, 742:766]
    cropped2_32 = img1[30:70, 766:790]
    cropped2_33 = img1[30:70, 789:813]
    cropped2_34 = img1[30:70, 813:837]
    cropped2_35 = img1[30:70, 837:861]
    cropped2_36 = img1[30:70, 860:884]
    cropped2_37 = img1[30:70, 884:908]
    cropped2_38 = img1[30:70, 908:932]
    cropped2_39 = img1[30:70, 932:956]
    cropped2_40 = img1[30:70, 956:980]
    cropped2_41 = img1[70:110, 28:53]
    cropped2_42 = img1[70:110, 53:78]
    cropped2_43 = img1[70:110, 78:103]
    cropped2_44 = img1[70:110, 103:128]
    cropped2_45 = img1[70:110, 128:153]
    cropped2_46 = img1[70:110, 153:178]
    cropped2_47 = img1[70:110, 175:200]
    cropped2_48 = img1[70:110, 200:225]
    cropped2_49 = img1[70:110, 223:248]
    cropped2_50 = img1[70:110, 248:273]
    cropped2_51 = img1[70:110, 272:296]
    cropped2_52 = img1[70:110, 295:320]
    cropped2_53 = img1[70:110, 320:344]
    cropped2_54 = img1[70:110, 343:368]
    cropped2_55 = img1[70:110, 367:391]
    cropped2_56 = img1[70:110, 391:415]
    cropped2_57 = img1[70:110, 415:439]
    cropped2_58 = img1[70:110, 439:463]
    cropped2_59 = img1[70:110, 462:486]
    cropped2_60 = img1[70:110, 486:510]
    cropped2_61 = img1[70:110, 510:533]
    cropped2_62 = img1[70:110, 532:556]
    cropped2_63 = img1[70:110, 557:579]
    cropped2_64 = img1[70:110, 579:603]
    cropped2_65 = img1[70:110, 603:625]
    cropped2_66 = img1[70:110, 625:648]
    cropped2_67 = img1[70:110, 648:672]
    cropped2_68 = img1[70:110, 672:696]
    cropped2_69 = img1[70:110, 695:719]
    cropped2_70 = img1[70:110, 719:743]
    cropped2_71 = img1[70:110, 742:766]
    cropped2_72 = img1[70:110, 766:790]
    cropped2_73 = img1[70:110, 789:813]
    cropped2_74 = img1[70:110, 813:837]
    cropped2_75 = img1[70:110, 837:861]
    cropped2_76 = img1[70:110, 860:884]
    cropped2_77 = img1[70:110, 884:908]
    cropped2_78 = img1[70:110, 908:932]
    cropped2_79 = img1[70:110, 932:956]
    cropped2_80 = img1[70:110, 956:980]
    cropped2_81 = img1[110:150, 28:53]
    cropped2_82 = img1[110:150, 53:78]
    cropped2_83 = img1[110:150, 78:103]
    cropped2_84 = img1[110:150, 103:128]
    cropped2_85 = img1[110:150, 128:153]
    cropped2_86 = img1[110:150, 153:178]
    cropped2_87 = img1[110:150, 175:200]
    cropped2_88 = img1[110:150, 200:225]
    cropped2_89 = img1[110:150, 223:248]
    cropped2_90 = img1[110:150, 248:273]
    cropped2_91 = img1[110:150, 272:296]
    cropped2_92 = img1[110:150, 295:320]
    cropped2_93 = img1[110:150, 320:344]
    cropped2_94 = img1[110:150, 343:368]
    cropped2_95 = img1[110:150, 367:391]
    cropped2_96 = img1[110:150, 391:415]
    cropped2_97 = img1[110:150, 415:439]
    cropped2_98 = img1[110:150, 439:463]
    cropped2_99 = img1[110:150, 462:486]
    cropped2_100 = img1[110:150, 486:510]
    cropped2_101 = img1[110:150, 510:533]
    cropped2_102 = img1[110:150, 532:556]
    cropped2_103 = img1[110:150, 557:579]
    cropped2_104 = img1[110:150, 579:603]
    cropped2_105 = img1[110:150, 603:625]
    cropped2_106 = img1[110:150, 625:648]
    cropped2_107 = img1[110:150, 648:672]
    cropped2_108 = img1[110:150, 672:696]
    cropped2_109 = img1[110:150, 695:719]
    cropped2_110 = img1[110:150, 719:743]
    cropped2_111 = img1[110:150, 742:766]
    cropped2_112 = img1[110:150, 766:790]
    cropped2_113 = img1[110:150, 789:813]
    cropped2_114 = img1[110:150, 813:837]
    cropped2_115 = img1[110:150, 837:861]
    cropped2_116 = img1[110:150, 860:884]
    cropped2_117 = img1[110:150, 884:908]
    cropped2_118 = img1[110:150, 908:932]
    cropped2_119 = img1[110:150, 932:956]
    cropped2_120 = img1[110:150, 956:980]
    cropped2_121 = img1[150:190, 28:53]
    cropped2_122 = img1[150:190, 53:78]
    cropped2_123 = img1[150:190, 78:103]
    cropped2_124 = img1[150:190, 103:128]
    cropped2_125 = img1[150:190, 128:153]
    cropped2_126 = img1[150:190, 153:178]
    cropped2_127 = img1[150:190, 175:200]
    cropped2_128 = img1[150:190, 200:225]
    cropped2_129 = img1[150:190, 223:248]
    cropped2_130 = img1[150:190, 248:273]
    cropped2_131 = img1[150:190, 272:296]
    cropped2_132 = img1[150:190, 295:320]
    cropped2_133 = img1[150:190, 320:344]
    cropped2_134 = img1[150:190, 343:368]
    cropped2_135 = img1[150:190, 367:391]
    cropped2_136 = img1[150:190, 391:415]
    cropped2_137 = img1[150:190, 415:439]
    cropped2_138 = img1[150:190, 439:463]
    cropped2_139 = img1[150:190, 462:486]
    cropped2_140 = img1[150:190, 486:510]
    cropped2_141 = img1[150:190, 510:533]
    cropped2_142 = img1[150:190, 532:556]
    cropped2_143 = img1[150:190, 557:579]
    cropped2_144 = img1[150:190, 579:603]
    cropped2_145 = img1[150:190, 603:625]
    cropped2_146 = img1[150:190, 625:648]
    cropped2_147 = img1[150:190, 648:672]
    cropped2_148 = img1[150:190, 672:696]
    cropped2_149 = img1[150:190, 695:719]
    cropped2_150 = img1[150:190, 719:743]
    cropped2_151 = img1[150:190, 742:766]
    cropped2_152 = img1[150:190, 766:790]
    cropped2_153 = img1[150:190, 789:813]
    cropped2_154 = img1[150:190, 813:837]
    cropped2_155 = img1[150:190, 837:861]
    cropped2_156 = img1[150:190, 860:884]
    cropped2_157 = img1[150:190, 884:908]
    cropped2_158 = img1[150:190, 908:932]
    cropped2_159 = img1[150:190, 932:956]
    cropped2_160 = img1[150:190, 956:980]
    cropped2_161 = img1[190:230, 28:53]
    cropped2_162 = img1[190:230, 53:78]
    cropped2_163 = img1[190:230, 78:103]
    cropped2_164 = img1[190:230, 103:128]
    cropped2_165 = img1[190:230, 128:153]
    cropped2_166 = img1[190:230, 153:178]
    cropped2_167 = img1[190:230, 175:200]
    cropped2_168 = img1[190:230, 200:225]
    cropped2_169 = img1[190:230, 223:248]
    cropped2_170 = img1[190:230, 248:273]
    cropped2_171 = img1[190:230, 272:296]
    cropped2_172 = img1[190:230, 295:320]
    cropped2_173 = img1[190:230, 320:344]
    cropped2_174 = img1[190:230, 343:368]
    cropped2_175 = img1[190:230, 367:391]
    cropped2_176 = img1[190:230, 391:415]
    cropped2_177 = img1[190:230, 415:439]
    cropped2_178 = img1[190:230, 439:463]
    cropped2_179 = img1[190:230, 462:486]
    cropped2_180 = img1[190:230, 486:510]
    cropped2_181 = img1[190:230, 510:533]
    cropped2_182 = img1[190:230, 532:556]
    cropped2_183 = img1[190:230, 557:579]
    cropped2_184 = img1[190:230, 579:603]
    cropped2_185 = img1[190:230, 603:625]
    cropped2_186 = img1[190:230, 625:648]
    cropped2_187 = img1[190:230, 648:672]
    cropped2_188 = img1[190:230, 672:696]
    cropped2_189 = img1[190:230, 695:719]
    cropped2_190 = img1[190:230, 719:743]
    cropped2_191 = img1[190:230, 742:766]
    cropped2_192 = img1[190:230, 766:790]
    cropped2_193 = img1[190:230, 789:813]
    cropped2_194 = img1[190:230, 813:837]
    cropped2_195 = img1[190:230, 837:861]
    cropped2_196 = img1[190:230, 860:884]
    cropped2_197 = img1[190:230, 884:908]
    cropped2_198 = img1[190:230, 908:932]
    cropped2_199 = img1[190:230, 932:956]
    cropped2_200 = img1[190:230, 956:980]
    cropped2_201 = img1[228:268, 28:53]
    cropped2_202 = img1[228:268, 53:78]
    cropped2_203 = img1[228:268, 78:103]
    cropped2_204 = img1[228:268, 103:128]
    cropped2_205 = img1[228:268, 128:153]
    cropped2_206 = img1[228:268, 153:178]
    cropped2_207 = img1[228:268, 175:200]
    cropped2_208 = img1[228:268, 200:225]
    cropped2_209 = img1[228:268, 223:248]
    cropped2_210 = img1[228:268, 248:273]
    cropped2_211 = img1[228:268, 272:296]
    cropped2_212 = img1[228:268, 295:320]
    cropped2_213 = img1[228:268, 320:344]
    cropped2_214 = img1[228:268, 343:368]
    cropped2_215 = img1[228:268, 367:391]
    cropped2_216 = img1[228:268, 391:415]
    cropped2_217 = img1[228:268, 415:439]
    cropped2_218 = img1[228:268, 439:463]
    cropped2_219 = img1[228:268, 462:486]
    cropped2_220 = img1[228:268, 486:510]
    cropped2_221 = img1[228:268, 510:533]
    cropped2_222 = img1[228:268, 532:556]
    cropped2_223 = img1[228:268, 557:579]
    cropped2_224 = img1[228:268, 579:603]
    cropped2_225 = img1[228:268, 603:625]
    cropped2_226 = img1[228:268, 625:648]
    cropped2_227 = img1[228:268, 648:672]
    cropped2_228 = img1[228:268, 672:696]
    cropped2_229 = img1[228:268, 695:719]
    cropped2_230 = img1[228:268, 719:743]
    cropped2_231 = img1[228:268, 742:766]
    cropped2_232 = img1[228:268, 766:790]
    cropped2_233 = img1[228:268, 789:813]
    cropped2_234 = img1[228:268, 813:837]
    cropped2_235 = img1[228:268, 837:861]
    cropped2_236 = img1[228:268, 860:884]
    cropped2_237 = img1[228:268, 884:908]
    cropped2_238 = img1[228:268, 908:932]
    cropped2_239 = img1[228:268, 932:956]
    cropped2_240 = img1[228:268, 956:980]
    cropped2_241 = img1[268:308, 28:53]
    cropped2_242 = img1[268:308, 53:78]
    cropped2_243 = img1[268:308, 78:103]
    cropped2_244 = img1[268:308, 103:128]
    cropped2_245 = img1[268:308, 128:153]
    cropped2_246 = img1[268:308, 153:178]
    cropped2_247 = img1[268:308, 175:200]
    cropped2_248 = img1[268:308, 200:225]
    cropped2_249 = img1[268:308, 223:248]
    cropped2_250 = img1[268:308, 248:273]
    cropped2_251 = img1[268:308, 272:296]
    cropped2_252 = img1[268:308, 295:320]
    cropped2_253 = img1[268:308, 320:344]
    cropped2_254 = img1[268:308, 343:368]
    cropped2_255 = img1[268:308, 367:391]
    cropped2_256 = img1[268:308, 391:415]
    cropped2_257 = img1[268:308, 415:439]
    cropped2_258 = img1[268:308, 439:463]
    cropped2_259 = img1[268:308, 462:486]
    cropped2_260 = img1[268:308, 486:510]
    cropped2_261 = img1[268:308, 510:533]
    cropped2_262 = img1[268:308, 532:556]
    cropped2_263 = img1[268:308, 557:579]
    cropped2_264 = img1[268:308, 579:603]
    cropped2_265 = img1[268:308, 603:625]
    cropped2_266 = img1[268:308, 625:648]
    cropped2_267 = img1[268:308, 648:672]
    cropped2_268 = img1[268:308, 672:696]
    cropped2_269 = img1[268:308, 695:719]
    cropped2_270 = img1[268:308, 719:743]
    cropped2_271 = img1[268:308, 742:766]
    cropped2_272 = img1[268:308, 766:790]
    cropped2_273 = img1[268:308, 789:813]
    cropped2_274 = img1[268:308, 813:837]
    cropped2_275 = img1[268:308, 837:861]
    cropped2_276 = img1[268:308, 860:884]
    cropped2_277 = img1[268:308, 884:908]
    cropped2_278 = img1[268:308, 908:932]
    cropped2_279 = img1[268:308, 932:956]
    cropped2_280 = img1[268:308, 956:980]
    ###
    ###3
    cropped3_1 = img2[30:70, 28:53]
    cropped3_2 = img2[30:70, 53:78]
    cropped3_3 = img2[30:70, 78:103]
    cropped3_4 = img2[30:70, 103:128]
    cropped3_5 = img2[30:70, 128:153]
    cropped3_6 = img2[30:70, 153:178]
    cropped3_7 = img2[30:70, 175:200]
    cropped3_8 = img2[30:70, 200:225]
    cropped3_9 = img2[30:70, 223:248]
    cropped3_10 = img2[30:70, 248:273]
    cropped3_11 = img2[30:70, 272:296]
    cropped3_12 = img2[30:70, 295:320]
    cropped3_13 = img2[30:70, 320:344]
    cropped3_14 = img2[30:70, 343:368]
    cropped3_15 = img2[30:70, 367:391]
    cropped3_16 = img2[30:70, 391:415]
    cropped3_17 = img2[30:70, 415:439]
    cropped3_18 = img2[30:70, 439:463]
    cropped3_19 = img2[30:70, 462:486]
    cropped3_20 = img2[30:70, 486:510]
    cropped3_21 = img2[30:70, 510:533]
    cropped3_22 = img2[30:70, 532:556]
    cropped3_23 = img2[30:70, 557:579]
    cropped3_24 = img2[30:70, 579:603]
    cropped3_25 = img2[30:70, 603:625]
    cropped3_26 = img2[30:70, 625:648]
    cropped3_27 = img2[30:70, 648:672]
    cropped3_28 = img2[30:70, 672:696]
    cropped3_29 = img2[30:70, 695:719]
    cropped3_30 = img2[30:70, 719:743]
    cropped3_31 = img2[30:70, 742:766]
    cropped3_32 = img2[30:70, 766:790]
    cropped3_33 = img2[30:70, 789:813]
    cropped3_34 = img2[30:70, 813:837]
    cropped3_35 = img2[30:70, 837:861]
    cropped3_36 = img2[30:70, 860:884]
    cropped3_37 = img2[30:70, 884:908]
    cropped3_38 = img2[30:70, 908:932]
    cropped3_39 = img2[30:70, 932:956]
    cropped3_40 = img2[30:70, 956:980]
    cropped3_41 = img2[70:110, 28:53]
    cropped3_42 = img2[70:110, 53:78]
    cropped3_43 = img2[70:110, 78:103]
    cropped3_44 = img2[70:110, 103:128]
    cropped3_45 = img2[70:110, 128:153]
    cropped3_46 = img2[70:110, 153:178]
    cropped3_47 = img2[70:110, 175:200]
    cropped3_48 = img2[70:110, 200:225]
    cropped3_49 = img2[70:110, 223:248]
    cropped3_50 = img2[70:110, 248:273]
    cropped3_51 = img2[70:110, 272:296]
    cropped3_52 = img2[70:110, 295:320]
    cropped3_53 = img2[70:110, 320:344]
    cropped3_54 = img2[70:110, 343:368]
    cropped3_55 = img2[70:110, 367:391]
    cropped3_56 = img2[70:110, 391:415]
    cropped3_57 = img2[70:110, 415:439]
    cropped3_58 = img2[70:110, 439:463]
    cropped3_59 = img2[70:110, 462:486]
    cropped3_60 = img2[70:110, 486:510]
    cropped3_61 = img2[70:110, 510:533]
    cropped3_62 = img2[70:110, 532:556]
    cropped3_63 = img2[70:110, 557:579]
    cropped3_64 = img2[70:110, 579:603]
    cropped3_65 = img2[70:110, 603:625]
    cropped3_66 = img2[70:110, 625:648]
    cropped3_67 = img2[70:110, 648:672]
    cropped3_68 = img2[70:110, 672:696]
    cropped3_69 = img2[70:110, 695:719]
    cropped3_70 = img2[70:110, 719:743]
    cropped3_71 = img2[70:110, 742:766]
    cropped3_72 = img2[70:110, 766:790]
    cropped3_73 = img2[70:110, 789:813]
    cropped3_74 = img2[70:110, 813:837]
    cropped3_75 = img2[70:110, 837:861]
    cropped3_76 = img2[70:110, 860:884]
    cropped3_77 = img2[70:110, 884:908]
    cropped3_78 = img2[70:110, 908:932]
    cropped3_79 = img2[70:110, 932:956]
    cropped3_80 = img2[70:110, 956:980]
    cropped3_81 = img2[110:150, 28:53]
    cropped3_82 = img2[110:150, 53:78]
    cropped3_83 = img2[110:150, 78:103]
    cropped3_84 = img2[110:150, 103:128]
    cropped3_85 = img2[110:150, 128:153]
    cropped3_86 = img2[110:150, 153:178]
    cropped3_87 = img2[110:150, 175:200]
    cropped3_88 = img2[110:150, 200:225]
    cropped3_89 = img2[110:150, 223:248]
    cropped3_90 = img2[110:150, 248:273]
    cropped3_91 = img2[110:150, 272:296]
    cropped3_92 = img2[110:150, 295:320]
    cropped3_93 = img2[110:150, 320:344]
    cropped3_94 = img2[110:150, 343:368]
    cropped3_95 = img2[110:150, 367:391]
    cropped3_96 = img2[110:150, 391:415]
    cropped3_97 = img2[110:150, 415:439]
    cropped3_98 = img2[110:150, 439:463]
    cropped3_99 = img2[110:150, 462:486]
    cropped3_100 = img2[110:150, 486:510]
    cropped3_101 = img2[110:150, 510:533]
    cropped3_102 = img2[110:150, 532:556]
    cropped3_103 = img2[110:150, 557:579]
    cropped3_104 = img2[110:150, 579:603]
    cropped3_105 = img2[110:150, 603:625]
    cropped3_106 = img2[110:150, 625:648]
    cropped3_107 = img2[110:150, 648:672]
    cropped3_108 = img2[110:150, 672:696]
    cropped3_109 = img2[110:150, 695:719]
    cropped3_110 = img2[110:150, 719:743]
    cropped3_111 = img2[110:150, 742:766]
    cropped3_112 = img2[110:150, 766:790]
    cropped3_113 = img2[110:150, 789:813]
    cropped3_114 = img2[110:150, 813:837]
    cropped3_115 = img2[110:150, 837:861]
    cropped3_116 = img2[110:150, 860:884]
    cropped3_117 = img2[110:150, 884:908]
    cropped3_118 = img2[110:150, 908:932]
    cropped3_119 = img2[110:150, 932:956]
    cropped3_120 = img2[110:150, 956:980]
    cropped3_121 = img2[150:190, 28:53]
    cropped3_122 = img2[150:190, 53:78]
    cropped3_123 = img2[150:190, 78:103]
    cropped3_124 = img2[150:190, 103:128]
    cropped3_125 = img2[150:190, 128:153]
    cropped3_126 = img2[150:190, 153:178]
    cropped3_127 = img2[150:190, 175:200]
    cropped3_128 = img2[150:190, 200:225]
    cropped3_129 = img2[150:190, 223:248]
    cropped3_130 = img2[150:190, 248:273]
    cropped3_131 = img2[150:190, 272:296]
    cropped3_132 = img2[150:190, 295:320]
    cropped3_133 = img2[150:190, 320:344]
    cropped3_134 = img2[150:190, 343:368]
    cropped3_135 = img2[150:190, 367:391]
    cropped3_136 = img2[150:190, 391:415]
    cropped3_137 = img2[150:190, 415:439]
    cropped3_138 = img2[150:190, 439:463]
    cropped3_139 = img2[150:190, 462:486]
    cropped3_140 = img2[150:190, 486:510]
    cropped3_141 = img2[150:190, 510:533]
    cropped3_142 = img2[150:190, 532:556]
    cropped3_143 = img2[150:190, 557:579]
    cropped3_144 = img2[150:190, 579:603]
    cropped3_145 = img2[150:190, 603:625]
    cropped3_146 = img2[150:190, 625:648]
    cropped3_147 = img2[150:190, 648:672]
    cropped3_148 = img2[150:190, 672:696]
    cropped3_149 = img2[150:190, 695:719]
    cropped3_150 = img2[150:190, 719:743]
    cropped3_151 = img2[150:190, 742:766]
    cropped3_152 = img2[150:190, 766:790]
    cropped3_153 = img2[150:190, 789:813]
    cropped3_154 = img2[150:190, 813:837]
    cropped3_155 = img2[150:190, 837:861]
    cropped3_156 = img2[150:190, 860:884]
    cropped3_157 = img2[150:190, 884:908]
    cropped3_158 = img2[150:190, 908:932]
    cropped3_159 = img2[150:190, 932:956]
    cropped3_160 = img2[150:190, 956:980]
    cropped3_161 = img2[190:230, 28:53]
    cropped3_162 = img2[190:230, 53:78]
    cropped3_163 = img2[190:230, 78:103]
    cropped3_164 = img2[190:230, 103:128]
    cropped3_165 = img2[190:230, 128:153]
    cropped3_166 = img2[190:230, 153:178]
    cropped3_167 = img2[190:230, 175:200]
    cropped3_168 = img2[190:230, 200:225]
    cropped3_169 = img2[190:230, 223:248]
    cropped3_170 = img2[190:230, 248:273]
    cropped3_171 = img2[190:230, 272:296]
    cropped3_172 = img2[190:230, 295:320]
    cropped3_173 = img2[190:230, 320:344]
    cropped3_174 = img2[190:230, 343:368]
    cropped3_175 = img2[190:230, 367:391]
    cropped3_176 = img2[190:230, 391:415]
    cropped3_177 = img2[190:230, 415:439]
    cropped3_178 = img2[190:230, 439:463]
    cropped3_179 = img2[190:230, 462:486]
    cropped3_180 = img2[190:230, 486:510]
    cropped3_181 = img2[190:230, 510:533]
    cropped3_182 = img2[190:230, 532:556]
    cropped3_183 = img2[190:230, 557:579]
    cropped3_184 = img2[190:230, 579:603]
    cropped3_185 = img2[190:230, 603:625]
    cropped3_186 = img2[190:230, 625:648]
    cropped3_187 = img2[190:230, 648:672]
    cropped3_188 = img2[190:230, 672:696]
    cropped3_189 = img2[190:230, 695:719]
    cropped3_190 = img2[190:230, 719:743]
    cropped3_191 = img2[190:230, 742:766]
    cropped3_192 = img2[190:230, 766:790]
    cropped3_193 = img2[190:230, 789:813]
    cropped3_194 = img2[190:230, 813:837]
    cropped3_195 = img2[190:230, 837:861]
    cropped3_196 = img2[190:230, 860:884]
    cropped3_197 = img2[190:230, 884:908]
    cropped3_198 = img2[190:230, 908:932]
    cropped3_199 = img2[190:230, 932:956]
    cropped3_200 = img2[190:230, 956:980]
    cropped3_201 = img2[228:268, 28:53]
    cropped3_202 = img2[228:268, 53:78]
    cropped3_203 = img2[228:268, 78:103]
    cropped3_204 = img2[228:268, 103:128]
    cropped3_205 = img2[228:268, 128:153]
    cropped3_206 = img2[228:268, 153:178]
    cropped3_207 = img2[228:268, 175:200]
    cropped3_208 = img2[228:268, 200:225]
    cropped3_209 = img2[228:268, 223:248]
    cropped3_210 = img2[228:268, 248:273]
    cropped3_211 = img2[228:268, 272:296]
    cropped3_212 = img2[228:268, 295:320]
    cropped3_213 = img2[228:268, 320:344]
    cropped3_214 = img2[228:268, 343:368]
    cropped3_215 = img2[228:268, 367:391]
    cropped3_216 = img2[228:268, 391:415]
    cropped3_217 = img2[228:268, 415:439]
    cropped3_218 = img2[228:268, 439:463]
    cropped3_219 = img2[228:268, 462:486]
    cropped3_220 = img2[228:268, 486:510]
    cropped3_221 = img2[228:268, 510:533]
    cropped3_222 = img2[228:268, 532:556]
    cropped3_223 = img2[228:268, 557:579]
    cropped3_224 = img2[228:268, 579:603]
    cropped3_225 = img2[228:268, 603:625]
    cropped3_226 = img2[228:268, 625:648]
    cropped3_227 = img2[228:268, 648:672]
    cropped3_228 = img2[228:268, 672:696]
    cropped3_229 = img2[228:268, 695:719]
    cropped3_230 = img2[228:268, 719:743]
    cropped3_231 = img2[228:268, 742:766]
    cropped3_232 = img2[228:268, 766:790]
    cropped3_233 = img2[228:268, 789:813]
    cropped3_234 = img2[228:268, 813:837]
    cropped3_235 = img2[228:268, 837:861]
    cropped3_236 = img2[228:268, 860:884]
    cropped3_237 = img2[228:268, 884:908]
    cropped3_238 = img2[228:268, 908:932]
    cropped3_239 = img2[228:268, 932:956]
    cropped3_240 = img2[228:268, 956:980]
    cropped3_241 = img2[268:308, 28:53]
    cropped3_242 = img2[268:308, 53:78]
    cropped3_243 = img2[268:308, 78:103]
    cropped3_244 = img2[268:308, 103:128]
    cropped3_245 = img2[268:308, 128:153]
    cropped3_246 = img2[268:308, 153:178]
    cropped3_247 = img2[268:308, 175:200]
    cropped3_248 = img2[268:308, 200:225]
    cropped3_249 = img2[268:308, 223:248]
    cropped3_250 = img2[268:308, 248:273]
    cropped3_251 = img2[268:308, 272:296]
    cropped3_252 = img2[268:308, 295:320]
    cropped3_253 = img2[268:308, 320:344]
    cropped3_254 = img2[268:308, 343:368]
    cropped3_255 = img2[268:308, 367:391]
    cropped3_256 = img2[268:308, 391:415]
    cropped3_257 = img2[268:308, 415:439]
    cropped3_258 = img2[268:308, 439:463]
    cropped3_259 = img2[268:308, 462:486]
    cropped3_260 = img2[268:308, 486:510]
    cropped3_261 = img2[268:308, 510:533]
    cropped3_262 = img2[268:308, 532:556]
    cropped3_263 = img2[268:308, 557:579]
    cropped3_264 = img2[268:308, 579:603]
    cropped3_265 = img2[268:308, 603:625]
    cropped3_266 = img2[268:308, 625:648]
    cropped3_267 = img2[268:308, 648:672]
    cropped3_268 = img2[268:308, 672:696]
    cropped3_269 = img2[268:308, 695:719]
    cropped3_270 = img2[268:308, 719:743]
    cropped3_271 = img2[268:308, 742:766]
    cropped3_272 = img2[268:308, 766:790]
    cropped3_273 = img2[268:308, 789:813]
    cropped3_274 = img2[268:308, 813:837]
    cropped3_275 = img2[268:308, 837:861]
    cropped3_276 = img2[268:308, 860:884]
    cropped3_277 = img2[268:308, 884:908]
    cropped3_278 = img2[268:308, 908:932]
    cropped3_279 = img2[268:308, 932:956]
    cropped3_280 = img2[268:308, 956:980]
    ###

    #writing for crops
    cv2.imwrite('crops/cropped_1.jpg',cropped_1)
    cv2.imwrite('crops/cropped_2.jpg',cropped_2)
    cv2.imwrite('crops/cropped_3.jpg',cropped_3)
    cv2.imwrite('crops/cropped_4.jpg',cropped_4)
    cv2.imwrite('crops/cropped_5.jpg',cropped_5)
    cv2.imwrite('crops/cropped_6.jpg',cropped_6)
    cv2.imwrite('crops/cropped_7.jpg',cropped_7)
    cv2.imwrite('crops/cropped_8.jpg',cropped_8)
    cv2.imwrite('crops/cropped_9.jpg',cropped_9)
    cv2.imwrite('crops/cropped_10.jpg',cropped_10)
    cv2.imwrite('crops/cropped_11.jpg',cropped_11)
    cv2.imwrite('crops/cropped_12.jpg',cropped_12)
    cv2.imwrite('crops/cropped_13.jpg',cropped_13)
    cv2.imwrite('crops/cropped_14.jpg',cropped_14)
    cv2.imwrite('crops/cropped_15.jpg',cropped_15)
    cv2.imwrite('crops/cropped_16.jpg',cropped_16)
    cv2.imwrite('crops/cropped_17.jpg',cropped_17)
    cv2.imwrite('crops/cropped_18.jpg',cropped_18)
    cv2.imwrite('crops/cropped_19.jpg',cropped_19)
    cv2.imwrite('crops/cropped_20.jpg',cropped_20)
    cv2.imwrite('crops/cropped_21.jpg',cropped_21)
    cv2.imwrite('crops/cropped_22.jpg',cropped_22)
    cv2.imwrite('crops/cropped_23.jpg',cropped_23)
    cv2.imwrite('crops/cropped_24.jpg',cropped_24)
    cv2.imwrite('crops/cropped_25.jpg',cropped_25)
    cv2.imwrite('crops/cropped_26.jpg',cropped_26)
    cv2.imwrite('crops/cropped_27.jpg',cropped_27)
    cv2.imwrite('crops/cropped_28.jpg',cropped_28)
    cv2.imwrite('crops/cropped_29.jpg',cropped_29)
    cv2.imwrite('crops/cropped_30.jpg',cropped_30)
    cv2.imwrite('crops/cropped_31.jpg',cropped_31)
    cv2.imwrite('crops/cropped_32.jpg',cropped_32)
    cv2.imwrite('crops/cropped_33.jpg',cropped_33)
    cv2.imwrite('crops/cropped_34.jpg',cropped_34)
    cv2.imwrite('crops/cropped_35.jpg',cropped_35)
    cv2.imwrite('crops/cropped_36.jpg',cropped_36)
    cv2.imwrite('crops/cropped_37.jpg',cropped_37)
    cv2.imwrite('crops/cropped_38.jpg',cropped_38)
    cv2.imwrite('crops/cropped_39.jpg',cropped_39)
    cv2.imwrite('crops/cropped_40.jpg',cropped_40)
    cv2.imwrite('crops/cropped_41.jpg',cropped_41)
    cv2.imwrite('crops/cropped_42.jpg',cropped_42)
    cv2.imwrite('crops/cropped_43.jpg',cropped_43)
    cv2.imwrite('crops/cropped_44.jpg',cropped_44)
    cv2.imwrite('crops/cropped_45.jpg',cropped_45)
    cv2.imwrite('crops/cropped_46.jpg',cropped_46)
    cv2.imwrite('crops/cropped_47.jpg',cropped_47)
    cv2.imwrite('crops/cropped_48.jpg',cropped_48)
    cv2.imwrite('crops/cropped_49.jpg',cropped_49)
    cv2.imwrite('crops/cropped_50.jpg',cropped_50)
    cv2.imwrite('crops/cropped_51.jpg',cropped_51)
    cv2.imwrite('crops/cropped_52.jpg',cropped_52)
    cv2.imwrite('crops/cropped_53.jpg',cropped_53)
    cv2.imwrite('crops/cropped_54.jpg',cropped_54)
    cv2.imwrite('crops/cropped_55.jpg',cropped_55)
    cv2.imwrite('crops/cropped_56.jpg',cropped_56)
    cv2.imwrite('crops/cropped_57.jpg',cropped_57)
    cv2.imwrite('crops/cropped_58.jpg',cropped_58)
    cv2.imwrite('crops/cropped_59.jpg',cropped_59)
    cv2.imwrite('crops/cropped_60.jpg',cropped_60)
    cv2.imwrite('crops/cropped_61.jpg',cropped_61)
    cv2.imwrite('crops/cropped_62.jpg',cropped_62)
    cv2.imwrite('crops/cropped_63.jpg',cropped_63)
    cv2.imwrite('crops/cropped_64.jpg',cropped_64)
    cv2.imwrite('crops/cropped_65.jpg',cropped_65)
    cv2.imwrite('crops/cropped_66.jpg',cropped_66)
    cv2.imwrite('crops/cropped_67.jpg',cropped_67)
    cv2.imwrite('crops/cropped_68.jpg',cropped_68)
    cv2.imwrite('crops/cropped_69.jpg',cropped_69)
    cv2.imwrite('crops/cropped_70.jpg',cropped_70)
    cv2.imwrite('crops/cropped_71.jpg',cropped_71)
    cv2.imwrite('crops/cropped_72.jpg',cropped_72)
    cv2.imwrite('crops/cropped_73.jpg',cropped_73)
    cv2.imwrite('crops/cropped_74.jpg',cropped_74)
    cv2.imwrite('crops/cropped_75.jpg',cropped_75)
    cv2.imwrite('crops/cropped_76.jpg',cropped_76)
    cv2.imwrite('crops/cropped_77.jpg',cropped_77)
    cv2.imwrite('crops/cropped_78.jpg',cropped_78)
    cv2.imwrite('crops/cropped_79.jpg',cropped_79)
    cv2.imwrite('crops/cropped_80.jpg',cropped_80)
    cv2.imwrite('crops/cropped_81.jpg',cropped_81)
    cv2.imwrite('crops/cropped_82.jpg',cropped_82)
    cv2.imwrite('crops/cropped_83.jpg',cropped_83)
    cv2.imwrite('crops/cropped_84.jpg',cropped_84)
    cv2.imwrite('crops/cropped_85.jpg',cropped_85)
    cv2.imwrite('crops/cropped_86.jpg',cropped_86)
    cv2.imwrite('crops/cropped_87.jpg',cropped_87)
    cv2.imwrite('crops/cropped_88.jpg',cropped_88)
    cv2.imwrite('crops/cropped_89.jpg',cropped_89)
    cv2.imwrite('crops/cropped_90.jpg',cropped_90)
    cv2.imwrite('crops/cropped_91.jpg',cropped_91)
    cv2.imwrite('crops/cropped_92.jpg',cropped_92)
    cv2.imwrite('crops/cropped_93.jpg',cropped_93)
    cv2.imwrite('crops/cropped_94.jpg',cropped_94)
    cv2.imwrite('crops/cropped_95.jpg',cropped_95)
    cv2.imwrite('crops/cropped_96.jpg',cropped_96)
    cv2.imwrite('crops/cropped_97.jpg',cropped_97)
    cv2.imwrite('crops/cropped_98.jpg',cropped_98)
    cv2.imwrite('crops/cropped_99.jpg',cropped_99)
    cv2.imwrite('crops/cropped_100.jpg',cropped_100)
    cv2.imwrite('crops/cropped_101.jpg',cropped_101)
    cv2.imwrite('crops/cropped_102.jpg',cropped_102)
    cv2.imwrite('crops/cropped_103.jpg',cropped_103)
    cv2.imwrite('crops/cropped_104.jpg',cropped_104)
    cv2.imwrite('crops/cropped_105.jpg',cropped_105)
    cv2.imwrite('crops/cropped_106.jpg',cropped_106)
    cv2.imwrite('crops/cropped_107.jpg',cropped_107)
    cv2.imwrite('crops/cropped_108.jpg',cropped_108)
    cv2.imwrite('crops/cropped_109.jpg',cropped_109)
    cv2.imwrite('crops/cropped_110.jpg',cropped_110)
    cv2.imwrite('crops/cropped_111.jpg',cropped_111)
    cv2.imwrite('crops/cropped_112.jpg',cropped_112)
    cv2.imwrite('crops/cropped_113.jpg',cropped_113)
    cv2.imwrite('crops/cropped_114.jpg',cropped_114)
    cv2.imwrite('crops/cropped_115.jpg',cropped_115)
    cv2.imwrite('crops/cropped_116.jpg',cropped_116)
    cv2.imwrite('crops/cropped_117.jpg',cropped_117)
    cv2.imwrite('crops/cropped_118.jpg',cropped_118)
    cv2.imwrite('crops/cropped_119.jpg',cropped_119)
    cv2.imwrite('crops/cropped_120.jpg',cropped_120)
    cv2.imwrite('crops/cropped_121.jpg',cropped_121)
    cv2.imwrite('crops/cropped_122.jpg',cropped_122)
    cv2.imwrite('crops/cropped_123.jpg',cropped_123)
    cv2.imwrite('crops/cropped_124.jpg',cropped_124)
    cv2.imwrite('crops/cropped_125.jpg',cropped_125)
    cv2.imwrite('crops/cropped_126.jpg',cropped_126)
    cv2.imwrite('crops/cropped_127.jpg',cropped_127)
    cv2.imwrite('crops/cropped_128.jpg',cropped_128)
    cv2.imwrite('crops/cropped_129.jpg',cropped_129)
    cv2.imwrite('crops/cropped_130.jpg',cropped_130)
    cv2.imwrite('crops/cropped_131.jpg',cropped_131)
    cv2.imwrite('crops/cropped_132.jpg',cropped_132)
    cv2.imwrite('crops/cropped_133.jpg',cropped_133)
    cv2.imwrite('crops/cropped_134.jpg',cropped_134)
    cv2.imwrite('crops/cropped_135.jpg',cropped_135)
    cv2.imwrite('crops/cropped_136.jpg',cropped_136)
    cv2.imwrite('crops/cropped_137.jpg',cropped_137)
    cv2.imwrite('crops/cropped_138.jpg',cropped_138)
    cv2.imwrite('crops/cropped_139.jpg',cropped_139)
    cv2.imwrite('crops/cropped_140.jpg',cropped_140)
    cv2.imwrite('crops/cropped_141.jpg',cropped_141)
    cv2.imwrite('crops/cropped_142.jpg',cropped_142)
    cv2.imwrite('crops/cropped_143.jpg',cropped_143)
    cv2.imwrite('crops/cropped_144.jpg',cropped_144)
    cv2.imwrite('crops/cropped_145.jpg',cropped_145)
    cv2.imwrite('crops/cropped_146.jpg',cropped_146)
    cv2.imwrite('crops/cropped_147.jpg',cropped_147)
    cv2.imwrite('crops/cropped_148.jpg',cropped_148)
    cv2.imwrite('crops/cropped_149.jpg',cropped_149)
    cv2.imwrite('crops/cropped_150.jpg',cropped_150)
    cv2.imwrite('crops/cropped_151.jpg',cropped_151)
    cv2.imwrite('crops/cropped_152.jpg',cropped_152)
    cv2.imwrite('crops/cropped_153.jpg',cropped_153)
    cv2.imwrite('crops/cropped_154.jpg',cropped_154)
    cv2.imwrite('crops/cropped_155.jpg',cropped_155)
    cv2.imwrite('crops/cropped_156.jpg',cropped_156)
    cv2.imwrite('crops/cropped_157.jpg',cropped_157)
    cv2.imwrite('crops/cropped_158.jpg',cropped_158)
    cv2.imwrite('crops/cropped_159.jpg',cropped_159)
    cv2.imwrite('crops/cropped_160.jpg',cropped_160)
    cv2.imwrite('crops/cropped_161.jpg',cropped_161)
    cv2.imwrite('crops/cropped_162.jpg',cropped_162)
    cv2.imwrite('crops/cropped_163.jpg',cropped_163)
    cv2.imwrite('crops/cropped_164.jpg',cropped_164)
    cv2.imwrite('crops/cropped_165.jpg',cropped_165)
    cv2.imwrite('crops/cropped_166.jpg',cropped_166)
    cv2.imwrite('crops/cropped_167.jpg',cropped_167)
    cv2.imwrite('crops/cropped_168.jpg',cropped_168)
    cv2.imwrite('crops/cropped_169.jpg',cropped_169)
    cv2.imwrite('crops/cropped_170.jpg',cropped_170)
    cv2.imwrite('crops/cropped_171.jpg',cropped_171)
    cv2.imwrite('crops/cropped_172.jpg',cropped_172)
    cv2.imwrite('crops/cropped_173.jpg',cropped_173)
    cv2.imwrite('crops/cropped_174.jpg',cropped_174)
    cv2.imwrite('crops/cropped_175.jpg',cropped_175)
    cv2.imwrite('crops/cropped_176.jpg',cropped_176)
    cv2.imwrite('crops/cropped_177.jpg',cropped_177)
    cv2.imwrite('crops/cropped_178.jpg',cropped_178)
    cv2.imwrite('crops/cropped_179.jpg',cropped_179)
    cv2.imwrite('crops/cropped_180.jpg',cropped_180)
    cv2.imwrite('crops/cropped_181.jpg',cropped_181)
    cv2.imwrite('crops/cropped_182.jpg',cropped_182)
    cv2.imwrite('crops/cropped_183.jpg',cropped_183)
    cv2.imwrite('crops/cropped_184.jpg',cropped_184)
    cv2.imwrite('crops/cropped_185.jpg',cropped_185)
    cv2.imwrite('crops/cropped_186.jpg',cropped_186)
    cv2.imwrite('crops/cropped_187.jpg',cropped_187)
    cv2.imwrite('crops/cropped_188.jpg',cropped_188)
    cv2.imwrite('crops/cropped_189.jpg',cropped_189)
    cv2.imwrite('crops/cropped_190.jpg',cropped_190)
    cv2.imwrite('crops/cropped_191.jpg',cropped_191)
    cv2.imwrite('crops/cropped_192.jpg',cropped_192)
    cv2.imwrite('crops/cropped_193.jpg',cropped_193)
    cv2.imwrite('crops/cropped_194.jpg',cropped_194)
    cv2.imwrite('crops/cropped_195.jpg',cropped_195)
    cv2.imwrite('crops/cropped_196.jpg',cropped_196)
    cv2.imwrite('crops/cropped_197.jpg',cropped_197)
    cv2.imwrite('crops/cropped_198.jpg',cropped_198)
    cv2.imwrite('crops/cropped_199.jpg',cropped_199)
    cv2.imwrite('crops/cropped_200.jpg',cropped_200)
    cv2.imwrite('crops/cropped_201.jpg',cropped_201)
    cv2.imwrite('crops/cropped_202.jpg',cropped_202)
    cv2.imwrite('crops/cropped_203.jpg',cropped_203)
    cv2.imwrite('crops/cropped_204.jpg',cropped_204)
    cv2.imwrite('crops/cropped_205.jpg',cropped_205)
    cv2.imwrite('crops/cropped_206.jpg',cropped_206)
    cv2.imwrite('crops/cropped_207.jpg',cropped_207)
    cv2.imwrite('crops/cropped_208.jpg',cropped_208)
    cv2.imwrite('crops/cropped_209.jpg',cropped_209)
    cv2.imwrite('crops/cropped_210.jpg',cropped_210)
    cv2.imwrite('crops/cropped_211.jpg',cropped_211)
    cv2.imwrite('crops/cropped_212.jpg',cropped_212)
    cv2.imwrite('crops/cropped_213.jpg',cropped_213)
    cv2.imwrite('crops/cropped_214.jpg',cropped_214)
    cv2.imwrite('crops/cropped_215.jpg',cropped_215)
    cv2.imwrite('crops/cropped_216.jpg',cropped_216)
    cv2.imwrite('crops/cropped_217.jpg',cropped_217)
    cv2.imwrite('crops/cropped_218.jpg',cropped_218)
    cv2.imwrite('crops/cropped_219.jpg',cropped_219)
    cv2.imwrite('crops/cropped_220.jpg',cropped_220)
    cv2.imwrite('crops/cropped_221.jpg',cropped_221)
    cv2.imwrite('crops/cropped_222.jpg',cropped_222)
    cv2.imwrite('crops/cropped_223.jpg',cropped_223)
    cv2.imwrite('crops/cropped_224.jpg',cropped_224)
    cv2.imwrite('crops/cropped_225.jpg',cropped_225)
    cv2.imwrite('crops/cropped_226.jpg',cropped_226)
    cv2.imwrite('crops/cropped_227.jpg',cropped_227)
    cv2.imwrite('crops/cropped_228.jpg',cropped_228)
    cv2.imwrite('crops/cropped_229.jpg',cropped_229)
    cv2.imwrite('crops/cropped_230.jpg',cropped_230)
    cv2.imwrite('crops/cropped_231.jpg',cropped_231)
    cv2.imwrite('crops/cropped_232.jpg',cropped_232)
    cv2.imwrite('crops/cropped_233.jpg',cropped_233)
    cv2.imwrite('crops/cropped_234.jpg',cropped_234)
    cv2.imwrite('crops/cropped_235.jpg',cropped_235)
    cv2.imwrite('crops/cropped_236.jpg',cropped_236)
    cv2.imwrite('crops/cropped_237.jpg',cropped_237)
    cv2.imwrite('crops/cropped_238.jpg',cropped_238)
    cv2.imwrite('crops/cropped_239.jpg',cropped_239)
    cv2.imwrite('crops/cropped_240.jpg',cropped_240)
    cv2.imwrite('crops/cropped_241.jpg',cropped_241)
    cv2.imwrite('crops/cropped_242.jpg',cropped_242)
    cv2.imwrite('crops/cropped_243.jpg',cropped_243)
    cv2.imwrite('crops/cropped_244.jpg',cropped_244)
    cv2.imwrite('crops/cropped_245.jpg',cropped_245)
    cv2.imwrite('crops/cropped_246.jpg',cropped_246)
    cv2.imwrite('crops/cropped_247.jpg',cropped_247)
    cv2.imwrite('crops/cropped_248.jpg',cropped_248)
    cv2.imwrite('crops/cropped_249.jpg',cropped_249)
    cv2.imwrite('crops/cropped_250.jpg',cropped_250)
    cv2.imwrite('crops/cropped_251.jpg',cropped_251)
    cv2.imwrite('crops/cropped_252.jpg',cropped_252)
    cv2.imwrite('crops/cropped_253.jpg',cropped_253)
    cv2.imwrite('crops/cropped_254.jpg',cropped_254)
    cv2.imwrite('crops/cropped_255.jpg',cropped_255)
    cv2.imwrite('crops/cropped_256.jpg',cropped_256)
    cv2.imwrite('crops/cropped_257.jpg',cropped_257)
    cv2.imwrite('crops/cropped_258.jpg',cropped_258)
    cv2.imwrite('crops/cropped_259.jpg',cropped_259)
    cv2.imwrite('crops/cropped_260.jpg',cropped_260)
    cv2.imwrite('crops/cropped_261.jpg',cropped_261)
    cv2.imwrite('crops/cropped_262.jpg',cropped_262)
    cv2.imwrite('crops/cropped_263.jpg',cropped_263)
    cv2.imwrite('crops/cropped_264.jpg',cropped_264)
    cv2.imwrite('crops/cropped_265.jpg',cropped_265)
    cv2.imwrite('crops/cropped_266.jpg',cropped_266)
    cv2.imwrite('crops/cropped_267.jpg',cropped_267)
    cv2.imwrite('crops/cropped_268.jpg',cropped_268)
    cv2.imwrite('crops/cropped_269.jpg',cropped_269)
    cv2.imwrite('crops/cropped_270.jpg',cropped_270)
    cv2.imwrite('crops/cropped_271.jpg',cropped_271)
    cv2.imwrite('crops/cropped_272.jpg',cropped_272)
    cv2.imwrite('crops/cropped_273.jpg',cropped_273)
    cv2.imwrite('crops/cropped_274.jpg',cropped_274)
    cv2.imwrite('crops/cropped_275.jpg',cropped_275)
    cv2.imwrite('crops/cropped_276.jpg',cropped_276)
    cv2.imwrite('crops/cropped_277.jpg',cropped_277)
    cv2.imwrite('crops/cropped_278.jpg',cropped_278)
    cv2.imwrite('crops/cropped_279.jpg',cropped_279)
    cv2.imwrite('crops/cropped_280.jpg',cropped_280)

    #writing for crops22
    cv2.imwrite('crops2/cropped_1.jpg',cropped2_1)
    cv2.imwrite('crops2/cropped_2.jpg',cropped2_2)
    cv2.imwrite('crops2/cropped_3.jpg',cropped2_3)
    cv2.imwrite('crops2/cropped_4.jpg',cropped2_4)
    cv2.imwrite('crops2/cropped_5.jpg',cropped2_5)
    cv2.imwrite('crops2/cropped_6.jpg',cropped2_6)
    cv2.imwrite('crops2/cropped_7.jpg',cropped2_7)
    cv2.imwrite('crops2/cropped_8.jpg',cropped2_8)
    cv2.imwrite('crops2/cropped_9.jpg',cropped2_9)
    cv2.imwrite('crops2/cropped_10.jpg',cropped2_10)
    cv2.imwrite('crops2/cropped_11.jpg',cropped2_11)
    cv2.imwrite('crops2/cropped_12.jpg',cropped2_12)
    cv2.imwrite('crops2/cropped_13.jpg',cropped2_13)
    cv2.imwrite('crops2/cropped_14.jpg',cropped2_14)
    cv2.imwrite('crops2/cropped_15.jpg',cropped2_15)
    cv2.imwrite('crops2/cropped_16.jpg',cropped2_16)
    cv2.imwrite('crops2/cropped_17.jpg',cropped2_17)
    cv2.imwrite('crops2/cropped_18.jpg',cropped2_18)
    cv2.imwrite('crops2/cropped_19.jpg',cropped2_19)
    cv2.imwrite('crops2/cropped_20.jpg',cropped2_20)
    cv2.imwrite('crops2/cropped_21.jpg',cropped2_21)
    cv2.imwrite('crops2/cropped_22.jpg',cropped2_22)
    cv2.imwrite('crops2/cropped_23.jpg',cropped2_23)
    cv2.imwrite('crops2/cropped_24.jpg',cropped2_24)
    cv2.imwrite('crops2/cropped_25.jpg',cropped2_25)
    cv2.imwrite('crops2/cropped_26.jpg',cropped2_26)
    cv2.imwrite('crops2/cropped_27.jpg',cropped2_27)
    cv2.imwrite('crops2/cropped_28.jpg',cropped2_28)
    cv2.imwrite('crops2/cropped_29.jpg',cropped2_29)
    cv2.imwrite('crops2/cropped_30.jpg',cropped2_30)
    cv2.imwrite('crops2/cropped_31.jpg',cropped2_31)
    cv2.imwrite('crops2/cropped_32.jpg',cropped2_32)
    cv2.imwrite('crops2/cropped_33.jpg',cropped2_33)
    cv2.imwrite('crops2/cropped_34.jpg',cropped2_34)
    cv2.imwrite('crops2/cropped_35.jpg',cropped2_35)
    cv2.imwrite('crops2/cropped_36.jpg',cropped2_36)
    cv2.imwrite('crops2/cropped_37.jpg',cropped2_37)
    cv2.imwrite('crops2/cropped_38.jpg',cropped2_38)
    cv2.imwrite('crops2/cropped_39.jpg',cropped2_39)
    cv2.imwrite('crops2/cropped_40.jpg',cropped2_40)
    cv2.imwrite('crops2/cropped_41.jpg',cropped2_41)
    cv2.imwrite('crops2/cropped_42.jpg',cropped2_42)
    cv2.imwrite('crops2/cropped_43.jpg',cropped2_43)
    cv2.imwrite('crops2/cropped_44.jpg',cropped2_44)
    cv2.imwrite('crops2/cropped_45.jpg',cropped2_45)
    cv2.imwrite('crops2/cropped_46.jpg',cropped2_46)
    cv2.imwrite('crops2/cropped_47.jpg',cropped2_47)
    cv2.imwrite('crops2/cropped_48.jpg',cropped2_48)
    cv2.imwrite('crops2/cropped_49.jpg',cropped2_49)
    cv2.imwrite('crops2/cropped_50.jpg',cropped2_50)
    cv2.imwrite('crops2/cropped_51.jpg',cropped2_51)
    cv2.imwrite('crops2/cropped_52.jpg',cropped2_52)
    cv2.imwrite('crops2/cropped_53.jpg',cropped2_53)
    cv2.imwrite('crops2/cropped_54.jpg',cropped2_54)
    cv2.imwrite('crops2/cropped_55.jpg',cropped2_55)
    cv2.imwrite('crops2/cropped_56.jpg',cropped2_56)
    cv2.imwrite('crops2/cropped_57.jpg',cropped2_57)
    cv2.imwrite('crops2/cropped_58.jpg',cropped2_58)
    cv2.imwrite('crops2/cropped_59.jpg',cropped2_59)
    cv2.imwrite('crops2/cropped_60.jpg',cropped2_60)
    cv2.imwrite('crops2/cropped_61.jpg',cropped2_61)
    cv2.imwrite('crops2/cropped_62.jpg',cropped2_62)
    cv2.imwrite('crops2/cropped_63.jpg',cropped2_63)
    cv2.imwrite('crops2/cropped_64.jpg',cropped2_64)
    cv2.imwrite('crops2/cropped_65.jpg',cropped2_65)
    cv2.imwrite('crops2/cropped_66.jpg',cropped2_66)
    cv2.imwrite('crops2/cropped_67.jpg',cropped2_67)
    cv2.imwrite('crops2/cropped_68.jpg',cropped2_68)
    cv2.imwrite('crops2/cropped_69.jpg',cropped2_69)
    cv2.imwrite('crops2/cropped_70.jpg',cropped2_70)
    cv2.imwrite('crops2/cropped_71.jpg',cropped2_71)
    cv2.imwrite('crops2/cropped_72.jpg',cropped2_72)
    cv2.imwrite('crops2/cropped_73.jpg',cropped2_73)
    cv2.imwrite('crops2/cropped_74.jpg',cropped2_74)
    cv2.imwrite('crops2/cropped_75.jpg',cropped2_75)
    cv2.imwrite('crops2/cropped_76.jpg',cropped2_76)
    cv2.imwrite('crops2/cropped_77.jpg',cropped2_77)
    cv2.imwrite('crops2/cropped_78.jpg',cropped2_78)
    cv2.imwrite('crops2/cropped_79.jpg',cropped2_79)
    cv2.imwrite('crops2/cropped_80.jpg',cropped2_80)
    cv2.imwrite('crops2/cropped_81.jpg',cropped2_81)
    cv2.imwrite('crops2/cropped_82.jpg',cropped2_82)
    cv2.imwrite('crops2/cropped_83.jpg',cropped2_83)
    cv2.imwrite('crops2/cropped_84.jpg',cropped2_84)
    cv2.imwrite('crops2/cropped_85.jpg',cropped2_85)
    cv2.imwrite('crops2/cropped_86.jpg',cropped2_86)
    cv2.imwrite('crops2/cropped_87.jpg',cropped2_87)
    cv2.imwrite('crops2/cropped_88.jpg',cropped2_88)
    cv2.imwrite('crops2/cropped_89.jpg',cropped2_89)
    cv2.imwrite('crops2/cropped_90.jpg',cropped2_90)
    cv2.imwrite('crops2/cropped_91.jpg',cropped2_91)
    cv2.imwrite('crops2/cropped_92.jpg',cropped2_92)
    cv2.imwrite('crops2/cropped_93.jpg',cropped2_93)
    cv2.imwrite('crops2/cropped_94.jpg',cropped2_94)
    cv2.imwrite('crops2/cropped_95.jpg',cropped2_95)
    cv2.imwrite('crops2/cropped_96.jpg',cropped2_96)
    cv2.imwrite('crops2/cropped_97.jpg',cropped2_97)
    cv2.imwrite('crops2/cropped_98.jpg',cropped2_98)
    cv2.imwrite('crops2/cropped_99.jpg',cropped2_99)
    cv2.imwrite('crops2/cropped_100.jpg',cropped2_100)
    cv2.imwrite('crops2/cropped_101.jpg',cropped2_101)
    cv2.imwrite('crops2/cropped_102.jpg',cropped2_102)
    cv2.imwrite('crops2/cropped_103.jpg',cropped2_103)
    cv2.imwrite('crops2/cropped_104.jpg',cropped2_104)
    cv2.imwrite('crops2/cropped_105.jpg',cropped2_105)
    cv2.imwrite('crops2/cropped_106.jpg',cropped2_106)
    cv2.imwrite('crops2/cropped_107.jpg',cropped2_107)
    cv2.imwrite('crops2/cropped_108.jpg',cropped2_108)
    cv2.imwrite('crops2/cropped_109.jpg',cropped2_109)
    cv2.imwrite('crops2/cropped_110.jpg',cropped2_110)
    cv2.imwrite('crops2/cropped_111.jpg',cropped2_111)
    cv2.imwrite('crops2/cropped_112.jpg',cropped2_112)
    cv2.imwrite('crops2/cropped_113.jpg',cropped2_113)
    cv2.imwrite('crops2/cropped_114.jpg',cropped2_114)
    cv2.imwrite('crops2/cropped_115.jpg',cropped2_115)
    cv2.imwrite('crops2/cropped_116.jpg',cropped2_116)
    cv2.imwrite('crops2/cropped_117.jpg',cropped2_117)
    cv2.imwrite('crops2/cropped_118.jpg',cropped2_118)
    cv2.imwrite('crops2/cropped_119.jpg',cropped2_119)
    cv2.imwrite('crops2/cropped_120.jpg',cropped2_120)
    cv2.imwrite('crops2/cropped_121.jpg',cropped2_121)
    cv2.imwrite('crops2/cropped_122.jpg',cropped2_122)
    cv2.imwrite('crops2/cropped_123.jpg',cropped2_123)
    cv2.imwrite('crops2/cropped_124.jpg',cropped2_124)
    cv2.imwrite('crops2/cropped_125.jpg',cropped2_125)
    cv2.imwrite('crops2/cropped_126.jpg',cropped2_126)
    cv2.imwrite('crops2/cropped_127.jpg',cropped2_127)
    cv2.imwrite('crops2/cropped_128.jpg',cropped2_128)
    cv2.imwrite('crops2/cropped_129.jpg',cropped2_129)
    cv2.imwrite('crops2/cropped_130.jpg',cropped2_130)
    cv2.imwrite('crops2/cropped_131.jpg',cropped2_131)
    cv2.imwrite('crops2/cropped_132.jpg',cropped2_132)
    cv2.imwrite('crops2/cropped_133.jpg',cropped2_133)
    cv2.imwrite('crops2/cropped_134.jpg',cropped2_134)
    cv2.imwrite('crops2/cropped_135.jpg',cropped2_135)
    cv2.imwrite('crops2/cropped_136.jpg',cropped2_136)
    cv2.imwrite('crops2/cropped_137.jpg',cropped2_137)
    cv2.imwrite('crops2/cropped_138.jpg',cropped2_138)
    cv2.imwrite('crops2/cropped_139.jpg',cropped2_139)
    cv2.imwrite('crops2/cropped_140.jpg',cropped2_140)
    cv2.imwrite('crops2/cropped_141.jpg',cropped2_141)
    cv2.imwrite('crops2/cropped_142.jpg',cropped2_142)
    cv2.imwrite('crops2/cropped_143.jpg',cropped2_143)
    cv2.imwrite('crops2/cropped_144.jpg',cropped2_144)
    cv2.imwrite('crops2/cropped_145.jpg',cropped2_145)
    cv2.imwrite('crops2/cropped_146.jpg',cropped2_146)
    cv2.imwrite('crops2/cropped_147.jpg',cropped2_147)
    cv2.imwrite('crops2/cropped_148.jpg',cropped2_148)
    cv2.imwrite('crops2/cropped_149.jpg',cropped2_149)
    cv2.imwrite('crops2/cropped_150.jpg',cropped2_150)
    cv2.imwrite('crops2/cropped_151.jpg',cropped2_151)
    cv2.imwrite('crops2/cropped_152.jpg',cropped2_152)
    cv2.imwrite('crops2/cropped_153.jpg',cropped2_153)
    cv2.imwrite('crops2/cropped_154.jpg',cropped2_154)
    cv2.imwrite('crops2/cropped_155.jpg',cropped2_155)
    cv2.imwrite('crops2/cropped_156.jpg',cropped2_156)
    cv2.imwrite('crops2/cropped_157.jpg',cropped2_157)
    cv2.imwrite('crops2/cropped_158.jpg',cropped2_158)
    cv2.imwrite('crops2/cropped_159.jpg',cropped2_159)
    cv2.imwrite('crops2/cropped_160.jpg',cropped2_160)
    cv2.imwrite('crops2/cropped_161.jpg',cropped2_161)
    cv2.imwrite('crops2/cropped_162.jpg',cropped2_162)
    cv2.imwrite('crops2/cropped_163.jpg',cropped2_163)
    cv2.imwrite('crops2/cropped_164.jpg',cropped2_164)
    cv2.imwrite('crops2/cropped_165.jpg',cropped2_165)
    cv2.imwrite('crops2/cropped_166.jpg',cropped2_166)
    cv2.imwrite('crops2/cropped_167.jpg',cropped2_167)
    cv2.imwrite('crops2/cropped_168.jpg',cropped2_168)
    cv2.imwrite('crops2/cropped_169.jpg',cropped2_169)
    cv2.imwrite('crops2/cropped_170.jpg',cropped2_170)
    cv2.imwrite('crops2/cropped_171.jpg',cropped2_171)
    cv2.imwrite('crops2/cropped_172.jpg',cropped2_172)
    cv2.imwrite('crops2/cropped_173.jpg',cropped2_173)
    cv2.imwrite('crops2/cropped_174.jpg',cropped2_174)
    cv2.imwrite('crops2/cropped_175.jpg',cropped2_175)
    cv2.imwrite('crops2/cropped_176.jpg',cropped2_176)
    cv2.imwrite('crops2/cropped_177.jpg',cropped2_177)
    cv2.imwrite('crops2/cropped_178.jpg',cropped2_178)
    cv2.imwrite('crops2/cropped_179.jpg',cropped2_179)
    cv2.imwrite('crops2/cropped_180.jpg',cropped2_180)
    cv2.imwrite('crops2/cropped_181.jpg',cropped2_181)
    cv2.imwrite('crops2/cropped_182.jpg',cropped2_182)
    cv2.imwrite('crops2/cropped_183.jpg',cropped2_183)
    cv2.imwrite('crops2/cropped_184.jpg',cropped2_184)
    cv2.imwrite('crops2/cropped_185.jpg',cropped2_185)
    cv2.imwrite('crops2/cropped_186.jpg',cropped2_186)
    cv2.imwrite('crops2/cropped_187.jpg',cropped2_187)
    cv2.imwrite('crops2/cropped_188.jpg',cropped2_188)
    cv2.imwrite('crops2/cropped_189.jpg',cropped2_189)
    cv2.imwrite('crops2/cropped_190.jpg',cropped2_190)
    cv2.imwrite('crops2/cropped_191.jpg',cropped2_191)
    cv2.imwrite('crops2/cropped_192.jpg',cropped2_192)
    cv2.imwrite('crops2/cropped_193.jpg',cropped2_193)
    cv2.imwrite('crops2/cropped_194.jpg',cropped2_194)
    cv2.imwrite('crops2/cropped_195.jpg',cropped2_195)
    cv2.imwrite('crops2/cropped_196.jpg',cropped2_196)
    cv2.imwrite('crops2/cropped_197.jpg',cropped2_197)
    cv2.imwrite('crops2/cropped_198.jpg',cropped2_198)
    cv2.imwrite('crops2/cropped_199.jpg',cropped2_199)
    cv2.imwrite('crops2/cropped_200.jpg',cropped2_200)
    cv2.imwrite('crops2/cropped_201.jpg',cropped2_201)
    cv2.imwrite('crops2/cropped_202.jpg',cropped2_202)
    cv2.imwrite('crops2/cropped_203.jpg',cropped2_203)
    cv2.imwrite('crops2/cropped_204.jpg',cropped2_204)
    cv2.imwrite('crops2/cropped_205.jpg',cropped2_205)
    cv2.imwrite('crops2/cropped_206.jpg',cropped2_206)
    cv2.imwrite('crops2/cropped_207.jpg',cropped2_207)
    cv2.imwrite('crops2/cropped_208.jpg',cropped2_208)
    cv2.imwrite('crops2/cropped_209.jpg',cropped2_209)
    cv2.imwrite('crops2/cropped_210.jpg',cropped2_210)
    cv2.imwrite('crops2/cropped_211.jpg',cropped2_211)
    cv2.imwrite('crops2/cropped_212.jpg',cropped2_212)
    cv2.imwrite('crops2/cropped_213.jpg',cropped2_213)
    cv2.imwrite('crops2/cropped_214.jpg',cropped2_214)
    cv2.imwrite('crops2/cropped_215.jpg',cropped2_215)
    cv2.imwrite('crops2/cropped_216.jpg',cropped2_216)
    cv2.imwrite('crops2/cropped_217.jpg',cropped2_217)
    cv2.imwrite('crops2/cropped_218.jpg',cropped2_218)
    cv2.imwrite('crops2/cropped_219.jpg',cropped2_219)
    cv2.imwrite('crops2/cropped_220.jpg',cropped2_220)
    cv2.imwrite('crops2/cropped_221.jpg',cropped2_221)
    cv2.imwrite('crops2/cropped_222.jpg',cropped2_222)
    cv2.imwrite('crops2/cropped_223.jpg',cropped2_223)
    cv2.imwrite('crops2/cropped_224.jpg',cropped2_224)
    cv2.imwrite('crops2/cropped_225.jpg',cropped2_225)
    cv2.imwrite('crops2/cropped_226.jpg',cropped2_226)
    cv2.imwrite('crops2/cropped_227.jpg',cropped2_227)
    cv2.imwrite('crops2/cropped_228.jpg',cropped2_228)
    cv2.imwrite('crops2/cropped_229.jpg',cropped2_229)
    cv2.imwrite('crops2/cropped_230.jpg',cropped2_230)
    cv2.imwrite('crops2/cropped_231.jpg',cropped2_231)
    cv2.imwrite('crops2/cropped_232.jpg',cropped2_232)
    cv2.imwrite('crops2/cropped_233.jpg',cropped2_233)
    cv2.imwrite('crops2/cropped_234.jpg',cropped2_234)
    cv2.imwrite('crops2/cropped_235.jpg',cropped2_235)
    cv2.imwrite('crops2/cropped_236.jpg',cropped2_236)
    cv2.imwrite('crops2/cropped_237.jpg',cropped2_237)
    cv2.imwrite('crops2/cropped_238.jpg',cropped2_238)
    cv2.imwrite('crops2/cropped_239.jpg',cropped2_239)
    cv2.imwrite('crops2/cropped_240.jpg',cropped2_240)
    cv2.imwrite('crops2/cropped_241.jpg',cropped2_241)
    cv2.imwrite('crops2/cropped_242.jpg',cropped2_242)
    cv2.imwrite('crops2/cropped_243.jpg',cropped2_243)
    cv2.imwrite('crops2/cropped_244.jpg',cropped2_244)
    cv2.imwrite('crops2/cropped_245.jpg',cropped2_245)
    cv2.imwrite('crops2/cropped_246.jpg',cropped2_246)
    cv2.imwrite('crops2/cropped_247.jpg',cropped2_247)
    cv2.imwrite('crops2/cropped_248.jpg',cropped2_248)
    cv2.imwrite('crops2/cropped_249.jpg',cropped2_249)
    cv2.imwrite('crops2/cropped_250.jpg',cropped2_250)
    cv2.imwrite('crops2/cropped_251.jpg',cropped2_251)
    cv2.imwrite('crops2/cropped_252.jpg',cropped2_252)
    cv2.imwrite('crops2/cropped_253.jpg',cropped2_253)
    cv2.imwrite('crops2/cropped_254.jpg',cropped2_254)
    cv2.imwrite('crops2/cropped_255.jpg',cropped2_255)
    cv2.imwrite('crops2/cropped_256.jpg',cropped2_256)
    cv2.imwrite('crops2/cropped_257.jpg',cropped2_257)
    cv2.imwrite('crops2/cropped_258.jpg',cropped2_258)
    cv2.imwrite('crops2/cropped_259.jpg',cropped2_259)
    cv2.imwrite('crops2/cropped_260.jpg',cropped2_260)
    cv2.imwrite('crops2/cropped_261.jpg',cropped2_261)
    cv2.imwrite('crops2/cropped_262.jpg',cropped2_262)
    cv2.imwrite('crops2/cropped_263.jpg',cropped2_263)
    cv2.imwrite('crops2/cropped_264.jpg',cropped2_264)
    cv2.imwrite('crops2/cropped_265.jpg',cropped2_265)
    cv2.imwrite('crops2/cropped_266.jpg',cropped2_266)
    cv2.imwrite('crops2/cropped_267.jpg',cropped2_267)
    cv2.imwrite('crops2/cropped_268.jpg',cropped2_268)
    cv2.imwrite('crops2/cropped_269.jpg',cropped2_269)
    cv2.imwrite('crops2/cropped_270.jpg',cropped2_270)
    cv2.imwrite('crops2/cropped_271.jpg',cropped2_271)
    cv2.imwrite('crops2/cropped_272.jpg',cropped2_272)
    cv2.imwrite('crops2/cropped_273.jpg',cropped2_273)
    cv2.imwrite('crops2/cropped_274.jpg',cropped2_274)
    cv2.imwrite('crops2/cropped_275.jpg',cropped2_275)
    cv2.imwrite('crops2/cropped_276.jpg',cropped2_276)
    cv2.imwrite('crops2/cropped_277.jpg',cropped2_277)
    cv2.imwrite('crops2/cropped_278.jpg',cropped2_278)
    cv2.imwrite('crops2/cropped_279.jpg',cropped2_279)
    cv2.imwrite('crops2/cropped_280.jpg',cropped2_280)

    #writing for crops3
    cv2.imwrite('crops3/cropped_1.jpg',cropped3_1)
    cv2.imwrite('crops3/cropped_2.jpg',cropped3_2)
    cv2.imwrite('crops3/cropped_3.jpg',cropped3_3)
    cv2.imwrite('crops3/cropped_4.jpg',cropped3_4)
    cv2.imwrite('crops3/cropped_5.jpg',cropped3_5)
    cv2.imwrite('crops3/cropped_6.jpg',cropped3_6)
    cv2.imwrite('crops3/cropped_7.jpg',cropped3_7)
    cv2.imwrite('crops3/cropped_8.jpg',cropped3_8)
    cv2.imwrite('crops3/cropped_9.jpg',cropped3_9)
    cv2.imwrite('crops3/cropped_10.jpg',cropped3_10)
    cv2.imwrite('crops3/cropped_11.jpg',cropped3_11)
    cv2.imwrite('crops3/cropped_12.jpg',cropped3_12)
    cv2.imwrite('crops3/cropped_13.jpg',cropped3_13)
    cv2.imwrite('crops3/cropped_14.jpg',cropped3_14)
    cv2.imwrite('crops3/cropped_15.jpg',cropped3_15)
    cv2.imwrite('crops3/cropped_16.jpg',cropped3_16)
    cv2.imwrite('crops3/cropped_17.jpg',cropped3_17)
    cv2.imwrite('crops3/cropped_18.jpg',cropped3_18)
    cv2.imwrite('crops3/cropped_19.jpg',cropped3_19)
    cv2.imwrite('crops3/cropped_20.jpg',cropped3_20)
    cv2.imwrite('crops3/cropped_21.jpg',cropped3_21)
    cv2.imwrite('crops3/cropped_22.jpg',cropped3_22)
    cv2.imwrite('crops3/cropped_23.jpg',cropped3_23)
    cv2.imwrite('crops3/cropped_24.jpg',cropped3_24)
    cv2.imwrite('crops3/cropped_25.jpg',cropped3_25)
    cv2.imwrite('crops3/cropped_26.jpg',cropped3_26)
    cv2.imwrite('crops3/cropped_27.jpg',cropped3_27)
    cv2.imwrite('crops3/cropped_28.jpg',cropped3_28)
    cv2.imwrite('crops3/cropped_29.jpg',cropped3_29)
    cv2.imwrite('crops3/cropped_30.jpg',cropped3_30)
    cv2.imwrite('crops3/cropped_31.jpg',cropped3_31)
    cv2.imwrite('crops3/cropped_32.jpg',cropped3_32)
    cv2.imwrite('crops3/cropped_33.jpg',cropped3_33)
    cv2.imwrite('crops3/cropped_34.jpg',cropped3_34)
    cv2.imwrite('crops3/cropped_35.jpg',cropped3_35)
    cv2.imwrite('crops3/cropped_36.jpg',cropped3_36)
    cv2.imwrite('crops3/cropped_37.jpg',cropped3_37)
    cv2.imwrite('crops3/cropped_38.jpg',cropped3_38)
    cv2.imwrite('crops3/cropped_39.jpg',cropped3_39)
    cv2.imwrite('crops3/cropped_40.jpg',cropped3_40)
    cv2.imwrite('crops3/cropped_41.jpg',cropped3_41)
    cv2.imwrite('crops3/cropped_42.jpg',cropped3_42)
    cv2.imwrite('crops3/cropped_43.jpg',cropped3_43)
    cv2.imwrite('crops3/cropped_44.jpg',cropped3_44)
    cv2.imwrite('crops3/cropped_45.jpg',cropped3_45)
    cv2.imwrite('crops3/cropped_46.jpg',cropped3_46)
    cv2.imwrite('crops3/cropped_47.jpg',cropped3_47)
    cv2.imwrite('crops3/cropped_48.jpg',cropped3_48)
    cv2.imwrite('crops3/cropped_49.jpg',cropped3_49)
    cv2.imwrite('crops3/cropped_50.jpg',cropped3_50)
    cv2.imwrite('crops3/cropped_51.jpg',cropped3_51)
    cv2.imwrite('crops3/cropped_52.jpg',cropped3_52)
    cv2.imwrite('crops3/cropped_53.jpg',cropped3_53)
    cv2.imwrite('crops3/cropped_54.jpg',cropped3_54)
    cv2.imwrite('crops3/cropped_55.jpg',cropped3_55)
    cv2.imwrite('crops3/cropped_56.jpg',cropped3_56)
    cv2.imwrite('crops3/cropped_57.jpg',cropped3_57)
    cv2.imwrite('crops3/cropped_58.jpg',cropped3_58)
    cv2.imwrite('crops3/cropped_59.jpg',cropped3_59)
    cv2.imwrite('crops3/cropped_60.jpg',cropped3_60)
    cv2.imwrite('crops3/cropped_61.jpg',cropped3_61)
    cv2.imwrite('crops3/cropped_62.jpg',cropped3_62)
    cv2.imwrite('crops3/cropped_63.jpg',cropped3_63)
    cv2.imwrite('crops3/cropped_64.jpg',cropped3_64)
    cv2.imwrite('crops3/cropped_65.jpg',cropped3_65)
    cv2.imwrite('crops3/cropped_66.jpg',cropped3_66)
    cv2.imwrite('crops3/cropped_67.jpg',cropped3_67)
    cv2.imwrite('crops3/cropped_68.jpg',cropped3_68)
    cv2.imwrite('crops3/cropped_69.jpg',cropped3_69)
    cv2.imwrite('crops3/cropped_70.jpg',cropped3_70)
    cv2.imwrite('crops3/cropped_71.jpg',cropped3_71)
    cv2.imwrite('crops3/cropped_72.jpg',cropped3_72)
    cv2.imwrite('crops3/cropped_73.jpg',cropped3_73)
    cv2.imwrite('crops3/cropped_74.jpg',cropped3_74)
    cv2.imwrite('crops3/cropped_75.jpg',cropped3_75)
    cv2.imwrite('crops3/cropped_76.jpg',cropped3_76)
    cv2.imwrite('crops3/cropped_77.jpg',cropped3_77)
    cv2.imwrite('crops3/cropped_78.jpg',cropped3_78)
    cv2.imwrite('crops3/cropped_79.jpg',cropped3_79)
    cv2.imwrite('crops3/cropped_80.jpg',cropped3_80)
    cv2.imwrite('crops3/cropped_81.jpg',cropped3_81)
    cv2.imwrite('crops3/cropped_82.jpg',cropped3_82)
    cv2.imwrite('crops3/cropped_83.jpg',cropped3_83)
    cv2.imwrite('crops3/cropped_84.jpg',cropped3_84)
    cv2.imwrite('crops3/cropped_85.jpg',cropped3_85)
    cv2.imwrite('crops3/cropped_86.jpg',cropped3_86)
    cv2.imwrite('crops3/cropped_87.jpg',cropped3_87)
    cv2.imwrite('crops3/cropped_88.jpg',cropped3_88)
    cv2.imwrite('crops3/cropped_89.jpg',cropped3_89)
    cv2.imwrite('crops3/cropped_90.jpg',cropped3_90)
    cv2.imwrite('crops3/cropped_91.jpg',cropped3_91)
    cv2.imwrite('crops3/cropped_92.jpg',cropped3_92)
    cv2.imwrite('crops3/cropped_93.jpg',cropped3_93)
    cv2.imwrite('crops3/cropped_94.jpg',cropped3_94)
    cv2.imwrite('crops3/cropped_95.jpg',cropped3_95)
    cv2.imwrite('crops3/cropped_96.jpg',cropped3_96)
    cv2.imwrite('crops3/cropped_97.jpg',cropped3_97)
    cv2.imwrite('crops3/cropped_98.jpg',cropped3_98)
    cv2.imwrite('crops3/cropped_99.jpg',cropped3_99)
    cv2.imwrite('crops3/cropped_100.jpg',cropped3_100)
    cv2.imwrite('crops3/cropped_101.jpg',cropped3_101)
    cv2.imwrite('crops3/cropped_102.jpg',cropped3_102)
    cv2.imwrite('crops3/cropped_103.jpg',cropped3_103)
    cv2.imwrite('crops3/cropped_104.jpg',cropped3_104)
    cv2.imwrite('crops3/cropped_105.jpg',cropped3_105)
    cv2.imwrite('crops3/cropped_106.jpg',cropped3_106)
    cv2.imwrite('crops3/cropped_107.jpg',cropped3_107)
    cv2.imwrite('crops3/cropped_108.jpg',cropped3_108)
    cv2.imwrite('crops3/cropped_109.jpg',cropped3_109)
    cv2.imwrite('crops3/cropped_110.jpg',cropped3_110)
    cv2.imwrite('crops3/cropped_111.jpg',cropped3_111)
    cv2.imwrite('crops3/cropped_112.jpg',cropped3_112)
    cv2.imwrite('crops3/cropped_113.jpg',cropped3_113)
    cv2.imwrite('crops3/cropped_114.jpg',cropped3_114)
    cv2.imwrite('crops3/cropped_115.jpg',cropped3_115)
    cv2.imwrite('crops3/cropped_116.jpg',cropped3_116)
    cv2.imwrite('crops3/cropped_117.jpg',cropped3_117)
    cv2.imwrite('crops3/cropped_118.jpg',cropped3_118)
    cv2.imwrite('crops3/cropped_119.jpg',cropped3_119)
    cv2.imwrite('crops3/cropped_120.jpg',cropped3_120)
    cv2.imwrite('crops3/cropped_121.jpg',cropped3_121)
    cv2.imwrite('crops3/cropped_122.jpg',cropped3_122)
    cv2.imwrite('crops3/cropped_123.jpg',cropped3_123)
    cv2.imwrite('crops3/cropped_124.jpg',cropped3_124)
    cv2.imwrite('crops3/cropped_125.jpg',cropped3_125)
    cv2.imwrite('crops3/cropped_126.jpg',cropped3_126)
    cv2.imwrite('crops3/cropped_127.jpg',cropped3_127)
    cv2.imwrite('crops3/cropped_128.jpg',cropped3_128)
    cv2.imwrite('crops3/cropped_129.jpg',cropped3_129)
    cv2.imwrite('crops3/cropped_130.jpg',cropped3_130)
    cv2.imwrite('crops3/cropped_131.jpg',cropped3_131)
    cv2.imwrite('crops3/cropped_132.jpg',cropped3_132)
    cv2.imwrite('crops3/cropped_133.jpg',cropped3_133)
    cv2.imwrite('crops3/cropped_134.jpg',cropped3_134)
    cv2.imwrite('crops3/cropped_135.jpg',cropped3_135)
    cv2.imwrite('crops3/cropped_136.jpg',cropped3_136)
    cv2.imwrite('crops3/cropped_137.jpg',cropped3_137)
    cv2.imwrite('crops3/cropped_138.jpg',cropped3_138)
    cv2.imwrite('crops3/cropped_139.jpg',cropped3_139)
    cv2.imwrite('crops3/cropped_140.jpg',cropped3_140)
    cv2.imwrite('crops3/cropped_141.jpg',cropped3_141)
    cv2.imwrite('crops3/cropped_142.jpg',cropped3_142)
    cv2.imwrite('crops3/cropped_143.jpg',cropped3_143)
    cv2.imwrite('crops3/cropped_144.jpg',cropped3_144)
    cv2.imwrite('crops3/cropped_145.jpg',cropped3_145)
    cv2.imwrite('crops3/cropped_146.jpg',cropped3_146)
    cv2.imwrite('crops3/cropped_147.jpg',cropped3_147)
    cv2.imwrite('crops3/cropped_148.jpg',cropped3_148)
    cv2.imwrite('crops3/cropped_149.jpg',cropped3_149)
    cv2.imwrite('crops3/cropped_150.jpg',cropped3_150)
    cv2.imwrite('crops3/cropped_151.jpg',cropped3_151)
    cv2.imwrite('crops3/cropped_152.jpg',cropped3_152)
    cv2.imwrite('crops3/cropped_153.jpg',cropped3_153)
    cv2.imwrite('crops3/cropped_154.jpg',cropped3_154)
    cv2.imwrite('crops3/cropped_155.jpg',cropped3_155)
    cv2.imwrite('crops3/cropped_156.jpg',cropped3_156)
    cv2.imwrite('crops3/cropped_157.jpg',cropped3_157)
    cv2.imwrite('crops3/cropped_158.jpg',cropped3_158)
    cv2.imwrite('crops3/cropped_159.jpg',cropped3_159)
    cv2.imwrite('crops3/cropped_160.jpg',cropped3_160)
    cv2.imwrite('crops3/cropped_161.jpg',cropped3_161)
    cv2.imwrite('crops3/cropped_162.jpg',cropped3_162)
    cv2.imwrite('crops3/cropped_163.jpg',cropped3_163)
    cv2.imwrite('crops3/cropped_164.jpg',cropped3_164)
    cv2.imwrite('crops3/cropped_165.jpg',cropped3_165)
    cv2.imwrite('crops3/cropped_166.jpg',cropped3_166)
    cv2.imwrite('crops3/cropped_167.jpg',cropped3_167)
    cv2.imwrite('crops3/cropped_168.jpg',cropped3_168)
    cv2.imwrite('crops3/cropped_169.jpg',cropped3_169)
    cv2.imwrite('crops3/cropped_170.jpg',cropped3_170)
    cv2.imwrite('crops3/cropped_171.jpg',cropped3_171)
    cv2.imwrite('crops3/cropped_172.jpg',cropped3_172)
    cv2.imwrite('crops3/cropped_173.jpg',cropped3_173)
    cv2.imwrite('crops3/cropped_174.jpg',cropped3_174)
    cv2.imwrite('crops3/cropped_175.jpg',cropped3_175)
    cv2.imwrite('crops3/cropped_176.jpg',cropped3_176)
    cv2.imwrite('crops3/cropped_177.jpg',cropped3_177)
    cv2.imwrite('crops3/cropped_178.jpg',cropped3_178)
    cv2.imwrite('crops3/cropped_179.jpg',cropped3_179)
    cv2.imwrite('crops3/cropped_180.jpg',cropped3_180)
    cv2.imwrite('crops3/cropped_181.jpg',cropped3_181)
    cv2.imwrite('crops3/cropped_182.jpg',cropped3_182)
    cv2.imwrite('crops3/cropped_183.jpg',cropped3_183)
    cv2.imwrite('crops3/cropped_184.jpg',cropped3_184)
    cv2.imwrite('crops3/cropped_185.jpg',cropped3_185)
    cv2.imwrite('crops3/cropped_186.jpg',cropped3_186)
    cv2.imwrite('crops3/cropped_187.jpg',cropped3_187)
    cv2.imwrite('crops3/cropped_188.jpg',cropped3_188)
    cv2.imwrite('crops3/cropped_189.jpg',cropped3_189)
    cv2.imwrite('crops3/cropped_190.jpg',cropped3_190)
    cv2.imwrite('crops3/cropped_191.jpg',cropped3_191)
    cv2.imwrite('crops3/cropped_192.jpg',cropped3_192)
    cv2.imwrite('crops3/cropped_193.jpg',cropped3_193)
    cv2.imwrite('crops3/cropped_194.jpg',cropped3_194)
    cv2.imwrite('crops3/cropped_195.jpg',cropped3_195)
    cv2.imwrite('crops3/cropped_196.jpg',cropped3_196)
    cv2.imwrite('crops3/cropped_197.jpg',cropped3_197)
    cv2.imwrite('crops3/cropped_198.jpg',cropped3_198)
    cv2.imwrite('crops3/cropped_199.jpg',cropped3_199)
    cv2.imwrite('crops3/cropped_200.jpg',cropped3_200)
    cv2.imwrite('crops3/cropped_201.jpg',cropped3_201)
    cv2.imwrite('crops3/cropped_202.jpg',cropped3_202)
    cv2.imwrite('crops3/cropped_203.jpg',cropped3_203)
    cv2.imwrite('crops3/cropped_204.jpg',cropped3_204)
    cv2.imwrite('crops3/cropped_205.jpg',cropped3_205)
    cv2.imwrite('crops3/cropped_206.jpg',cropped3_206)
    cv2.imwrite('crops3/cropped_207.jpg',cropped3_207)
    cv2.imwrite('crops3/cropped_208.jpg',cropped3_208)
    cv2.imwrite('crops3/cropped_209.jpg',cropped3_209)
    cv2.imwrite('crops3/cropped_210.jpg',cropped3_210)
    cv2.imwrite('crops3/cropped_211.jpg',cropped3_211)
    cv2.imwrite('crops3/cropped_212.jpg',cropped3_212)
    cv2.imwrite('crops3/cropped_213.jpg',cropped3_213)
    cv2.imwrite('crops3/cropped_214.jpg',cropped3_214)
    cv2.imwrite('crops3/cropped_215.jpg',cropped3_215)
    cv2.imwrite('crops3/cropped_216.jpg',cropped3_216)
    cv2.imwrite('crops3/cropped_217.jpg',cropped3_217)
    cv2.imwrite('crops3/cropped_218.jpg',cropped3_218)
    cv2.imwrite('crops3/cropped_219.jpg',cropped3_219)
    cv2.imwrite('crops3/cropped_220.jpg',cropped3_220)
    cv2.imwrite('crops3/cropped_221.jpg',cropped3_221)
    cv2.imwrite('crops3/cropped_222.jpg',cropped3_222)
    cv2.imwrite('crops3/cropped_223.jpg',cropped3_223)
    cv2.imwrite('crops3/cropped_224.jpg',cropped3_224)
    cv2.imwrite('crops3/cropped_225.jpg',cropped3_225)
    cv2.imwrite('crops3/cropped_226.jpg',cropped3_226)
    cv2.imwrite('crops3/cropped_227.jpg',cropped3_227)
    cv2.imwrite('crops3/cropped_228.jpg',cropped3_228)
    cv2.imwrite('crops3/cropped_229.jpg',cropped3_229)
    cv2.imwrite('crops3/cropped_230.jpg',cropped3_230)
    cv2.imwrite('crops3/cropped_231.jpg',cropped3_231)
    cv2.imwrite('crops3/cropped_232.jpg',cropped3_232)
    cv2.imwrite('crops3/cropped_233.jpg',cropped3_233)
    cv2.imwrite('crops3/cropped_234.jpg',cropped3_234)
    cv2.imwrite('crops3/cropped_235.jpg',cropped3_235)
    cv2.imwrite('crops3/cropped_236.jpg',cropped3_236)
    cv2.imwrite('crops3/cropped_237.jpg',cropped3_237)
    cv2.imwrite('crops3/cropped_238.jpg',cropped3_238)
    cv2.imwrite('crops3/cropped_239.jpg',cropped3_239)
    cv2.imwrite('crops3/cropped_240.jpg',cropped3_240)
    cv2.imwrite('crops3/cropped_241.jpg',cropped3_241)
    cv2.imwrite('crops3/cropped_242.jpg',cropped3_242)
    cv2.imwrite('crops3/cropped_243.jpg',cropped3_243)
    cv2.imwrite('crops3/cropped_244.jpg',cropped3_244)
    cv2.imwrite('crops3/cropped_245.jpg',cropped3_245)
    cv2.imwrite('crops3/cropped_246.jpg',cropped3_246)
    cv2.imwrite('crops3/cropped_247.jpg',cropped3_247)
    cv2.imwrite('crops3/cropped_248.jpg',cropped3_248)
    cv2.imwrite('crops3/cropped_249.jpg',cropped3_249)
    cv2.imwrite('crops3/cropped_250.jpg',cropped3_250)
    cv2.imwrite('crops3/cropped_251.jpg',cropped3_251)
    cv2.imwrite('crops3/cropped_252.jpg',cropped3_252)
    cv2.imwrite('crops3/cropped_253.jpg',cropped3_253)
    cv2.imwrite('crops3/cropped_254.jpg',cropped3_254)
    cv2.imwrite('crops3/cropped_255.jpg',cropped3_255)
    cv2.imwrite('crops3/cropped_256.jpg',cropped3_256)
    cv2.imwrite('crops3/cropped_257.jpg',cropped3_257)
    cv2.imwrite('crops3/cropped_258.jpg',cropped3_258)
    cv2.imwrite('crops3/cropped_259.jpg',cropped3_259)
    cv2.imwrite('crops3/cropped_260.jpg',cropped3_260)
    cv2.imwrite('crops3/cropped_261.jpg',cropped3_261)
    cv2.imwrite('crops3/cropped_262.jpg',cropped3_262)
    cv2.imwrite('crops3/cropped_263.jpg',cropped3_263)
    cv2.imwrite('crops3/cropped_264.jpg',cropped3_264)
    cv2.imwrite('crops3/cropped_265.jpg',cropped3_265)
    cv2.imwrite('crops3/cropped_266.jpg',cropped3_266)
    cv2.imwrite('crops3/cropped_267.jpg',cropped3_267)
    cv2.imwrite('crops3/cropped_268.jpg',cropped3_268)
    cv2.imwrite('crops3/cropped_269.jpg',cropped3_269)
    cv2.imwrite('crops3/cropped_270.jpg',cropped3_270)
    cv2.imwrite('crops3/cropped_271.jpg',cropped3_271)
    cv2.imwrite('crops3/cropped_272.jpg',cropped3_272)
    cv2.imwrite('crops3/cropped_273.jpg',cropped3_273)
    cv2.imwrite('crops3/cropped_274.jpg',cropped3_274)
    cv2.imwrite('crops3/cropped_275.jpg',cropped3_275)
    cv2.imwrite('crops3/cropped_276.jpg',cropped3_276)
    cv2.imwrite('crops3/cropped_277.jpg',cropped3_277)
    cv2.imwrite('crops3/cropped_278.jpg',cropped3_278)
    cv2.imwrite('crops3/cropped_279.jpg',cropped3_279)
    cv2.imwrite('crops3/cropped_280.jpg',cropped3_280)

    #end = time.time()
    #print(f"time elapsed: {end-start}")
#for i in range(18):
	#start_row = horizontal
	#cropped = image[30:250,100:230]
	#image[start_row:end_row, start_column:end_column]
#cv2.waitKey()
#cv2.destroyAllWindows()

#storing the final image into a folder 
#cv2.imwrite('final2.jpg',final2)

# cropping
# for loop with numpy array slicing to crop out individual braille characters
# save them using cv2.imwrite within the for loop 
