import numpy as np
import cv2 
from imutils.object_detection import non_max_suppression 
import time

#checking
def preProcess(img, outdir, setting):
    # original image - captured from camera upon button activation
    # braille.jpg should be replaced with the file path from camera capture
    img = cv2.imread(img)
    img1 = img 
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    # Compute the background (use a large filter radius for excluding the dots)
    bg = cv2.medianBlur(gray, 151)
    # Compute absolute difference 
    fg = cv2.absdiff(gray, bg)  
    # Blur the image using Gaussian Blur filter
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 7, 2)
    # Apply Canny edge detection.
    edges = cv2.Canny(gray, threshold1=50, threshold2=100) 
    # Merge edges with thresh
    res = cv2.bitwise_or(thresh, edges)  
    # kernel matrix that includes size of the mask (2,2), for erosion and dilation
    kernel = np.ones((2,2), np.uint8)
    dkernel = np.ones((3,3),np.uint8)
    # Erosion to get rid of small dots noises 
    erosion = cv2.erode(thresh, kernel, iterations=1)
    erosion2 = cv2.erode(thresh, kernel, iterations=2)
    erosion_canny = cv2.erode(res, kernel, iterations=1)
    erosion2_canny = cv2.erode(res, kernel, iterations=2)
    # Dilation to expand the eroded dots to reasonable sizes 
    dilation = cv2.dilate(erosion, dkernel, iterations = 1)
    dilation2 = cv2.dilate(erosion2, dkernel, iterations = 2)
    dilation_canny = cv2.erode(erosion_canny, dkernel, iterations=1)
    dilation2_canny = cv2.erode(erosion2_canny, dkernel, iterations=2)

    k_dilation = cv2.dilate(erosion, kernel, iterations = 1)
    k_dilation2 = cv2.dilate(erosion2, kernel, iterations = 2)
    k_dilation_canny = cv2.erode(erosion_canny, kernel, iterations=1)
    k_dilation2_canny = cv2.erode(erosion2_canny, kernel, iterations=2)

    # Invert the image for white background & block dots 
    final1 = cv2.bitwise_not(dilation)
    final2 = cv2.bitwise_not(k_dilation)
    cv2.imshow('final1', final1)
    cv2.imshow('final2', final2)
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
    stats = matrix_stat[1:]
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

    filtered_stats = non_max_suppression(np.array(filtered_stats), overlapThresh= 0.2)
    temp2 = final2
    for x1, y1, x2, y2 in filtered_stats:
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        #(0,255,0)=green
        cv2.circle(img1, ((x1+x2)//2, (y1+y2)//2), 4, (0, 255, 0), -1)
    for x1, y1, x2, y2 in filtered_stats:
        cv2.circle(temp_final, ((x1+x2)//2, (y1+y2)//2), 4, (0, 255, 0), -1)

    # ***** Change the file location ***** 
    cv2.imwrite('final.jpg',img1)

    #[height, width]
    #row1
    '''
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
    '''
    
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
    
    '''
    #writing for crops folder 1
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
    '''

    #writing for crops 2 
    cv2.imwrite(f'{outdir}/cropped_1.jpg',cropped2_1)
    cv2.imwrite(f'{outdir}/cropped_2.jpg',cropped2_2)
    cv2.imwrite(f'{outdir}/cropped_3.jpg',cropped2_3)
    cv2.imwrite(f'{outdir}/cropped_4.jpg',cropped2_4)
    cv2.imwrite(f'{outdir}/cropped_5.jpg',cropped2_5)
    cv2.imwrite(f'{outdir}/cropped_6.jpg',cropped2_6)
    cv2.imwrite(f'{outdir}/cropped_7.jpg',cropped2_7)
    cv2.imwrite(f'{outdir}/cropped_8.jpg',cropped2_8)
    cv2.imwrite(f'{outdir}/cropped_9.jpg',cropped2_9)
    cv2.imwrite(f'{outdir}/cropped_10.jpg',cropped2_10)
    cv2.imwrite(f'{outdir}/cropped_11.jpg',cropped2_11)
    cv2.imwrite(f'{outdir}/cropped_12.jpg',cropped2_12)
    cv2.imwrite(f'{outdir}/cropped_13.jpg',cropped2_13)
    cv2.imwrite(f'{outdir}/cropped_14.jpg',cropped2_14)
    cv2.imwrite(f'{outdir}/cropped_15.jpg',cropped2_15)
    cv2.imwrite(f'{outdir}/cropped_16.jpg',cropped2_16)
    cv2.imwrite(f'{outdir}/cropped_17.jpg',cropped2_17)
    cv2.imwrite(f'{outdir}/cropped_18.jpg',cropped2_18)
    cv2.imwrite(f'{outdir}/cropped_19.jpg',cropped2_19)
    cv2.imwrite(f'{outdir}/cropped_20.jpg',cropped2_20)
    cv2.imwrite(f'{outdir}/cropped_21.jpg',cropped2_21)
    cv2.imwrite(f'{outdir}/cropped_22.jpg',cropped2_22)
    cv2.imwrite(f'{outdir}/cropped_23.jpg',cropped2_23)
    cv2.imwrite(f'{outdir}/cropped_24.jpg',cropped2_24)
    cv2.imwrite(f'{outdir}/cropped_25.jpg',cropped2_25)
    cv2.imwrite(f'{outdir}/cropped_26.jpg',cropped2_26)
    cv2.imwrite(f'{outdir}/cropped_27.jpg',cropped2_27)
    cv2.imwrite(f'{outdir}/cropped_28.jpg',cropped2_28)
    cv2.imwrite(f'{outdir}/cropped_29.jpg',cropped2_29)
    cv2.imwrite(f'{outdir}/cropped_30.jpg',cropped2_30)
    cv2.imwrite(f'{outdir}/cropped_31.jpg',cropped2_31)
    cv2.imwrite(f'{outdir}/cropped_32.jpg',cropped2_32)
    cv2.imwrite(f'{outdir}/cropped_33.jpg',cropped2_33)
    cv2.imwrite(f'{outdir}/cropped_34.jpg',cropped2_34)
    cv2.imwrite(f'{outdir}/cropped_35.jpg',cropped2_35)
    cv2.imwrite(f'{outdir}/cropped_36.jpg',cropped2_36)
    cv2.imwrite(f'{outdir}/cropped_37.jpg',cropped2_37)
    cv2.imwrite(f'{outdir}/cropped_38.jpg',cropped2_38)
    cv2.imwrite(f'{outdir}/cropped_39.jpg',cropped2_39)
    cv2.imwrite(f'{outdir}/cropped_40.jpg',cropped2_40)
    cv2.imwrite(f'{outdir}/cropped_41.jpg',cropped2_41)
    cv2.imwrite(f'{outdir}/cropped_42.jpg',cropped2_42)
    cv2.imwrite(f'{outdir}/cropped_43.jpg',cropped2_43)
    cv2.imwrite(f'{outdir}/cropped_44.jpg',cropped2_44)
    cv2.imwrite(f'{outdir}/cropped_45.jpg',cropped2_45)
    cv2.imwrite(f'{outdir}/cropped_46.jpg',cropped2_46)
    cv2.imwrite(f'{outdir}/cropped_47.jpg',cropped2_47)
    cv2.imwrite(f'{outdir}/cropped_48.jpg',cropped2_48)
    cv2.imwrite(f'{outdir}/cropped_49.jpg',cropped2_49)
    cv2.imwrite(f'{outdir}/cropped_50.jpg',cropped2_50)
    cv2.imwrite(f'{outdir}/cropped_51.jpg',cropped2_51)
    cv2.imwrite(f'{outdir}/cropped_52.jpg',cropped2_52)
    cv2.imwrite(f'{outdir}/cropped_53.jpg',cropped2_53)
    cv2.imwrite(f'{outdir}/cropped_54.jpg',cropped2_54)
    cv2.imwrite(f'{outdir}/cropped_55.jpg',cropped2_55)
    cv2.imwrite(f'{outdir}/cropped_56.jpg',cropped2_56)
    cv2.imwrite(f'{outdir}/cropped_57.jpg',cropped2_57)
    cv2.imwrite(f'{outdir}/cropped_58.jpg',cropped2_58)
    cv2.imwrite(f'{outdir}/cropped_59.jpg',cropped2_59)
    cv2.imwrite(f'{outdir}/cropped_60.jpg',cropped2_60)
    cv2.imwrite(f'{outdir}/cropped_61.jpg',cropped2_61)
    cv2.imwrite(f'{outdir}/cropped_62.jpg',cropped2_62)
    cv2.imwrite(f'{outdir}/cropped_63.jpg',cropped2_63)
    cv2.imwrite(f'{outdir}/cropped_64.jpg',cropped2_64)
    cv2.imwrite(f'{outdir}/cropped_65.jpg',cropped2_65)
    cv2.imwrite(f'{outdir}/cropped_66.jpg',cropped2_66)
    cv2.imwrite(f'{outdir}/cropped_67.jpg',cropped2_67)
    cv2.imwrite(f'{outdir}/cropped_68.jpg',cropped2_68)
    cv2.imwrite(f'{outdir}/cropped_69.jpg',cropped2_69)
    cv2.imwrite(f'{outdir}/cropped_70.jpg',cropped2_70)
    cv2.imwrite(f'{outdir}/cropped_71.jpg',cropped2_71)
    cv2.imwrite(f'{outdir}/cropped_72.jpg',cropped2_72)
    cv2.imwrite(f'{outdir}/cropped_73.jpg',cropped2_73)
    cv2.imwrite(f'{outdir}/cropped_74.jpg',cropped2_74)
    cv2.imwrite(f'{outdir}/cropped_75.jpg',cropped2_75)
    cv2.imwrite(f'{outdir}/cropped_76.jpg',cropped2_76)
    cv2.imwrite(f'{outdir}/cropped_77.jpg',cropped2_77)
    cv2.imwrite(f'{outdir}/cropped_78.jpg',cropped2_78)
    cv2.imwrite(f'{outdir}/cropped_79.jpg',cropped2_79)
    cv2.imwrite(f'{outdir}/cropped_80.jpg',cropped2_80)
    cv2.imwrite(f'{outdir}/cropped_81.jpg',cropped2_81)
    cv2.imwrite(f'{outdir}/cropped_82.jpg',cropped2_82)
    cv2.imwrite(f'{outdir}/cropped_83.jpg',cropped2_83)
    cv2.imwrite(f'{outdir}/cropped_84.jpg',cropped2_84)
    cv2.imwrite(f'{outdir}/cropped_85.jpg',cropped2_85)
    cv2.imwrite(f'{outdir}/cropped_86.jpg',cropped2_86)
    cv2.imwrite(f'{outdir}/cropped_87.jpg',cropped2_87)
    cv2.imwrite(f'{outdir}/cropped_88.jpg',cropped2_88)
    cv2.imwrite(f'{outdir}/cropped_89.jpg',cropped2_89)
    cv2.imwrite(f'{outdir}/cropped_90.jpg',cropped2_90)
    cv2.imwrite(f'{outdir}/cropped_91.jpg',cropped2_91)
    cv2.imwrite(f'{outdir}/cropped_92.jpg',cropped2_92)
    cv2.imwrite(f'{outdir}/cropped_93.jpg',cropped2_93)
    cv2.imwrite(f'{outdir}/cropped_94.jpg',cropped2_94)
    cv2.imwrite(f'{outdir}/cropped_95.jpg',cropped2_95)
    cv2.imwrite(f'{outdir}/cropped_96.jpg',cropped2_96)
    cv2.imwrite(f'{outdir}/cropped_97.jpg',cropped2_97)
    cv2.imwrite(f'{outdir}/cropped_98.jpg',cropped2_98)
    cv2.imwrite(f'{outdir}/cropped_99.jpg',cropped2_99)
    cv2.imwrite(f'{outdir}/cropped_100.jpg',cropped2_100)
    cv2.imwrite(f'{outdir}/cropped_101.jpg',cropped2_101)
    cv2.imwrite(f'{outdir}/cropped_102.jpg',cropped2_102)
    cv2.imwrite(f'{outdir}/cropped_103.jpg',cropped2_103)
    cv2.imwrite(f'{outdir}/cropped_104.jpg',cropped2_104)
    cv2.imwrite(f'{outdir}/cropped_105.jpg',cropped2_105)
    cv2.imwrite(f'{outdir}/cropped_106.jpg',cropped2_106)
    cv2.imwrite(f'{outdir}/cropped_107.jpg',cropped2_107)
    cv2.imwrite(f'{outdir}/cropped_108.jpg',cropped2_108)
    cv2.imwrite(f'{outdir}/cropped_109.jpg',cropped2_109)
    cv2.imwrite(f'{outdir}/cropped_110.jpg',cropped2_110)
    cv2.imwrite(f'{outdir}/cropped_111.jpg',cropped2_111)
    cv2.imwrite(f'{outdir}/cropped_112.jpg',cropped2_112)
    cv2.imwrite(f'{outdir}/cropped_113.jpg',cropped2_113)
    cv2.imwrite(f'{outdir}/cropped_114.jpg',cropped2_114)
    cv2.imwrite(f'{outdir}/cropped_115.jpg',cropped2_115)
    cv2.imwrite(f'{outdir}/cropped_116.jpg',cropped2_116)
    cv2.imwrite(f'{outdir}/cropped_117.jpg',cropped2_117)
    cv2.imwrite(f'{outdir}/cropped_118.jpg',cropped2_118)
    cv2.imwrite(f'{outdir}/cropped_119.jpg',cropped2_119)
    cv2.imwrite(f'{outdir}/cropped_120.jpg',cropped2_120)
    cv2.imwrite(f'{outdir}/cropped_121.jpg',cropped2_121)
    cv2.imwrite(f'{outdir}/cropped_122.jpg',cropped2_122)
    cv2.imwrite(f'{outdir}/cropped_123.jpg',cropped2_123)
    cv2.imwrite(f'{outdir}/cropped_124.jpg',cropped2_124)
    cv2.imwrite(f'{outdir}/cropped_125.jpg',cropped2_125)
    cv2.imwrite(f'{outdir}/cropped_126.jpg',cropped2_126)
    cv2.imwrite(f'{outdir}/cropped_127.jpg',cropped2_127)
    cv2.imwrite(f'{outdir}/cropped_128.jpg',cropped2_128)
    cv2.imwrite(f'{outdir}/cropped_129.jpg',cropped2_129)
    cv2.imwrite(f'{outdir}/cropped_130.jpg',cropped2_130)
    cv2.imwrite(f'{outdir}/cropped_131.jpg',cropped2_131)
    cv2.imwrite(f'{outdir}/cropped_132.jpg',cropped2_132)
    cv2.imwrite(f'{outdir}/cropped_133.jpg',cropped2_133)
    cv2.imwrite(f'{outdir}/cropped_134.jpg',cropped2_134)
    cv2.imwrite(f'{outdir}/cropped_135.jpg',cropped2_135)
    cv2.imwrite(f'{outdir}/cropped_136.jpg',cropped2_136)
    cv2.imwrite(f'{outdir}/cropped_137.jpg',cropped2_137)
    cv2.imwrite(f'{outdir}/cropped_138.jpg',cropped2_138)
    cv2.imwrite(f'{outdir}/cropped_139.jpg',cropped2_139)
    cv2.imwrite(f'{outdir}/cropped_140.jpg',cropped2_140)
    cv2.imwrite(f'{outdir}/cropped_141.jpg',cropped2_141)
    cv2.imwrite(f'{outdir}/cropped_142.jpg',cropped2_142)
    cv2.imwrite(f'{outdir}/cropped_143.jpg',cropped2_143)
    cv2.imwrite(f'{outdir}/cropped_144.jpg',cropped2_144)
    cv2.imwrite(f'{outdir}/cropped_145.jpg',cropped2_145)
    cv2.imwrite(f'{outdir}/cropped_146.jpg',cropped2_146)
    cv2.imwrite(f'{outdir}/cropped_147.jpg',cropped2_147)
    cv2.imwrite(f'{outdir}/cropped_148.jpg',cropped2_148)
    cv2.imwrite(f'{outdir}/cropped_149.jpg',cropped2_149)
    cv2.imwrite(f'{outdir}/cropped_150.jpg',cropped2_150)
    cv2.imwrite(f'{outdir}/cropped_151.jpg',cropped2_151)
    cv2.imwrite(f'{outdir}/cropped_152.jpg',cropped2_152)
    cv2.imwrite(f'{outdir}/cropped_153.jpg',cropped2_153)
    cv2.imwrite(f'{outdir}/cropped_154.jpg',cropped2_154)
    cv2.imwrite(f'{outdir}/cropped_155.jpg',cropped2_155)
    cv2.imwrite(f'{outdir}/cropped_156.jpg',cropped2_156)
    cv2.imwrite(f'{outdir}/cropped_157.jpg',cropped2_157)
    cv2.imwrite(f'{outdir}/cropped_158.jpg',cropped2_158)
    cv2.imwrite(f'{outdir}/cropped_159.jpg',cropped2_159)
    cv2.imwrite(f'{outdir}/cropped_160.jpg',cropped2_160)
    cv2.imwrite(f'{outdir}/cropped_161.jpg',cropped2_161)
    cv2.imwrite(f'{outdir}/cropped_162.jpg',cropped2_162)
    cv2.imwrite(f'{outdir}/cropped_163.jpg',cropped2_163)
    cv2.imwrite(f'{outdir}/cropped_164.jpg',cropped2_164)
    cv2.imwrite(f'{outdir}/cropped_165.jpg',cropped2_165)
    cv2.imwrite(f'{outdir}/cropped_166.jpg',cropped2_166)
    cv2.imwrite(f'{outdir}/cropped_167.jpg',cropped2_167)
    cv2.imwrite(f'{outdir}/cropped_168.jpg',cropped2_168)
    cv2.imwrite(f'{outdir}/cropped_169.jpg',cropped2_169)
    cv2.imwrite(f'{outdir}/cropped_170.jpg',cropped2_170)
    cv2.imwrite(f'{outdir}/cropped_171.jpg',cropped2_171)
    cv2.imwrite(f'{outdir}/cropped_172.jpg',cropped2_172)
    cv2.imwrite(f'{outdir}/cropped_173.jpg',cropped2_173)
    cv2.imwrite(f'{outdir}/cropped_174.jpg',cropped2_174)
    cv2.imwrite(f'{outdir}/cropped_175.jpg',cropped2_175)
    cv2.imwrite(f'{outdir}/cropped_176.jpg',cropped2_176)
    cv2.imwrite(f'{outdir}/cropped_177.jpg',cropped2_177)
    cv2.imwrite(f'{outdir}/cropped_178.jpg',cropped2_178)
    cv2.imwrite(f'{outdir}/cropped_179.jpg',cropped2_179)
    cv2.imwrite(f'{outdir}/cropped_180.jpg',cropped2_180)
    cv2.imwrite(f'{outdir}/cropped_181.jpg',cropped2_181)
    cv2.imwrite(f'{outdir}/cropped_182.jpg',cropped2_182)
    cv2.imwrite(f'{outdir}/cropped_183.jpg',cropped2_183)
    cv2.imwrite(f'{outdir}/cropped_184.jpg',cropped2_184)
    cv2.imwrite(f'{outdir}/cropped_185.jpg',cropped2_185)
    cv2.imwrite(f'{outdir}/cropped_186.jpg',cropped2_186)
    cv2.imwrite(f'{outdir}/cropped_187.jpg',cropped2_187)
    cv2.imwrite(f'{outdir}/cropped_188.jpg',cropped2_188)
    cv2.imwrite(f'{outdir}/cropped_189.jpg',cropped2_189)
    cv2.imwrite(f'{outdir}/cropped_190.jpg',cropped2_190)
    cv2.imwrite(f'{outdir}/cropped_191.jpg',cropped2_191)
    cv2.imwrite(f'{outdir}/cropped_192.jpg',cropped2_192)
    cv2.imwrite(f'{outdir}/cropped_193.jpg',cropped2_193)
    cv2.imwrite(f'{outdir}/cropped_194.jpg',cropped2_194)
    cv2.imwrite(f'{outdir}/cropped_195.jpg',cropped2_195)
    cv2.imwrite(f'{outdir}/cropped_196.jpg',cropped2_196)
    cv2.imwrite(f'{outdir}/cropped_197.jpg',cropped2_197)
    cv2.imwrite(f'{outdir}/cropped_198.jpg',cropped2_198)
    cv2.imwrite(f'{outdir}/cropped_199.jpg',cropped2_199)
    cv2.imwrite(f'{outdir}/cropped_200.jpg',cropped2_200)
    cv2.imwrite(f'{outdir}/cropped_201.jpg',cropped2_201)
    cv2.imwrite(f'{outdir}/cropped_202.jpg',cropped2_202)
    cv2.imwrite(f'{outdir}/cropped_203.jpg',cropped2_203)
    cv2.imwrite(f'{outdir}/cropped_204.jpg',cropped2_204)
    cv2.imwrite(f'{outdir}/cropped_205.jpg',cropped2_205)
    cv2.imwrite(f'{outdir}/cropped_206.jpg',cropped2_206)
    cv2.imwrite(f'{outdir}/cropped_207.jpg',cropped2_207)
    cv2.imwrite(f'{outdir}/cropped_208.jpg',cropped2_208)
    cv2.imwrite(f'{outdir}/cropped_209.jpg',cropped2_209)
    cv2.imwrite(f'{outdir}/cropped_210.jpg',cropped2_210)
    cv2.imwrite(f'{outdir}/cropped_211.jpg',cropped2_211)
    cv2.imwrite(f'{outdir}/cropped_212.jpg',cropped2_212)
    cv2.imwrite(f'{outdir}/cropped_213.jpg',cropped2_213)
    cv2.imwrite(f'{outdir}/cropped_214.jpg',cropped2_214)
    cv2.imwrite(f'{outdir}/cropped_215.jpg',cropped2_215)
    cv2.imwrite(f'{outdir}/cropped_216.jpg',cropped2_216)
    cv2.imwrite(f'{outdir}/cropped_217.jpg',cropped2_217)
    cv2.imwrite(f'{outdir}/cropped_218.jpg',cropped2_218)
    cv2.imwrite(f'{outdir}/cropped_219.jpg',cropped2_219)
    cv2.imwrite(f'{outdir}/cropped_220.jpg',cropped2_220)
    cv2.imwrite(f'{outdir}/cropped_221.jpg',cropped2_221)
    cv2.imwrite(f'{outdir}/cropped_222.jpg',cropped2_222)
    cv2.imwrite(f'{outdir}/cropped_223.jpg',cropped2_223)
    cv2.imwrite(f'{outdir}/cropped_224.jpg',cropped2_224)
    cv2.imwrite(f'{outdir}/cropped_225.jpg',cropped2_225)
    cv2.imwrite(f'{outdir}/cropped_226.jpg',cropped2_226)
    cv2.imwrite(f'{outdir}/cropped_227.jpg',cropped2_227)
    cv2.imwrite(f'{outdir}/cropped_228.jpg',cropped2_228)
    cv2.imwrite(f'{outdir}/cropped_229.jpg',cropped2_229)
    cv2.imwrite(f'{outdir}/cropped_230.jpg',cropped2_230)
    cv2.imwrite(f'{outdir}/cropped_231.jpg',cropped2_231)
    cv2.imwrite(f'{outdir}/cropped_232.jpg',cropped2_232)
    cv2.imwrite(f'{outdir}/cropped_233.jpg',cropped2_233)
    cv2.imwrite(f'{outdir}/cropped_234.jpg',cropped2_234)
    cv2.imwrite(f'{outdir}/cropped_235.jpg',cropped2_235)
    cv2.imwrite(f'{outdir}/cropped_236.jpg',cropped2_236)
    cv2.imwrite(f'{outdir}/cropped_237.jpg',cropped2_237)
    cv2.imwrite(f'{outdir}/cropped_238.jpg',cropped2_238)
    cv2.imwrite(f'{outdir}/cropped_239.jpg',cropped2_239)
    cv2.imwrite(f'{outdir}/cropped_240.jpg',cropped2_240)
    cv2.imwrite(f'{outdir}/cropped_241.jpg',cropped2_241)
    cv2.imwrite(f'{outdir}/cropped_242.jpg',cropped2_242)
    cv2.imwrite(f'{outdir}/cropped_243.jpg',cropped2_243)
    cv2.imwrite(f'{outdir}/cropped_244.jpg',cropped2_244)
    cv2.imwrite(f'{outdir}/cropped_245.jpg',cropped2_245)
    cv2.imwrite(f'{outdir}/cropped_246.jpg',cropped2_246)
    cv2.imwrite(f'{outdir}/cropped_247.jpg',cropped2_247)
    cv2.imwrite(f'{outdir}/cropped_248.jpg',cropped2_248)
    cv2.imwrite(f'{outdir}/cropped_249.jpg',cropped2_249)
    cv2.imwrite(f'{outdir}/cropped_250.jpg',cropped2_250)
    cv2.imwrite(f'{outdir}/cropped_251.jpg',cropped2_251)
    cv2.imwrite(f'{outdir}/cropped_252.jpg',cropped2_252)
    cv2.imwrite(f'{outdir}/cropped_253.jpg',cropped2_253)
    cv2.imwrite(f'{outdir}/cropped_254.jpg',cropped2_254)
    cv2.imwrite(f'{outdir}/cropped_255.jpg',cropped2_255)
    cv2.imwrite(f'{outdir}/cropped_256.jpg',cropped2_256)
    cv2.imwrite(f'{outdir}/cropped_257.jpg',cropped2_257)
    cv2.imwrite(f'{outdir}/cropped_258.jpg',cropped2_258)
    cv2.imwrite(f'{outdir}/cropped_259.jpg',cropped2_259)
    cv2.imwrite(f'{outdir}/cropped_260.jpg',cropped2_260)
    cv2.imwrite(f'{outdir}/cropped_261.jpg',cropped2_261)
    cv2.imwrite(f'{outdir}/cropped_262.jpg',cropped2_262)
    cv2.imwrite(f'{outdir}/cropped_263.jpg',cropped2_263)
    cv2.imwrite(f'{outdir}/cropped_264.jpg',cropped2_264)
    cv2.imwrite(f'{outdir}/cropped_265.jpg',cropped2_265)
    cv2.imwrite(f'{outdir}/cropped_266.jpg',cropped2_266)
    cv2.imwrite(f'{outdir}/cropped_267.jpg',cropped2_267)
    cv2.imwrite(f'{outdir}/cropped_268.jpg',cropped2_268)
    cv2.imwrite(f'{outdir}/cropped_269.jpg',cropped2_269)
    cv2.imwrite(f'{outdir}/cropped_270.jpg',cropped2_270)
    cv2.imwrite(f'{outdir}/cropped_271.jpg',cropped2_271)
    cv2.imwrite(f'{outdir}/cropped_272.jpg',cropped2_272)
    cv2.imwrite(f'{outdir}/cropped_273.jpg',cropped2_273)
    cv2.imwrite(f'{outdir}/cropped_274.jpg',cropped2_274)
    cv2.imwrite(f'{outdir}/cropped_275.jpg',cropped2_275)
    cv2.imwrite(f'{outdir}/cropped_276.jpg',cropped2_276)
    cv2.imwrite(f'{outdir}/cropped_277.jpg',cropped2_277)
    cv2.imwrite(f'{outdir}/cropped_278.jpg',cropped2_278)
    cv2.imwrite(f'{outdir}/cropped_279.jpg',cropped2_279)
    cv2.imwrite(f'{outdir}/cropped_280.jpg',cropped2_280)

if __name__ == '__main__':
    img = input("Image path: ")
    outdir = input("Output path: ")
    preProcess(img, outdir, False)
