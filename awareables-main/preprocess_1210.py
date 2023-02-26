import numpy as np
import cv2 
from imutils.object_detection import non_max_suppression 
import time

def preProcess(img, outdir, setting):
    # original image - captured from camera upon button activation
    # braille.jpg should be replaced with the file path from camera capture
    img = cv2.imread(img)
    img1 = img 

    #
    source = img1
    scaleX = 0.6
    scaleY = 0.6

    # Scaling up the image 1.8 times
    scaleUp = cv2.resize(source, None, fx= scaleX*3, fy= scaleY*3, interpolation= cv2.INTER_LINEAR)
    img1 = cv2.resize(source, None, fx= scaleX*3, fy= scaleY*3, interpolation= cv2.INTER_LINEAR)
    img = scaleUp

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
    #kernel = np.ones((2,2), np.uint8)
    #dkernel = np.ones((3,3),np.uint8)
    # Erosion to get rid of small dots noises 
    #erosion = cv2.erode(thresh, kernel, iterations=1)
    #erosion2 = cv2.erode(thresh, kernel, iterations=2)
    #erosion_canny = cv2.erode(res, kernel, iterations=1)
    #erosion2_canny = cv2.erode(res, kernel, iterations=2)
    # Dilation to expand the eroded dots to reasonable sizes 
    #dilation = cv2.dilate(erosion, dkernel, iterations = 1)
    #dilation2 = cv2.dilate(erosion2, dkernel, iterations = 2)
    #dilation_canny = cv2.erode(erosion_canny, dkernel, iterations=1)
    #dilation2_canny = cv2.erode(erosion2_canny, dkernel, iterations=2)

    #k_dilation = cv2.dilate(erosion, kernel, iterations = 1)
    #k_dilation2 = cv2.dilate(erosion2, kernel, iterations = 2)
    #k_dilation_canny = cv2.erode(erosion_canny, kernel, iterations=1)
    #k_dilation2_canny = cv2.erode(erosion2_canny, kernel, iterations=2)

    # Invert the image for white background & block dots 
    #final1 = cv2.bitwise_not(dilation)
    #final2 = cv2.bitwise_not(k_dilation)
    #cv2.imshow('final1', final1)
    #cv2.imshow('final2', final2)
    #temp_final = final1
    #img2 = final1

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
    #temp2 = final2
    for x1, y1, x2, y2 in filtered_stats:
        # cv2.circle(image, center_coordinates, radius, color, thickness)
        #(0,255,0)=green
        cv2.circle(img1, ((x1+x2)//2, (y1+y2)//2), 3, (0, 255, 0), -1)

    #for x1, y1, x2, y2 in filtered_stats:
    #    cv2.circle(temp_final, ((x1+x2)//2, (y1+y2)//2), 4, (0, 255, 0), -1)

    # ***** Change the file location ***** 
    cv2.imwrite('final.jpg',img1)

    #[height, width]
    #row1
    cropped2_1 = img1[64:100, 121:142]
    cropped2_2 = img1[64:100, 142:163]
    cropped2_3 = img1[64:100, 163:184]
    cropped2_4 = img1[64:100, 184:205]
    cropped2_5 = img1[64:100, 207:228]
    cropped2_6 = img1[64:100, 228:249]
    cropped2_7 = img1[64:100, 249:270]
    cropped2_8 = img1[64:100, 271:292]
    cropped2_9 = img1[64:100, 292:313]
    cropped2_10 = img1[64:100, 313:335]
    cropped2_11 = img1[64:100, 335:356]
    cropped2_12 = img1[64:100, 357:378]
    cropped2_13 = img1[64:100, 378:399]
    cropped2_14 = img1[64:100, 399:420]
    cropped2_15 = img1[64:100, 423:444]
    cropped2_16 = img1[64:100, 445:466]
    cropped2_17 = img1[64:100, 466:487]
    cropped2_18 = img1[64:100, 487:508]
    cropped2_19 = img1[64:100, 508:529]
    cropped2_20 = img1[64:100, 529:550]
    cropped2_21 = img1[64:100, 550:571]
    cropped2_22 = img1[64:100, 571:592]
    cropped2_23 = img1[64:100, 592:613]
    cropped2_24 = img1[64:100, 614:635]
    cropped2_25 = img1[64:100, 635:656]
    cropped2_26 = img1[64:100, 656:678]
    cropped2_27 = img1[64:100, 678:699]
    cropped2_28 = img1[64:100, 699:720]
    cropped2_29 = img1[64:100, 722:743]
    cropped2_30 = img1[64:100, 743:764]
    cropped2_31 = img1[64:100, 764:786]
    cropped2_32 = img1[64:100, 786:807]
    cropped2_33 = img1[64:100, 808:829]
    cropped2_34 = img1[64:100, 829:850]
    cropped2_35 = img1[64:100, 850:871]
    cropped2_36 = img1[64:100, 871:892]
    cropped2_37 = img1[64:100, 894:915]
    cropped2_38 = img1[64:100, 913:934]
    cropped2_39 = img1[64:100, 934:955]
    cropped2_40 = img1[64:100, 955:976]
    #row2
    cropped2_41 = img1[101:137, 121:142]
    cropped2_42 = img1[101:137, 142:163]
    cropped2_43 = img1[101:137, 163:184]
    cropped2_44 = img1[101:137, 184:205]
    cropped2_45 = img1[101:137, 207:228]
    cropped2_46 = img1[101:137, 228:249]
    cropped2_47 = img1[101:137, 249:270]
    cropped2_48 = img1[101:137, 271:292]
    cropped2_49 = img1[101:137, 292:313]
    cropped2_50 = img1[101:137, 313:335]
    cropped2_51 = img1[101:137, 335:356]
    cropped2_52 = img1[101:137, 357:378]
    cropped2_53 = img1[101:137, 378:399]
    cropped2_54 = img1[101:137, 399:420]
    cropped2_55 = img1[101:137, 423:444]
    cropped2_56 = img1[101:137, 445:466]
    cropped2_57 = img1[101:137, 466:487]
    cropped2_58 = img1[101:137, 487:508]
    cropped2_59 = img1[101:137, 508:529]
    cropped2_60 = img1[101:137, 529:550]
    cropped2_61 = img1[101:137, 550:571]
    cropped2_62 = img1[101:137, 571:592]
    cropped2_63 = img1[101:137, 592:613]
    cropped2_64 = img1[101:137, 614:635]
    cropped2_65 = img1[101:137, 635:656]
    cropped2_66 = img1[101:137, 656:678]
    cropped2_67 = img1[101:137, 678:699]
    cropped2_68 = img1[101:137, 699:720]
    cropped2_69 = img1[101:137, 722:743]
    cropped2_70 = img1[101:137, 743:764]
    cropped2_71 = img1[101:137, 764:786]
    cropped2_72 = img1[101:137, 786:807]
    cropped2_73 = img1[101:137, 808:829]
    cropped2_74 = img1[101:137, 829:850]
    cropped2_75 = img1[101:137, 850:871]
    cropped2_76 = img1[101:137, 873:894]
    cropped2_77 = img1[101:137, 895:916]
    cropped2_78 = img1[101:137, 916:937]
    cropped2_79 = img1[101:137, 937:958]
    cropped2_80 = img1[101:137, 958:979]
    #row3
    cropped2_81 = img1[138:174, 121:142]
    cropped2_82 = img1[138:174, 142:163]
    cropped2_83 = img1[138:174, 163:184]
    cropped2_84 = img1[138:174, 184:205]
    cropped2_85 = img1[138:174, 207:228]
    cropped2_86 = img1[138:174, 228:249]
    cropped2_87 = img1[138:174, 249:270]
    cropped2_88 = img1[138:174, 271:292]
    cropped2_89 = img1[138:174, 292:313]
    cropped2_90 = img1[138:174, 313:335]
    cropped2_91 = img1[138:174, 335:356]
    cropped2_92 = img1[138:174, 357:378]
    cropped2_93 = img1[138:174, 378:399]
    cropped2_94 = img1[138:174, 399:420]
    cropped2_95 = img1[138:174, 423:444]
    cropped2_96 = img1[138:174, 445:466]
    cropped2_97 = img1[138:174, 466:487]
    cropped2_98 = img1[138:174, 487:508]
    cropped2_99 = img1[138:174, 508:529]
    cropped2_100 = img1[138:174, 529:550]
    cropped2_101 = img1[138:174, 550:571]
    cropped2_102 = img1[138:174, 571:592]
    cropped2_103 = img1[138:174, 592:613]
    cropped2_104 = img1[138:174, 614:635]
    cropped2_105 = img1[138:174, 635:656]
    cropped2_106 = img1[138:174, 656:678]
    cropped2_107 = img1[138:174, 678:699]
    cropped2_108 = img1[138:174, 699:720]
    cropped2_109 = img1[138:174, 722:743]
    cropped2_110 = img1[138:174, 743:764]
    cropped2_111 = img1[138:174, 764:786]
    cropped2_112 = img1[138:174, 786:807]
    cropped2_113 = img1[138:174, 808:829]
    cropped2_114 = img1[138:174, 829:850]
    cropped2_115 = img1[138:174, 850:871]
    cropped2_116 = img1[138:174, 873:894]
    cropped2_117 = img1[138:174, 895:916]
    cropped2_118 = img1[138:174, 916:937]
    cropped2_119 = img1[138:174, 937:958]
    cropped2_120 = img1[138:174, 958:979]
    #row4
    cropped2_121 = img1[175:211, 121:142]
    cropped2_122 = img1[175:211, 142:163]
    cropped2_123 = img1[175:211, 163:184]
    cropped2_124 = img1[175:211, 184:205]
    cropped2_125 = img1[175:211, 207:228]
    cropped2_126 = img1[175:211, 228:249]
    cropped2_127 = img1[175:211, 249:270]
    cropped2_128 = img1[175:211, 271:292]
    cropped2_129 = img1[175:211, 292:313]
    cropped2_130 = img1[175:211, 313:335]
    cropped2_131 = img1[175:211, 335:356]
    cropped2_132 = img1[175:211, 357:378]
    cropped2_133 = img1[175:211, 378:399]
    cropped2_134 = img1[175:211, 399:420]
    cropped2_135 = img1[175:211, 423:444]
    cropped2_136 = img1[175:211, 445:466]
    cropped2_137 = img1[175:211, 466:487]
    cropped2_138 = img1[175:211, 487:508]
    cropped2_139 = img1[175:211, 508:529]
    cropped2_140 = img1[175:211, 529:550]
    cropped2_141 = img1[175:211, 550:571]
    cropped2_142 = img1[175:211, 571:592]
    cropped2_143 = img1[175:211, 592:613]
    cropped2_144 = img1[175:211, 614:635]
    cropped2_145 = img1[175:211, 635:656]
    cropped2_146 = img1[175:211, 656:678]
    cropped2_147 = img1[175:211, 678:699]
    cropped2_148 = img1[175:211, 699:720]
    cropped2_149 = img1[175:211, 722:743]
    cropped2_150 = img1[175:211, 743:764]
    cropped2_151 = img1[175:211, 764:786]
    cropped2_152 = img1[175:211, 786:807]
    cropped2_153 = img1[175:211, 808:829]
    cropped2_154 = img1[175:211, 829:850]
    cropped2_155 = img1[175:211, 850:871]
    cropped2_156 = img1[175:211, 873:894]
    cropped2_157 = img1[175:211, 895:916]
    cropped2_158 = img1[175:211, 916:937]
    cropped2_159 = img1[175:211, 937:958]
    cropped2_160 = img1[175:211, 958:979]
    #row5
    cropped2_161 = img1[211:247, 121:142]
    cropped2_162 = img1[211:247, 142:163]
    cropped2_163 = img1[211:247, 163:184]
    cropped2_164 = img1[211:247, 184:205]
    cropped2_165 = img1[211:247, 207:228]
    cropped2_166 = img1[211:247, 228:249]
    cropped2_167 = img1[211:247, 249:270]
    cropped2_168 = img1[211:247, 271:292]
    cropped2_169 = img1[211:247, 292:313]
    cropped2_170 = img1[211:247, 313:335]
    cropped2_171 = img1[211:247, 335:356]
    cropped2_172 = img1[211:247, 357:378]
    cropped2_173 = img1[211:247, 378:399]
    cropped2_174 = img1[211:247, 401:422]
    cropped2_175 = img1[211:247, 423:444]
    cropped2_176 = img1[211:247, 445:466]
    cropped2_177 = img1[211:247, 466:487]
    cropped2_178 = img1[211:247, 487:508]
    cropped2_179 = img1[211:247, 508:529]
    cropped2_180 = img1[211:247, 529:550]
    cropped2_181 = img1[211:247, 550:571]
    cropped2_182 = img1[211:247, 571:592]
    cropped2_183 = img1[211:247, 592:613]
    cropped2_184 = img1[211:247, 614:635]
    cropped2_185 = img1[211:247, 635:656]
    cropped2_186 = img1[211:247, 656:678]
    cropped2_187 = img1[211:247, 678:699]
    cropped2_188 = img1[211:247, 699:720]
    cropped2_189 = img1[211:247, 722:743]
    cropped2_190 = img1[211:247, 743:764]
    cropped2_191 = img1[211:247, 764:786]
    cropped2_192 = img1[211:247, 786:807]
    cropped2_193 = img1[211:247, 808:829]
    cropped2_194 = img1[211:247, 829:850]
    cropped2_195 = img1[211:247, 850:871]
    cropped2_196 = img1[211:247, 873:894]
    cropped2_197 = img1[211:247, 895:916]
    cropped2_198 = img1[211:247, 916:937]
    cropped2_199 = img1[211:247, 937:958]
    cropped2_200 = img1[211:247, 958:979]

    #row6
    cropped2_201 = img1[247:283, 121:142]
    cropped2_202 = img1[247:283, 142:163]
    cropped2_203 = img1[247:283, 163:184]
    cropped2_204 = img1[247:283, 184:205]
    cropped2_205 = img1[247:283, 207:228]
    cropped2_206 = img1[247:283, 228:249]
    cropped2_207 = img1[247:283, 249:270]
    cropped2_208 = img1[247:283, 271:292]
    cropped2_209 = img1[247:283, 292:313]
    cropped2_210 = img1[247:283, 313:335]
    cropped2_211 = img1[247:283, 335:356]
    cropped2_212 = img1[247:283, 357:378]
    cropped2_213 = img1[247:283, 378:399]
    cropped2_214 = img1[247:283, 401:422]
    cropped2_215 = img1[247:283, 423:444]
    cropped2_216 = img1[247:283, 445:466]
    cropped2_217 = img1[247:283, 466:487]
    cropped2_218 = img1[247:283, 487:508]
    cropped2_219 = img1[247:283, 508:529]
    cropped2_220 = img1[247:283, 529:550]
    cropped2_221 = img1[247:283, 550:571]
    cropped2_222 = img1[247:283, 571:592]
    cropped2_223 = img1[247:283, 592:613]
    cropped2_224 = img1[247:283, 614:635]
    cropped2_225 = img1[247:283, 635:656]
    cropped2_226 = img1[247:283, 656:678]
    cropped2_227 = img1[247:283, 678:699]
    cropped2_228 = img1[247:283, 699:720]
    cropped2_229 = img1[247:283, 722:743]
    cropped2_230 = img1[247:283, 743:764]
    cropped2_231 = img1[247:283, 764:786]
    cropped2_232 = img1[247:283, 786:807]
    cropped2_233 = img1[247:283, 808:829]
    cropped2_234 = img1[247:283, 829:850]
    cropped2_235 = img1[247:283, 850:871]
    cropped2_236 = img1[247:283, 873:894]
    cropped2_237 = img1[247:283, 895:916]
    cropped2_238 = img1[247:283, 916:937]
    cropped2_239 = img1[247:283, 937:958]
    cropped2_240 = img1[247:283, 958:979]
    #row7
    cropped2_241 = img1[283:319, 121:142]
    cropped2_242 = img1[283:319, 142:163]
    cropped2_243 = img1[283:319, 163:184]
    cropped2_244 = img1[283:319, 184:205]
    cropped2_245 = img1[283:319, 207:228]
    cropped2_246 = img1[283:319, 228:249]
    cropped2_247 = img1[283:319, 249:270]
    cropped2_248 = img1[283:319, 271:292]
    cropped2_249 = img1[283:319, 292:313]
    cropped2_250 = img1[283:319, 313:335]
    cropped2_251 = img1[283:319, 335:356]
    cropped2_252 = img1[283:319, 357:378]
    cropped2_253 = img1[283:319, 378:399]
    cropped2_254 = img1[283:319, 401:422]
    cropped2_255 = img1[283:319, 423:444]
    cropped2_256 = img1[283:319, 445:466]
    cropped2_257 = img1[283:319, 466:487]
    cropped2_258 = img1[283:319, 487:508]
    cropped2_259 = img1[283:319, 508:529]
    cropped2_260 = img1[283:319, 529:550]
    cropped2_261 = img1[283:319, 550:571]
    cropped2_262 = img1[283:319, 571:592]
    cropped2_263 = img1[283:319, 592:613]
    cropped2_264 = img1[283:319, 614:635]
    cropped2_265 = img1[283:319, 635:656]
    cropped2_266 = img1[283:319, 656:678]
    cropped2_267 = img1[283:319, 678:699]
    cropped2_268 = img1[283:319, 699:720]
    cropped2_269 = img1[283:319, 722:743]
    cropped2_270 = img1[283:319, 743:764]
    cropped2_271 = img1[283:319, 764:786]
    cropped2_272 = img1[283:319, 786:807]
    cropped2_273 = img1[283:319, 808:829]
    cropped2_274 = img1[283:319, 829:850]
    cropped2_275 = img1[283:319, 850:871]
    cropped2_276 = img1[283:319, 873:894]
    cropped2_277 = img1[283:319, 895:916]
    cropped2_278 = img1[283:319, 916:937]
    cropped2_279 = img1[283:319, 937:958]
    cropped2_280 = img1[283:319, 958:979]
    #row8
    cropped2_281 = img1[110:150, 28:53]
    cropped2_282 = img1[110:150, 53:78]
    cropped2_283 = img1[110:150, 78:103]
    cropped2_284 = img1[110:150, 103:128]
    cropped2_285 = img1[110:150, 128:153]
    cropped2_286 = img1[110:150, 153:178]
    cropped2_287 = img1[110:150, 175:200]
    cropped2_288 = img1[110:150, 200:225]
    cropped2_289 = img1[110:150, 223:248]
    cropped2_290 = img1[110:150, 248:273]
    cropped2_291 = img1[110:150, 272:296]
    cropped2_292 = img1[110:150, 295:320]
    cropped2_293 = img1[110:150, 320:344]
    cropped2_294 = img1[110:150, 343:368]
    cropped2_295 = img1[110:150, 367:391]
    cropped2_296 = img1[110:150, 391:415]
    cropped2_297 = img1[110:150, 415:439]
    cropped2_298 = img1[110:150, 439:463]
    cropped2_299 = img1[110:150, 462:486]
    cropped2_300 = img1[110:150, 486:510]
    cropped2_301 = img1[110:150, 510:533]
    cropped2_302 = img1[110:150, 532:556]
    cropped2_303 = img1[110:150, 557:579]
    cropped2_304 = img1[110:150, 579:603]
    cropped2_305 = img1[110:150, 603:625]
    cropped2_306 = img1[110:150, 625:648]
    cropped2_307 = img1[110:150, 648:672]
    cropped2_308 = img1[110:150, 672:696]
    cropped2_309 = img1[110:150, 695:719]
    cropped2_310 = img1[110:150, 719:743]
    cropped2_311 = img1[110:150, 742:766]
    cropped2_312 = img1[110:150, 766:790]
    cropped2_313 = img1[110:150, 789:813]
    cropped2_314 = img1[110:150, 813:837]
    cropped2_315 = img1[110:150, 837:861]
    cropped2_316 = img1[110:150, 860:884]
    cropped2_317 = img1[110:150, 884:908]
    cropped2_318 = img1[110:150, 908:932]
    cropped2_319 = img1[110:150, 932:956]
    cropped2_320 = img1[110:150, 956:980]
    #row9
    cropped2_321 = img1[150:190, 28:53]
    cropped2_322 = img1[150:190, 53:78]
    cropped2_323 = img1[150:190, 78:103]
    cropped2_324 = img1[150:190, 103:128]
    cropped2_325 = img1[150:190, 128:153]
    cropped2_326 = img1[150:190, 153:178]
    cropped2_327 = img1[150:190, 175:200]
    cropped2_328 = img1[150:190, 200:225]
    cropped2_329 = img1[150:190, 223:248]
    cropped2_330 = img1[150:190, 248:273]
    cropped2_331 = img1[150:190, 272:296]
    cropped2_332 = img1[150:190, 295:320]
    cropped2_333 = img1[150:190, 320:344]
    cropped2_334 = img1[150:190, 343:368]
    cropped2_335 = img1[150:190, 367:391]
    cropped2_336 = img1[150:190, 391:415]
    cropped2_337 = img1[150:190, 415:439]
    cropped2_338 = img1[150:190, 439:463]
    cropped2_339 = img1[150:190, 462:486]
    cropped2_340 = img1[150:190, 486:510]
    cropped2_341 = img1[150:190, 510:533]
    cropped2_342 = img1[150:190, 532:556]
    cropped2_343 = img1[150:190, 557:579]
    cropped2_344 = img1[150:190, 579:603]
    cropped2_345 = img1[150:190, 603:625]
    cropped2_346 = img1[150:190, 625:648]
    cropped2_347 = img1[150:190, 648:672]
    cropped2_348 = img1[150:190, 672:696]
    cropped2_349 = img1[150:190, 695:719]
    cropped2_350 = img1[150:190, 719:743]
    cropped2_351 = img1[150:190, 742:766]
    cropped2_352 = img1[150:190, 766:790]
    cropped2_353 = img1[150:190, 789:813]
    cropped2_354 = img1[150:190, 813:837]
    cropped2_355 = img1[150:190, 837:861]
    cropped2_356 = img1[150:190, 860:884]
    cropped2_357 = img1[150:190, 884:908]
    cropped2_358 = img1[150:190, 908:932]
    cropped2_359 = img1[150:190, 932:956]
    cropped2_360 = img1[150:190, 956:980]
    #row10
    cropped2_361 = img1[190:230, 28:53]
    cropped2_362 = img1[190:230, 53:78]
    cropped2_363 = img1[190:230, 78:103]
    cropped2_364 = img1[190:230, 103:128]
    cropped2_365 = img1[190:230, 128:153]
    cropped2_366 = img1[190:230, 153:178]
    cropped2_367 = img1[190:230, 175:200]
    cropped2_368 = img1[190:230, 200:225]
    cropped2_369 = img1[190:230, 223:248]
    cropped2_370 = img1[190:230, 248:273]
    cropped2_371 = img1[190:230, 272:296]
    cropped2_372 = img1[190:230, 295:320]
    cropped2_373 = img1[190:230, 320:344]
    cropped2_374 = img1[190:230, 343:368]
    cropped2_375 = img1[190:230, 367:391]
    cropped2_376 = img1[190:230, 391:415]
    cropped2_377 = img1[190:230, 415:439]
    cropped2_378 = img1[190:230, 439:463]
    cropped2_379 = img1[190:230, 462:486]
    cropped2_380 = img1[190:230, 486:510]
    cropped2_381 = img1[190:230, 510:533]
    cropped2_382 = img1[190:230, 532:556]
    cropped2_383 = img1[190:230, 557:579]
    cropped2_384 = img1[190:230, 579:603]
    cropped2_385 = img1[190:230, 603:625]
    cropped2_386 = img1[190:230, 625:648]
    cropped2_387 = img1[190:230, 648:672]
    cropped2_388 = img1[190:230, 672:696]
    cropped2_389 = img1[190:230, 695:719]
    cropped2_390 = img1[190:230, 719:743]
    cropped2_391 = img1[190:230, 742:766]
    cropped2_392 = img1[190:230, 766:790]
    cropped2_393 = img1[190:230, 789:813]
    cropped2_394 = img1[190:230, 813:837]
    cropped2_395 = img1[190:230, 837:861]
    cropped2_396 = img1[190:230, 860:884]
    cropped2_397 = img1[190:230, 884:908]
    cropped2_398 = img1[190:230, 908:932]
    cropped2_399 = img1[190:230, 932:956]
    cropped2_400 = img1[190:230, 956:980]
    #row11
    cropped2_401 = img1[30:70, 28:53]
    cropped2_402 = img1[30:70, 53:78]
    cropped2_403 = img1[30:70, 78:103]
    cropped2_404 = img1[30:70, 103:128]
    cropped2_405 = img1[30:70, 128:153]
    cropped2_406 = img1[30:70, 153:178]
    cropped2_407 = img1[30:70, 175:200]
    cropped2_408 = img1[30:70, 200:225]
    cropped2_409 = img1[30:70, 223:248]
    cropped2_410 = img1[30:70, 248:273]
    cropped2_411 = img1[30:70, 272:296]
    cropped2_412 = img1[30:70, 295:320]
    cropped2_413 = img1[30:70, 320:344]
    cropped2_414 = img1[30:70, 343:368]
    cropped2_415 = img1[30:70, 367:391]
    cropped2_416 = img1[30:70, 391:415]
    cropped2_417 = img1[30:70, 415:439]
    cropped2_418 = img1[30:70, 439:463]
    cropped2_419 = img1[30:70, 462:486]
    cropped2_420 = img1[30:70, 486:510]
    cropped2_421 = img1[30:70, 510:533]
    cropped2_422 = img1[30:70, 532:556]
    cropped2_423 = img1[30:70, 557:579]
    cropped2_424 = img1[30:70, 579:603]
    cropped2_425 = img1[30:70, 603:625]
    cropped2_426 = img1[30:70, 625:648]
    cropped2_427 = img1[30:70, 648:672]
    cropped2_428 = img1[30:70, 672:696]
    cropped2_429 = img1[30:70, 695:719]
    cropped2_430 = img1[30:70, 719:743]
    cropped2_431 = img1[30:70, 742:766]
    cropped2_432 = img1[30:70, 766:790]
    cropped2_433 = img1[30:70, 789:813]
    cropped2_434 = img1[30:70, 813:837]
    cropped2_435 = img1[30:70, 837:861]
    cropped2_436 = img1[30:70, 860:884]
    cropped2_437 = img1[30:70, 884:908]
    cropped2_438 = img1[30:70, 908:932]
    cropped2_439 = img1[30:70, 932:956]
    cropped2_440 = img1[30:70, 956:980]
    #row12
    cropped2_441 = img1[70:110, 28:53]
    cropped2_442 = img1[70:110, 53:78]
    cropped2_443 = img1[70:110, 78:103]
    cropped2_444 = img1[70:110, 103:128]
    cropped2_445 = img1[70:110, 128:153]
    cropped2_446 = img1[70:110, 153:178]
    cropped2_447 = img1[70:110, 175:200]
    cropped2_448 = img1[70:110, 200:225]
    cropped2_449 = img1[70:110, 223:248]
    cropped2_450 = img1[70:110, 248:273]
    cropped2_451 = img1[70:110, 272:296]
    cropped2_452 = img1[70:110, 295:320]
    cropped2_453 = img1[70:110, 320:344]
    cropped2_454 = img1[70:110, 343:368]
    cropped2_455 = img1[70:110, 367:391]
    cropped2_456 = img1[70:110, 391:415]
    cropped2_457 = img1[70:110, 415:439]
    cropped2_458 = img1[70:110, 439:463]
    cropped2_459 = img1[70:110, 462:486]
    cropped2_460 = img1[70:110, 486:510]
    cropped2_461 = img1[70:110, 510:533]
    cropped2_462 = img1[70:110, 532:556]
    cropped2_463 = img1[70:110, 557:579]
    cropped2_464 = img1[70:110, 579:603]
    cropped2_465 = img1[70:110, 603:625]
    cropped2_466 = img1[70:110, 625:648]
    cropped2_467 = img1[70:110, 648:672]
    cropped2_468 = img1[70:110, 672:696]
    cropped2_469 = img1[70:110, 695:719]
    cropped2_470 = img1[70:110, 719:743]
    cropped2_471 = img1[70:110, 742:766]
    cropped2_472 = img1[70:110, 766:790]
    cropped2_473 = img1[70:110, 789:813]
    cropped2_474 = img1[70:110, 813:837]
    cropped2_475 = img1[70:110, 837:861]
    cropped2_476 = img1[70:110, 860:884]
    cropped2_477 = img1[70:110, 884:908]
    cropped2_478 = img1[70:110, 908:932]
    cropped2_479 = img1[70:110, 932:956]
    cropped2_480 = img1[70:110, 956:980]
    #row13
    cropped2_481 = img1[110:150, 28:53]
    cropped2_482 = img1[110:150, 53:78]
    cropped2_483 = img1[110:150, 78:103]
    cropped2_484 = img1[110:150, 103:128]
    cropped2_485 = img1[110:150, 128:153]
    cropped2_486 = img1[110:150, 153:178]
    cropped2_487 = img1[110:150, 175:200]
    cropped2_488 = img1[110:150, 200:225]
    cropped2_489 = img1[110:150, 223:248]
    cropped2_490 = img1[110:150, 248:273]
    cropped2_491 = img1[110:150, 272:296]
    cropped2_492 = img1[110:150, 295:320]
    cropped2_493 = img1[110:150, 320:344]
    cropped2_494 = img1[110:150, 343:368]
    cropped2_495 = img1[110:150, 367:391]
    cropped2_496 = img1[110:150, 391:415]
    cropped2_497 = img1[110:150, 415:439]
    cropped2_498 = img1[110:150, 439:463]
    cropped2_499 = img1[110:150, 462:486]
    cropped2_500 = img1[110:150, 486:510]
    cropped2_501 = img1[110:150, 510:533]
    cropped2_502 = img1[110:150, 532:556]
    cropped2_503 = img1[110:150, 557:579]
    cropped2_504 = img1[110:150, 579:603]
    cropped2_505 = img1[110:150, 603:625]
    cropped2_506 = img1[110:150, 625:648]
    cropped2_507 = img1[110:150, 648:672]
    cropped2_508 = img1[110:150, 672:696]
    cropped2_509 = img1[110:150, 695:719]
    cropped2_510 = img1[110:150, 719:743]
    cropped2_511 = img1[110:150, 742:766]
    cropped2_512 = img1[110:150, 766:790]
    cropped2_513 = img1[110:150, 789:813]
    cropped2_514 = img1[110:150, 813:837]
    cropped2_515 = img1[110:150, 837:861]
    cropped2_516 = img1[110:150, 860:884]
    cropped2_517 = img1[110:150, 884:908]
    cropped2_518 = img1[110:150, 908:932]
    cropped2_519 = img1[110:150, 932:956]
    cropped2_520 = img1[110:150, 956:980]
    #row14
    cropped2_521 = img1[150:190, 28:53]
    cropped2_522 = img1[150:190, 53:78]
    cropped2_523 = img1[150:190, 78:103]
    cropped2_524 = img1[150:190, 103:128]
    cropped2_525 = img1[150:190, 128:153]
    cropped2_526 = img1[150:190, 153:178]
    cropped2_527 = img1[150:190, 175:200]
    cropped2_528 = img1[150:190, 200:225]
    cropped2_529 = img1[150:190, 223:248]
    cropped2_530 = img1[150:190, 248:273]
    cropped2_531 = img1[150:190, 272:296]
    cropped2_532 = img1[150:190, 295:320]
    cropped2_533 = img1[150:190, 320:344]
    cropped2_534 = img1[150:190, 343:368]
    cropped2_535 = img1[150:190, 367:391]
    cropped2_536 = img1[150:190, 391:415]
    cropped2_537 = img1[150:190, 415:439]
    cropped2_538 = img1[150:190, 439:463]
    cropped2_539 = img1[150:190, 462:486]
    cropped2_540 = img1[150:190, 486:510]
    cropped2_541 = img1[150:190, 510:533]
    cropped2_542 = img1[150:190, 532:556]
    cropped2_543 = img1[150:190, 557:579]
    cropped2_544 = img1[150:190, 579:603]
    cropped2_545 = img1[150:190, 603:625]
    cropped2_546 = img1[150:190, 625:648]
    cropped2_547 = img1[150:190, 648:672]
    cropped2_548 = img1[150:190, 672:696]
    cropped2_549 = img1[150:190, 695:719]
    cropped2_550 = img1[150:190, 719:743]
    cropped2_551 = img1[150:190, 742:766]
    cropped2_552 = img1[150:190, 766:790]
    cropped2_553 = img1[150:190, 789:813]
    cropped2_554 = img1[150:190, 813:837]
    cropped2_555 = img1[150:190, 837:861]
    cropped2_556 = img1[150:190, 860:884]
    cropped2_557 = img1[150:190, 884:908]
    cropped2_558 = img1[150:190, 908:932]
    cropped2_559 = img1[150:190, 932:956]
    cropped2_560 = img1[150:190, 956:980]

    #writing for crops folder 1
    #writing for crops22
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
    cv2.imwrite(f'{outdir}/cropped_281.jpg',cropped2_281)
    cv2.imwrite(f'{outdir}/cropped_282.jpg',cropped2_282)
    cv2.imwrite(f'{outdir}/cropped_283.jpg',cropped2_283)
    cv2.imwrite(f'{outdir}/cropped_284.jpg',cropped2_284)
    cv2.imwrite(f'{outdir}/cropped_285.jpg',cropped2_285)
    cv2.imwrite(f'{outdir}/cropped_286.jpg',cropped2_286)
    cv2.imwrite(f'{outdir}/cropped_287.jpg',cropped2_287)
    cv2.imwrite(f'{outdir}/cropped_288.jpg',cropped2_288)
    cv2.imwrite(f'{outdir}/cropped_289.jpg',cropped2_289)
    cv2.imwrite(f'{outdir}/cropped_290.jpg',cropped2_290)
    cv2.imwrite(f'{outdir}/cropped_291.jpg',cropped2_291)
    cv2.imwrite(f'{outdir}/cropped_292.jpg',cropped2_292)
    cv2.imwrite(f'{outdir}/cropped_293.jpg',cropped2_293)
    cv2.imwrite(f'{outdir}/cropped_294.jpg',cropped2_294)
    cv2.imwrite(f'{outdir}/cropped_295.jpg',cropped2_295)
    cv2.imwrite(f'{outdir}/cropped_296.jpg',cropped2_296)
    cv2.imwrite(f'{outdir}/cropped_297.jpg',cropped2_297)
    cv2.imwrite(f'{outdir}/cropped_298.jpg',cropped2_298)
    cv2.imwrite(f'{outdir}/cropped_299.jpg',cropped2_299)
    cv2.imwrite(f'{outdir}/cropped_300.jpg',cropped2_300)
    cv2.imwrite(f'{outdir}/cropped_301.jpg',cropped2_301)
    cv2.imwrite(f'{outdir}/cropped_302.jpg',cropped2_302)
    cv2.imwrite(f'{outdir}/cropped_303.jpg',cropped2_303)
    cv2.imwrite(f'{outdir}/cropped_304.jpg',cropped2_304)
    cv2.imwrite(f'{outdir}/cropped_305.jpg',cropped2_305)
    cv2.imwrite(f'{outdir}/cropped_306.jpg',cropped2_306)
    cv2.imwrite(f'{outdir}/cropped_307.jpg',cropped2_307)
    cv2.imwrite(f'{outdir}/cropped_308.jpg',cropped2_308)
    cv2.imwrite(f'{outdir}/cropped_309.jpg',cropped2_309)
    cv2.imwrite(f'{outdir}/cropped_310.jpg',cropped2_310)
    cv2.imwrite(f'{outdir}/cropped_311.jpg',cropped2_311)
    cv2.imwrite(f'{outdir}/cropped_312.jpg',cropped2_312)
    cv2.imwrite(f'{outdir}/cropped_313.jpg',cropped2_313)
    cv2.imwrite(f'{outdir}/cropped_314.jpg',cropped2_314)
    cv2.imwrite(f'{outdir}/cropped_315.jpg',cropped2_315)
    cv2.imwrite(f'{outdir}/cropped_316.jpg',cropped2_316)
    cv2.imwrite(f'{outdir}/cropped_317.jpg',cropped2_317)
    cv2.imwrite(f'{outdir}/cropped_318.jpg',cropped2_318)
    cv2.imwrite(f'{outdir}/cropped_319.jpg',cropped2_319)
    cv2.imwrite(f'{outdir}/cropped_320.jpg',cropped2_320)
    cv2.imwrite(f'{outdir}/cropped_321.jpg',cropped2_321)
    cv2.imwrite(f'{outdir}/cropped_322.jpg',cropped2_322)
    cv2.imwrite(f'{outdir}/cropped_323.jpg',cropped2_323)
    cv2.imwrite(f'{outdir}/cropped_324.jpg',cropped2_324)
    cv2.imwrite(f'{outdir}/cropped_325.jpg',cropped2_325)
    cv2.imwrite(f'{outdir}/cropped_326.jpg',cropped2_326)
    cv2.imwrite(f'{outdir}/cropped_327.jpg',cropped2_327)
    cv2.imwrite(f'{outdir}/cropped_328.jpg',cropped2_328)
    cv2.imwrite(f'{outdir}/cropped_329.jpg',cropped2_329)
    cv2.imwrite(f'{outdir}/cropped_330.jpg',cropped2_330)
    cv2.imwrite(f'{outdir}/cropped_331.jpg',cropped2_331)
    cv2.imwrite(f'{outdir}/cropped_332.jpg',cropped2_332)
    cv2.imwrite(f'{outdir}/cropped_333.jpg',cropped2_333)
    cv2.imwrite(f'{outdir}/cropped_334.jpg',cropped2_334)
    cv2.imwrite(f'{outdir}/cropped_335.jpg',cropped2_335)
    cv2.imwrite(f'{outdir}/cropped_336.jpg',cropped2_336)
    cv2.imwrite(f'{outdir}/cropped_337.jpg',cropped2_337)
    cv2.imwrite(f'{outdir}/cropped_338.jpg',cropped2_338)
    cv2.imwrite(f'{outdir}/cropped_339.jpg',cropped2_339)
    cv2.imwrite(f'{outdir}/cropped_340.jpg',cropped2_340)
    cv2.imwrite(f'{outdir}/cropped_341.jpg',cropped2_341)
    cv2.imwrite(f'{outdir}/cropped_342.jpg',cropped2_342)
    cv2.imwrite(f'{outdir}/cropped_343.jpg',cropped2_343)
    cv2.imwrite(f'{outdir}/cropped_344.jpg',cropped2_344)
    cv2.imwrite(f'{outdir}/cropped_345.jpg',cropped2_345)
    cv2.imwrite(f'{outdir}/cropped_346.jpg',cropped2_346)
    cv2.imwrite(f'{outdir}/cropped_347.jpg',cropped2_347)
    cv2.imwrite(f'{outdir}/cropped_348.jpg',cropped2_348)
    cv2.imwrite(f'{outdir}/cropped_349.jpg',cropped2_349)
    cv2.imwrite(f'{outdir}/cropped_350.jpg',cropped2_350)
    cv2.imwrite(f'{outdir}/cropped_351.jpg',cropped2_351)
    cv2.imwrite(f'{outdir}/cropped_352.jpg',cropped2_352)
    cv2.imwrite(f'{outdir}/cropped_353.jpg',cropped2_353)
    cv2.imwrite(f'{outdir}/cropped_354.jpg',cropped2_354)
    cv2.imwrite(f'{outdir}/cropped_355.jpg',cropped2_355)
    cv2.imwrite(f'{outdir}/cropped_356.jpg',cropped2_356)
    cv2.imwrite(f'{outdir}/cropped_357.jpg',cropped2_357)
    cv2.imwrite(f'{outdir}/cropped_358.jpg',cropped2_358)
    cv2.imwrite(f'{outdir}/cropped_359.jpg',cropped2_359)
    cv2.imwrite(f'{outdir}/cropped_360.jpg',cropped2_360)
    cv2.imwrite(f'{outdir}/cropped_361.jpg',cropped2_361)
    cv2.imwrite(f'{outdir}/cropped_362.jpg',cropped2_362)
    cv2.imwrite(f'{outdir}/cropped_363.jpg',cropped2_363)
    cv2.imwrite(f'{outdir}/cropped_364.jpg',cropped2_364)
    cv2.imwrite(f'{outdir}/cropped_365.jpg',cropped2_365)
    cv2.imwrite(f'{outdir}/cropped_366.jpg',cropped2_366)
    cv2.imwrite(f'{outdir}/cropped_367.jpg',cropped2_367)
    cv2.imwrite(f'{outdir}/cropped_368.jpg',cropped2_368)
    cv2.imwrite(f'{outdir}/cropped_369.jpg',cropped2_369)
    cv2.imwrite(f'{outdir}/cropped_370.jpg',cropped2_370)
    cv2.imwrite(f'{outdir}/cropped_371.jpg',cropped2_371)
    cv2.imwrite(f'{outdir}/cropped_372.jpg',cropped2_372)
    cv2.imwrite(f'{outdir}/cropped_373.jpg',cropped2_373)
    cv2.imwrite(f'{outdir}/cropped_374.jpg',cropped2_374)
    cv2.imwrite(f'{outdir}/cropped_375.jpg',cropped2_375)
    cv2.imwrite(f'{outdir}/cropped_376.jpg',cropped2_376)
    cv2.imwrite(f'{outdir}/cropped_377.jpg',cropped2_377)
    cv2.imwrite(f'{outdir}/cropped_378.jpg',cropped2_378)
    cv2.imwrite(f'{outdir}/cropped_379.jpg',cropped2_379)
    cv2.imwrite(f'{outdir}/cropped_380.jpg',cropped2_380)
    cv2.imwrite(f'{outdir}/cropped_381.jpg',cropped2_381)
    cv2.imwrite(f'{outdir}/cropped_382.jpg',cropped2_382)
    cv2.imwrite(f'{outdir}/cropped_383.jpg',cropped2_383)
    cv2.imwrite(f'{outdir}/cropped_384.jpg',cropped2_384)
    cv2.imwrite(f'{outdir}/cropped_385.jpg',cropped2_385)
    cv2.imwrite(f'{outdir}/cropped_386.jpg',cropped2_386)
    cv2.imwrite(f'{outdir}/cropped_387.jpg',cropped2_387)
    cv2.imwrite(f'{outdir}/cropped_388.jpg',cropped2_388)
    cv2.imwrite(f'{outdir}/cropped_389.jpg',cropped2_389)
    cv2.imwrite(f'{outdir}/cropped_390.jpg',cropped2_390)
    cv2.imwrite(f'{outdir}/cropped_391.jpg',cropped2_391)
    cv2.imwrite(f'{outdir}/cropped_392.jpg',cropped2_392)
    cv2.imwrite(f'{outdir}/cropped_393.jpg',cropped2_393)
    cv2.imwrite(f'{outdir}/cropped_394.jpg',cropped2_394)
    cv2.imwrite(f'{outdir}/cropped_395.jpg',cropped2_395)
    cv2.imwrite(f'{outdir}/cropped_396.jpg',cropped2_396)
    cv2.imwrite(f'{outdir}/cropped_397.jpg',cropped2_397)
    cv2.imwrite(f'{outdir}/cropped_398.jpg',cropped2_398)
    cv2.imwrite(f'{outdir}/cropped_399.jpg',cropped2_399)
    cv2.imwrite(f'{outdir}/cropped_400.jpg',cropped2_400)
    cv2.imwrite(f'{outdir}/cropped_401.jpg',cropped2_401)
    cv2.imwrite(f'{outdir}/cropped_402.jpg',cropped2_402)
    cv2.imwrite(f'{outdir}/cropped_403.jpg',cropped2_403)
    cv2.imwrite(f'{outdir}/cropped_404.jpg',cropped2_404)
    cv2.imwrite(f'{outdir}/cropped_405.jpg',cropped2_405)
    cv2.imwrite(f'{outdir}/cropped_406.jpg',cropped2_406)
    cv2.imwrite(f'{outdir}/cropped_407.jpg',cropped2_407)
    cv2.imwrite(f'{outdir}/cropped_408.jpg',cropped2_408)
    cv2.imwrite(f'{outdir}/cropped_409.jpg',cropped2_409)
    cv2.imwrite(f'{outdir}/cropped_410.jpg',cropped2_410)
    cv2.imwrite(f'{outdir}/cropped_411.jpg',cropped2_411)
    cv2.imwrite(f'{outdir}/cropped_412.jpg',cropped2_412)
    cv2.imwrite(f'{outdir}/cropped_413.jpg',cropped2_413)
    cv2.imwrite(f'{outdir}/cropped_414.jpg',cropped2_414)
    cv2.imwrite(f'{outdir}/cropped_415.jpg',cropped2_415)
    cv2.imwrite(f'{outdir}/cropped_416.jpg',cropped2_416)
    cv2.imwrite(f'{outdir}/cropped_417.jpg',cropped2_417)
    cv2.imwrite(f'{outdir}/cropped_418.jpg',cropped2_418)
    cv2.imwrite(f'{outdir}/cropped_419.jpg',cropped2_419)
    cv2.imwrite(f'{outdir}/cropped_420.jpg',cropped2_420)
    cv2.imwrite(f'{outdir}/cropped_421.jpg',cropped2_421)
    cv2.imwrite(f'{outdir}/cropped_422.jpg',cropped2_422)
    cv2.imwrite(f'{outdir}/cropped_423.jpg',cropped2_423)
    cv2.imwrite(f'{outdir}/cropped_424.jpg',cropped2_424)
    cv2.imwrite(f'{outdir}/cropped_425.jpg',cropped2_425)
    cv2.imwrite(f'{outdir}/cropped_426.jpg',cropped2_426)
    cv2.imwrite(f'{outdir}/cropped_427.jpg',cropped2_427)
    cv2.imwrite(f'{outdir}/cropped_428.jpg',cropped2_428)
    cv2.imwrite(f'{outdir}/cropped_429.jpg',cropped2_429)
    cv2.imwrite(f'{outdir}/cropped_430.jpg',cropped2_430)
    cv2.imwrite(f'{outdir}/cropped_431.jpg',cropped2_431)
    cv2.imwrite(f'{outdir}/cropped_432.jpg',cropped2_432)
    cv2.imwrite(f'{outdir}/cropped_433.jpg',cropped2_433)
    cv2.imwrite(f'{outdir}/cropped_434.jpg',cropped2_434)
    cv2.imwrite(f'{outdir}/cropped_435.jpg',cropped2_435)
    cv2.imwrite(f'{outdir}/cropped_436.jpg',cropped2_436)
    cv2.imwrite(f'{outdir}/cropped_437.jpg',cropped2_437)
    cv2.imwrite(f'{outdir}/cropped_438.jpg',cropped2_438)
    cv2.imwrite(f'{outdir}/cropped_439.jpg',cropped2_439)
    cv2.imwrite(f'{outdir}/cropped_440.jpg',cropped2_440)
    cv2.imwrite(f'{outdir}/cropped_441.jpg',cropped2_441)
    cv2.imwrite(f'{outdir}/cropped_442.jpg',cropped2_442)
    cv2.imwrite(f'{outdir}/cropped_443.jpg',cropped2_443)
    cv2.imwrite(f'{outdir}/cropped_444.jpg',cropped2_444)
    cv2.imwrite(f'{outdir}/cropped_445.jpg',cropped2_445)
    cv2.imwrite(f'{outdir}/cropped_446.jpg',cropped2_446)
    cv2.imwrite(f'{outdir}/cropped_447.jpg',cropped2_447)
    cv2.imwrite(f'{outdir}/cropped_448.jpg',cropped2_448)
    cv2.imwrite(f'{outdir}/cropped_449.jpg',cropped2_449)
    cv2.imwrite(f'{outdir}/cropped_450.jpg',cropped2_450)
    cv2.imwrite(f'{outdir}/cropped_451.jpg',cropped2_451)
    cv2.imwrite(f'{outdir}/cropped_452.jpg',cropped2_452)
    cv2.imwrite(f'{outdir}/cropped_453.jpg',cropped2_453)
    cv2.imwrite(f'{outdir}/cropped_454.jpg',cropped2_454)
    cv2.imwrite(f'{outdir}/cropped_455.jpg',cropped2_455)
    cv2.imwrite(f'{outdir}/cropped_456.jpg',cropped2_456)
    cv2.imwrite(f'{outdir}/cropped_457.jpg',cropped2_457)
    cv2.imwrite(f'{outdir}/cropped_458.jpg',cropped2_458)
    cv2.imwrite(f'{outdir}/cropped_459.jpg',cropped2_459)
    cv2.imwrite(f'{outdir}/cropped_460.jpg',cropped2_460)
    cv2.imwrite(f'{outdir}/cropped_461.jpg',cropped2_461)
    cv2.imwrite(f'{outdir}/cropped_462.jpg',cropped2_462)
    cv2.imwrite(f'{outdir}/cropped_463.jpg',cropped2_463)
    cv2.imwrite(f'{outdir}/cropped_464.jpg',cropped2_464)
    cv2.imwrite(f'{outdir}/cropped_465.jpg',cropped2_465)
    cv2.imwrite(f'{outdir}/cropped_466.jpg',cropped2_466)
    cv2.imwrite(f'{outdir}/cropped_467.jpg',cropped2_467)
    cv2.imwrite(f'{outdir}/cropped_468.jpg',cropped2_468)
    cv2.imwrite(f'{outdir}/cropped_469.jpg',cropped2_469)
    cv2.imwrite(f'{outdir}/cropped_470.jpg',cropped2_470)
    cv2.imwrite(f'{outdir}/cropped_471.jpg',cropped2_471)
    cv2.imwrite(f'{outdir}/cropped_472.jpg',cropped2_472)
    cv2.imwrite(f'{outdir}/cropped_473.jpg',cropped2_473)
    cv2.imwrite(f'{outdir}/cropped_474.jpg',cropped2_474)
    cv2.imwrite(f'{outdir}/cropped_475.jpg',cropped2_475)
    cv2.imwrite(f'{outdir}/cropped_476.jpg',cropped2_476)
    cv2.imwrite(f'{outdir}/cropped_477.jpg',cropped2_477)
    cv2.imwrite(f'{outdir}/cropped_478.jpg',cropped2_478)
    cv2.imwrite(f'{outdir}/cropped_479.jpg',cropped2_479)
    cv2.imwrite(f'{outdir}/cropped_480.jpg',cropped2_480)
    cv2.imwrite(f'{outdir}/cropped_481.jpg',cropped2_481)
    cv2.imwrite(f'{outdir}/cropped_482.jpg',cropped2_482)
    cv2.imwrite(f'{outdir}/cropped_483.jpg',cropped2_483)
    cv2.imwrite(f'{outdir}/cropped_484.jpg',cropped2_484)
    cv2.imwrite(f'{outdir}/cropped_485.jpg',cropped2_485)
    cv2.imwrite(f'{outdir}/cropped_486.jpg',cropped2_486)
    cv2.imwrite(f'{outdir}/cropped_487.jpg',cropped2_487)
    cv2.imwrite(f'{outdir}/cropped_488.jpg',cropped2_488)
    cv2.imwrite(f'{outdir}/cropped_489.jpg',cropped2_489)
    cv2.imwrite(f'{outdir}/cropped_490.jpg',cropped2_490)
    cv2.imwrite(f'{outdir}/cropped_491.jpg',cropped2_491)
    cv2.imwrite(f'{outdir}/cropped_492.jpg',cropped2_492)
    cv2.imwrite(f'{outdir}/cropped_493.jpg',cropped2_493)
    cv2.imwrite(f'{outdir}/cropped_494.jpg',cropped2_494)
    cv2.imwrite(f'{outdir}/cropped_495.jpg',cropped2_495)
    cv2.imwrite(f'{outdir}/cropped_496.jpg',cropped2_496)
    cv2.imwrite(f'{outdir}/cropped_497.jpg',cropped2_497)
    cv2.imwrite(f'{outdir}/cropped_498.jpg',cropped2_498)
    cv2.imwrite(f'{outdir}/cropped_499.jpg',cropped2_499)
    cv2.imwrite(f'{outdir}/cropped_500.jpg',cropped2_500)
    cv2.imwrite(f'{outdir}/cropped_501.jpg',cropped2_501)
    cv2.imwrite(f'{outdir}/cropped_502.jpg',cropped2_502)
    cv2.imwrite(f'{outdir}/cropped_503.jpg',cropped2_503)
    cv2.imwrite(f'{outdir}/cropped_504.jpg',cropped2_504)
    cv2.imwrite(f'{outdir}/cropped_505.jpg',cropped2_505)
    cv2.imwrite(f'{outdir}/cropped_506.jpg',cropped2_506)
    cv2.imwrite(f'{outdir}/cropped_507.jpg',cropped2_507)
    cv2.imwrite(f'{outdir}/cropped_508.jpg',cropped2_508)
    cv2.imwrite(f'{outdir}/cropped_509.jpg',cropped2_509)
    cv2.imwrite(f'{outdir}/cropped_510.jpg',cropped2_510)
    cv2.imwrite(f'{outdir}/cropped_511.jpg',cropped2_511)
    cv2.imwrite(f'{outdir}/cropped_512.jpg',cropped2_512)
    cv2.imwrite(f'{outdir}/cropped_513.jpg',cropped2_513)
    cv2.imwrite(f'{outdir}/cropped_514.jpg',cropped2_514)
    cv2.imwrite(f'{outdir}/cropped_515.jpg',cropped2_515)
    cv2.imwrite(f'{outdir}/cropped_516.jpg',cropped2_516)
    cv2.imwrite(f'{outdir}/cropped_517.jpg',cropped2_517)
    cv2.imwrite(f'{outdir}/cropped_518.jpg',cropped2_518)
    cv2.imwrite(f'{outdir}/cropped_519.jpg',cropped2_519)
    cv2.imwrite(f'{outdir}/cropped_520.jpg',cropped2_520)
    cv2.imwrite(f'{outdir}/cropped_521.jpg',cropped2_521)
    cv2.imwrite(f'{outdir}/cropped_522.jpg',cropped2_522)
    cv2.imwrite(f'{outdir}/cropped_523.jpg',cropped2_523)
    cv2.imwrite(f'{outdir}/cropped_524.jpg',cropped2_524)
    cv2.imwrite(f'{outdir}/cropped_525.jpg',cropped2_525)
    cv2.imwrite(f'{outdir}/cropped_526.jpg',cropped2_526)
    cv2.imwrite(f'{outdir}/cropped_527.jpg',cropped2_527)
    cv2.imwrite(f'{outdir}/cropped_528.jpg',cropped2_528)
    cv2.imwrite(f'{outdir}/cropped_529.jpg',cropped2_529)
    cv2.imwrite(f'{outdir}/cropped_530.jpg',cropped2_530)
    cv2.imwrite(f'{outdir}/cropped_531.jpg',cropped2_531)
    cv2.imwrite(f'{outdir}/cropped_532.jpg',cropped2_532)
    cv2.imwrite(f'{outdir}/cropped_533.jpg',cropped2_533)
    cv2.imwrite(f'{outdir}/cropped_534.jpg',cropped2_534)
    cv2.imwrite(f'{outdir}/cropped_535.jpg',cropped2_535)
    cv2.imwrite(f'{outdir}/cropped_536.jpg',cropped2_536)
    cv2.imwrite(f'{outdir}/cropped_537.jpg',cropped2_537)
    cv2.imwrite(f'{outdir}/cropped_538.jpg',cropped2_538)
    cv2.imwrite(f'{outdir}/cropped_539.jpg',cropped2_539)
    cv2.imwrite(f'{outdir}/cropped_540.jpg',cropped2_540)
    cv2.imwrite(f'{outdir}/cropped_541.jpg',cropped2_541)
    cv2.imwrite(f'{outdir}/cropped_542.jpg',cropped2_542)
    cv2.imwrite(f'{outdir}/cropped_543.jpg',cropped2_543)
    cv2.imwrite(f'{outdir}/cropped_544.jpg',cropped2_544)
    cv2.imwrite(f'{outdir}/cropped_545.jpg',cropped2_545)
    cv2.imwrite(f'{outdir}/cropped_546.jpg',cropped2_546)
    cv2.imwrite(f'{outdir}/cropped_547.jpg',cropped2_547)
    cv2.imwrite(f'{outdir}/cropped_548.jpg',cropped2_548)
    cv2.imwrite(f'{outdir}/cropped_549.jpg',cropped2_549)
    cv2.imwrite(f'{outdir}/cropped_550.jpg',cropped2_550)
    cv2.imwrite(f'{outdir}/cropped_551.jpg',cropped2_551)
    cv2.imwrite(f'{outdir}/cropped_552.jpg',cropped2_552)
    cv2.imwrite(f'{outdir}/cropped_553.jpg',cropped2_553)
    cv2.imwrite(f'{outdir}/cropped_554.jpg',cropped2_554)
    cv2.imwrite(f'{outdir}/cropped_555.jpg',cropped2_555)
    cv2.imwrite(f'{outdir}/cropped_556.jpg',cropped2_556)
    cv2.imwrite(f'{outdir}/cropped_557.jpg',cropped2_557)
    cv2.imwrite(f'{outdir}/cropped_558.jpg',cropped2_558)
    cv2.imwrite(f'{outdir}/cropped_559.jpg',cropped2_559)
    cv2.imwrite(f'{outdir}/cropped_560.jpg',cropped2_560)
    
if __name__ == '__main__':
    img = input("Image path: ")
    outdir = input("Output path: ")
    preProcess(img, outdir, False)
