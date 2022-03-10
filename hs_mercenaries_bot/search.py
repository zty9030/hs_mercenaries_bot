import cv2
import logging
from matplotlib import pyplot as plt
import numpy as np


def feature_match(large_img,small_img,min_match_nums= 15, debug=True):
    train = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
    query = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()

    try:
        query = small_img
        kp1, des1 = sift.detectAndCompute(query,None)
        kp2, des2 = sift.detectAndCompute(train,None)
        if kp1 == () or kp2 == ():
            logging.debug("Err - good %s for action %s , less than threshold %s" % (0,'action',min_match_nums))
            return [],0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2) 
        good = []
        ratio  = 0.7
        for m,n in matches:
            if m.distance < ratio *n.distance:
                good.append([m])

        if debug:
            img_match = np.empty(( large_img.shape[0] + 100, query.shape[1] + large_img.shape[1], 3), dtype=np.uint8)
            img_debug = cv2.drawMatchesKnn(query,kp1,train,kp2,good,flags=2,outImg=img_match)
            plt.imshow(img_debug)

        ## re-group the pts .
        pts_groups = []
        pts = np.float32([kp2[m[0].trainIdx].pt for m in good])
        qh, qw = query.shape[:2]
        for pt in pts:
            in_group = False
            for pts_group in pts_groups:
                if pt in pts_group:
                    in_group = True
                    break
            if not in_group:
                approx_pts_rows = np.where(   (pts[:,1] >= pt[1] - qh)   &  (pts[:,1] <=   pt[1] + qh  ) & (pts[:,0] >= pt[0] - qw)   &  (pts[:,0] <=   pt[0] + qw  ) )
                approx_pts = pts[approx_pts_rows] 
                pts_groups.append(approx_pts)
        
        max_number_group = []
        for pts_group in pts_groups:
            if len(pts_group) > len(max_number_group):
                max_number_group = pts_group

        if len(max_number_group) < min_match_nums:
            logging.debug("Err - good %s for action %s , less than threshold %s" % (len(max_number_group),'action',min_match_nums))
            return [] , len(good)
        else:
            logging.debug(" good %s for action %s" % (len(max_number_group),'action'))

        x,y = np.mean(max_number_group, axis=0)
        return [int(x),int(y)] ,len(max_number_group)
    except Exception as e:
        print(e)
        return [],0
    
def find_bounties(img_path):
    large_img = cv2.imread(img_path)
    small_img = cv2.imread('files/iphone11pm/find_bounties.jpeg')
    result = feature_match(large_img, small_img)
    return result[0]
    pass

def find_team(img_path):
    pass

def find_map(img_path):
    pass

def find_battle(img_path):
    pass

def find_treasure(img_path):
    pass

def find_chest(img_path):
    pass
