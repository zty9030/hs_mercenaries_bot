"""search"""
import logging
import random
import os
import numpy as np
import cv2

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))+'/../'


class HSContonurMatch:
    def __init__(self, debug=True) -> None:
        self.debug = debug

    @staticmethod
    def crop_img(img, left_x_cut_pect, right_x_cut_pect,
                 top_y_cut_pect, bottom_y_cut_pect):
        height, weight = img.shape[:2]
        cropped_image = img[
            int(height*top_y_cut_pect): height - int(height*bottom_y_cut_pect),
            int(weight*left_x_cut_pect): weight - int(weight*right_x_cut_pect)]
        return cropped_image

    @staticmethod
    def sort_2d_array(array, by_x=True):
        if len(array) == 0:
            return array
        array_a = np.array(array)
        if by_x:
            ind = np.lexsort((array_a[:, 1], array_a[:, 0]))
        else:
            ind = np.lexsort((array_a[:, 0], array_a[:, 1]))
        return array_a[ind]

    def list_enemy_and_minion(self, imgpath):
        img = cv2.imread(imgpath)
        left_x_cut_pect = 1/4
        right_x_cut_pect = 1/4
        crop_img = HSContonurMatch.crop_img(
            img, left_x_cut_pect, right_x_cut_pect, 0, 0)
        img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img_gray, kernel)
        img_decrease_bright = cv2.divide(img_gray, dilate_img, scale=255)
        img_decrease_bright = cv2.divide(img_decrease_bright, dilate_img, scale=255)
        self.debug_img("list_decrease_bright",
                       img_decrease_bright)
        canny_img = cv2.Canny(img_decrease_bright, 0, 255)
        canny_img = cv2.dilate(canny_img, kernel)
        self.debug_img("list_canny_img", canny_img)
        contours, _ = cv2.findContours(
            canny_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        minion_locations = []
        enemy_locations = []
        all_locations = []
        height, weight = img.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            arc_length = cv2.arcLength(contour, False)
            if ((arc_length >= 250) and (area >= 100)):
                extTop = tuple(contour[contour[:, :, 1].argmin()][0])
                extBot = tuple(contour[contour[:, :, 1].argmax()][0])
                if height * 1 / 4 < extTop[1] <= height*3 / 4:
                    all_locations.append(
                        [extBot[0] + int(weight*left_x_cut_pect), extBot[1] - 50])
                if height * .55 < extTop[1] <= height*3 / 4:  # the minion position
                    minion_locations.append(
                        [extTop[0] - 30 + int(weight*left_x_cut_pect),
                         extTop[1] + 60])
                if height * 1 / 4 <= extBot[1] <= height*1 / 2:  # the enemy position
                    enemy_locations.append(
                        [extBot[0] + int(weight*left_x_cut_pect), extBot[1] - 50])
                if self.debug:
                    cv2.drawContours(
                        crop_img, [contour], 0,
                        (random.randint(0, 256), random.randint(0, 256),
                         random.randint(0, 256)), 2)
        self.debug_img("list_contour_img", crop_img)
        # print(minion_locations, enemy_locations)

        minion_locations = HSContonurMatch.sort_2d_array(minion_locations)
        enemy_locations = HSContonurMatch.sort_2d_array(enemy_locations)
        all_locations = HSContonurMatch.sort_2d_array(all_locations)
        if self.debug:
            self._debug_img_with_text(enemy_locations, img)
            self._debug_img_with_text(minion_locations, img)
            self.debug_img("list", img)
        if len(enemy_locations) == 0:
            return all_locations, []
        return enemy_locations, minion_locations

    def list_card_spells(self, imgpath):
        def position(w, h, cx, cy):
            if h*0.35 < cy < h*0.55 and w*0.3 < cx < w*0.7:
                return True
            return False
        return self._hsv_contour(
            imgpath, (30, 0, 255), (90, 255, 255), 200, 300, -10, 30,
            kernal=[9, 9], contour_position=position, img_name='spell')

    def _hsv_contour(self, imgpath, min_hsv, max_hsv,
                     min_arcLength, min_area,
                     cx_margin=0, cy_margin=0, kernal=[5, 5],
                     contour_position=None, img_name='dilate_img_hsv_contour'):
        def everywhere(w, h, cx, cy):
            return True
        if contour_position is None:
            contour_position = everywhere

        img = cv2.imread(imgpath)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hs_mask = cv2.inRange(img_hsv, min_hsv, max_hsv)
        kernel = np.ones(kernal, 'uint8')
        dilate_img = cv2.dilate(hs_mask, kernel)
        self.debug_img(img_name, dilate_img)
        contours, _ = cv2.findContours(
            dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        locations = []
        h, w = img.shape[:2]
        for contour in contours:
            arcLength = cv2.arcLength(contour, False)
            area = cv2.contourArea(contour)
            if arcLength >= min_arcLength and area >= min_area:
                M = cv2.moments(contour)
                if M['m00'] != 0.0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    if contour_position(w, h, cx, cy):
                        locations.append([cx + cx_margin, cy + cy_margin])
                        if self.debug:
                            cv2.drawContours(
                                img, [contour], 0,
                                (random.randint(0, 256),
                                 random.randint(0, 256),
                                 random.randint(0, 256)), 2)
        locations = HSContonurMatch.sort_2d_array(locations)
        self.debug_img(f"{img_name}_contour", img)
        locations = HSContonurMatch.sort_2d_array(locations)
        if self.debug:
            self._debug_img_with_text(locations, img)
            self.debug_img(f"{img_name}_final", img)
        return locations

    def list_map_moves(self, imgpath):
        def position(w, h, cx, cy):
            if w*0.15 < cx < 0.71 * w and 0.1*h <= cy <= 0.8*h:
                return True
            return False
        locations_1 = self._hsv_contour(
            imgpath, (19, 160, 100), (39, 240, 205), 200, 400, 0, -10,
            kernal=[13, 13], contour_position=position, img_name='map1')
        locations_2 = self._hsv_contour(
            imgpath, (40, 100, 0), (150, 255, 255), 200, 300, 0, -10,
            kernal=[11, 11], contour_position=position, img_name='map2')
        print('location1', locations_1, 'location2', locations_2)
        if len(locations_1) and len(locations_2):
            return np.concatenate((locations_1, locations_2))
        return locations_1 if len(locations_1) else locations_2

    def debug_img(self, img_name, img, save_to="files/debug/"):
        if self.debug:
            filename = "{0}.png".format(
                img_name)
            img_path = os.path.join(save_to, filename)
            print(f'Img save to {img_path}')
            cv2.imwrite(img_path, img)

    def _debug_img_with_text(self, locations, img):
        for idx, location in enumerate(locations):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, '%s' % (
                idx + 1), (location[0], location[1]),
                font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def _feature_match(self, large_img, small_img, min_match_nums=15,
                       img_name=''):
        train = cv2.cvtColor(large_img, cv2.COLOR_BGR2GRAY)
        query = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        try:
            query = small_img
            kp1, des1 = sift.detectAndCompute(query, None)
            kp2, des2 = sift.detectAndCompute(train, None)
            if kp1 == () or kp2 == ():
                return [], 0
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            good = []
            ratio = 0.7
            for m, n in matches:
                if m.distance < ratio * n.distance:
                    good.append([m])

            if img_name:
                img_match = np.empty(
                    (large_img.shape[0] + 100,
                     query.shape[1] + large_img.shape[1], 3), dtype=np.uint8)
                img_debug = cv2.drawMatchesKnn(
                    query, kp1, train, kp2, good, flags=2, outImg=img_match)
                self.debug_img(img_name, img_debug)
            # re-group the pts .
            pts_groups = []
            pts = np.array([kp2[m[0].trainIdx].pt for m in good]).astype(np.float32)
            qh, qw = query.shape[:2]
            for pt in pts:
                in_group = False
                for pts_group in pts_groups:
                    if pt in pts_group:
                        in_group = True
                        break
                if not in_group:
                    approx_pts_rows = np.where(
                        (pts[:, 1] >= pt[1] - qh) &
                        (pts[:, 1] <= pt[1] + qh) &
                        (pts[:, 0] >= pt[0] - qw) & (pts[:, 0] <= pt[0] + qw))
                    approx_pts = pts[approx_pts_rows]
                    pts_groups.append(approx_pts)
            max_number_group = []
            for pts_group in pts_groups:
                if len(pts_group) > len(max_number_group):
                    max_number_group = pts_group
            if len(max_number_group) < min_match_nums:
                return [], len(max_number_group)
            else:
                logging.debug(" good %s for action %s" ,
                              len(max_number_group), 'action')
            x, y = np.mean(max_number_group, axis=0)
            return np.array([int(x), int(y)]), len(max_number_group)
        except Exception as e:
            print(e)
            return [], 0

    def _find_object(self, img_path, target_name, min_match_num=15):
        large_img = cv2.imread(img_path)
        small_img = cv2.imread(
            CURRENT_PATH + f'files/iphone11pm/{target_name}.jpg')
        result = self._feature_match(
            large_img, small_img, min_match_nums=min_match_num,
            img_name=target_name)
        return result[0] if len(result[0])>0 else result[1]

    def find_bounties(self, img_path):
        return self._find_object(img_path, 'find_bounties', min_match_num=40)

    def find_team(self, img_path):
        return self._find_object(img_path, 'find_team', min_match_num=20)

    def find_map_next(self, img_path):
        result = self._find_object(img_path, 'find_map')
        if isinstance(result, int):
            return result
        res = self._find_object(img_path, 'find_map_camp', min_match_num=10)
        if not isinstance(res, int):
            return [0, 0]
        res = self._find_object(img_path, 'find_map_play')
        if not isinstance(res, int):
            return res.tolist()
        location_angel = self._find_object(img_path, 'find_angel')
        # print(location_angel)
        location_minion = self.list_map_moves(img_path)
        # print(location_minion)
        if not isinstance(location_angel, int):
            distance = [loc for loc in location_minion
                        if np.linalg.norm(location_angel-loc) < 80]
            if distance:
                return distance[0].tolist()
        return random.choice(location_minion).tolist() if len(location_minion) else 0

    def find_treasure(self, img_path):
        res = self._find_object(img_path, 'find_treasure')
        return res if not isinstance(res, int) else self._find_object(
            img_path, 'find_treasure_replace')

    def find_chest(self, img_path):
        return self._find_object(img_path, 'find_chest', min_match_num=8)

    def find_complete(self, img_path):
        return self._find_object(img_path, 'find_complete', min_match_num=10)

    def find_battle(self, img_path):
        res = self._find_object(img_path, 'find_battle', min_match_num=8)
        if isinstance(res, int):
            return res
        res = self._find_object(img_path, 'find_battle_ready', 10)
        if not isinstance(res, int):
            print('battle ready, go')
            return [0, 0]
        location_spell = self.list_card_spells(img_path)
        print('spell', location_spell)
        location_enemy, location_minion = self.list_enemy_and_minion(img_path)
        print(location_enemy)
        if len(location_enemy) > 0:
            enemy_idx = int(round(random.random()-1,0))
            enemy_location = location_enemy[enemy_idx].tolist()
            print('enemy', enemy_location)
            if len(location_spell):
                spell_idx = min(len(location_spell)-1, 0)
                return [location_spell[spell_idx].tolist(),
                        enemy_location]
            elif len(location_minion) > 0:
                minion_location = random.choice(location_minion).tolist()
                print('minion', minion_location)
                return minion_location
            else:
                print('minion not ready')
                return [0, 0]
        return 0
    def find_victory(self, img_path):
        return self._find_object(img_path, 'find_victory')

if __name__ == '__main__':
    hcm = HSContonurMatch(debug=True)
    # print(CURRENT_PATH)
    # print(hcm.find_bounties(CURRENT_PATH + 'files/iphone11pm/bounties.jpg'))
    # print(hcm.find_team(CURRENT_PATH + 'files/iphone11pm/team.jpg'))
    # print(hcm.find_map_next(CURRENT_PATH + 'files/debug/idle_124.png'))
    # print(hcm.find_map_next(CURRENT_PATH + 'files/debug/play_map_12.png'))
    print(hcm.find_battle(CURRENT_PATH + 'files/debug/paly_battle_134.png'))
    # print(hcm.find_battle(CURRENT_PATH + 'ios_game.png'))
    # print(hcm._find_object(CURRENT_PATH + 'files/debug/paly_battle_2.png','find_battle_ready', ))
    # print(hcm.find_chest(CURRENT_PATH + 'files/iphone11pm/chest.jpg'))
    # print(hcm.find_complete(CURRENT_PATH + 'files/debug/idle_34.png'))
    # print(hcm.find_battle(CURRENT_PATH + 'files/debug/paly_battle_10.png'))
    # print(hcm.find_battle(CURRENT_PATH + 'files/iphone11pm/battle1.jpg'))
    # print(hcm.find_battle(CURRENT_PATH + 'files/iphone11pm/battle2.jpg'))
    # print(hcm.find_battle(CURRENT_PATH + 'files/iphone11pm/battle3.jpg'))
    # print(hcm.find_victory(CURRENT_PATH + 'files/iphone11pm/victory.jpg'))
