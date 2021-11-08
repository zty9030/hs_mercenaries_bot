from ahk import AHK

import pyautogui

class CONSTANT:
    GAME_NAME = "Hearthstone"

#magic numbers for the different resolutions
class r19201080:
    debug = False
    # pickup treasure and vistor margin
    treasure_x_margin = 300
    treasure_y_margin = 200

    # allow move card margin
    allow_move_card_x_margin = 20
    allow_move_card_y_margin = -2

    # _allow_spell_cards margin
    allow_spell_card_x_margin = -40
    allow_spell_card_y_margin = 60
    
    #battle drag
    battle_drag_x_margin = 160


    left_black_x_margin =70
    bottom_black_y_margin = 200

    #map scoll up 
    up_y_margin = 100

    path="1920x1080"

from datetime import datetime
class HSSetting:
  
    def __init__(self,  resolution):
        self.resolution = resolution
        self.possibility = 0.6
        self.screenshot_id = 1
        self.ahk = AHK()
        self.win =self.ahk.win_get(title=CONSTANT.GAME_NAME)
        self.bring_game()
    
    def screenshot(self):
        #filename = "{0}.png".format( datetime.now().strftime("%Y%m%d%H%M%S"))
        id = int(self.screenshot_id % 50)
        filename = "screenshot_%s.png" % id
        imgpath = 'files/debug/' + filename
        pyautogui.screenshot(imgpath)
        self.debug_msg("Screenshot %s" % (id))
        self.screenshot_id +=1
        return imgpath

    def bring_game(self):
        self.win.show()
        self.win.restore()
        self.win.maximize()
        self.win.to_top()      
        self.win.activate()
    
    def debug_msg(self,msg,ci=None):
        if ci == None:
            print("%s - [None]: %s" % (datetime.now().strftime("%Y%m%d-%H:%M:%S") , msg))
        else:
            print("%s - [%s] : %s" % (datetime.now().strftime("%Y%m%d-%H:%M:%S"),ci.__class__.__name__,msg))



