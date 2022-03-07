import pyautogui

class AHK:
    @staticmethod
    def mouse_move(x, y, speed=0):
        pyautogui.moveTo(x, y, duration=speed)
        pass
    
    @staticmethod
    def mouse_drag(x, y, speed=0):
        pyautogui.dragTo(x, y, duration=speed)
        pass
    
    @staticmethod
    def click():
        pyautogui.click()
        pass
    
    @staticmethod
    def right_click():
        pyautogui.rightClick()
        pass

class win:
    height=1920
    weight=1080