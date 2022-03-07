from http import client
import time
import wda
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

class IOS_Action:
    def __init__(self, address='') -> None:
        if address:
            self.client = wda.Client(address)
        else:
            self.client = wda.USBClient()
        self.session = self.client.session()
        self.height = self.session.window_size().height
        self.width = self.session.window_size().width
        print(self.height, self.width)
    
    def get_screenshot(self):
        img = self.client.screenshot()
        img = img.resize([self.width, self.height])
        img.save('ios_screen.png')
        
    def phone_action(self, *args):
        act_second = round(random.random() * 0.4 + 0.1, 2)
        if len(args)==1:
            coord = args[0]
            print('Do click', [coord[0], coord[1]])
            self.session.click(coord[0], coord[1], act_second)
        else:
            coord1, coord2 = args[0], args[1]
            print('Do swipe from',[coord1[0],coord1[1]], 'to', [coord2[0], coord2[1]])
            self.session.swipe(coord1[0],coord1[1], coord2[0], coord2[1], act_second)
            
def mannual_mode():
    # ios_act = IOS_Action('http://192.168.1.5:8100')
    ios_act = IOS_Action()


    def update_data():
        img = Image.open('ios_screen.png')
        return np.array(img)
    
    def updatefig(*args):
        global cor
        global time_period
        global click_count
        
        if len(cor)>1:
            ios_act.phone_action(*cor)
            cor = []
        now = time.perf_counter()
        if len(cor)>1 or (time_period and (now - time_period) > 1):
            try:
                ios_act.phone_action(*cor)
            except:
                pass
            cor = []
            time_period = 0
            click_count = 0
        
        ios_act.get_screenshot()
        im.set_array(update_data())
        # time.sleep(3)
        return im,


    def on_click(event):
        global click_count
        global cor
        global time_period

        coords = [int(event.xdata), int(event.ydata)]
        # coords = [ix, iy]
        print('click = ', coords)
        cor.append(coords)
        time_period = time.perf_counter()
        click_count += 1
        
    # click_count = 0
    # cor = []
    # time_period = 0
    fig = plt.figure()
    ios_act.get_screenshot()
    img = update_data()
    im = plt.imshow(img, animated=True)

    fig.canvas.mpl_connect('button_press_event', on_click)
    _ = animation.FuncAnimation(fig, updatefig, interval=500, blit=True)
    plt.show()

if __name__=='__main__':
    click_count = 0
    cor = []
    time_period = 0
    mannual_mode()