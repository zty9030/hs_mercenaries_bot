from hs_mercenaries_bot import ios_operation, search, action
import sys, select
import cv2

def auto_process():
    ios_connect = ios_operation.IOS_Action()
    while True:
        ios_connect.get_screenshot()
        img = cv2.imread('ios_screen.png')
        cv2.imshow('imshow', img)
        cv2.waitKey(1)


# for _ in range(3):
#     timeout = 3
#     print (f"You have {timeout} seconds to answer!")
#     i, o, e = select.select([sys.stdin], [], [], timeout)
#     if (i):
#         print("You said", sys.stdin.readline().strip())
#     else:
#         print("You said nothing!")
  
if __name__=='__main__':
    auto_process()