from hs_mercenaries_bot import ios_operation, search, action
import sys, select
import cv2

def game(img, connection):
    hsgame = search.HSContonurMatch(debug=False)
    hsaction = action.Action(connection)
    if not isinstance(hsgame.find_bounties(img), int):
        return hsaction.action_bounties()
    if not isinstance(hsgame.find_team(img), int):
        return hsaction.action_team()
    location = hsgame.find_map_next(img)
    if not isinstance(location, int):
        return hsaction.action_map(location)
    location = hsgame.find_battle(img)
    if not isinstance(location, int):
        return hsaction.action_battle(location)
    if not isinstance(hsgame.find_treasure(img), int):
        return hsaction.action_treassure()
    if not isinstance(hsgame.find_chest(img), int):
        return hsaction.action_chest()
    if not isinstance(hsgame.find_complete(img), int):
        return hsaction.action_complete()
    
    hsaction.action_idle()

def auto_process():
    img_path = 'ios_game.png'
    ios_connect = ios_operation.IOS_Action()
    # hsgame = search.HSContonurMatch()
    while True:
        ios_connect.get_screenshot(img_path)
        game(img_path, ios_connect)
        img = cv2.imread(img_path)
        cv2.imshow('imshow', img)
        cv2.waitKey(1)
        

def step_process():
    img_path = 'ios_game.png'
    ios_connect = ios_operation.IOS_Action()
    ios_connect.get_screenshot(img_path)
    game(img_path, ios_connect)
# for _ in range(3):
#     timeout = 3
#     print (f"You have {timeout} seconds to answer!")
#     i, o, e = select.select([sys.stdin], [], [], timeout)
#     if (i):
#         print("You said", sys.stdin.readline().strip())
#     else:
#         print("You said nothing!")
  
if __name__=='__main__':
    # step_process()
    auto_process()