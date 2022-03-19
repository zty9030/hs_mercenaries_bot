from hs_mercenaries_bot import ios_operation, search, action
import sys, select
import cv2

def game(img, connection):
    hsgame = search.HSContonurMatch(debug=False)
    hsaction = action.Action(connection)
    if not isinstance(hsgame.find_bounties(img), int):
        hsaction.action_bounties()
        return 'play_bounties'
    if not isinstance(hsgame.find_team(img), int):
        hsaction.action_team()
        return 'play_team'
    location = hsgame.find_map_next(img)
    if not isinstance(location, int):
        hsaction.action_map(location)
        return 'play_map'
    location = hsgame.find_battle(img)
    if not isinstance(location, int):
        hsaction.action_battle(location)
        return 'paly_battle'
    if not isinstance(hsgame.find_treasure(img), int):
        hsaction.action_treassure()
        return 'play_treassure'
    if not isinstance(hsgame.find_chest(img), int):
        hsaction.action_chest()
        return 'play_chest'
    if not isinstance(hsgame.find_complete(img), int):
        hsaction.action_complete()
        return 'play_complete'
    hsaction.action_idle()
    return 'idle'

def auto_process(step_count=-1, round_count=1):
    img_path = 'ios_game.png'
    ios_connect = ios_operation.IOS_Action()
    step = 0
    while True:
        ios_connect.get_screenshot(img_path)
        res = game(img_path, ios_connect)
        print(res)
        step+=1
        if step_count>0 and step>=step_count:
            break
        
        if is_not_correct():
            img = cv2.imread(img_path)
            file_name = f'files/debug/{res}_{step}.png'
            print(file_name)
            cv2.imwrite(file_name, img)
        

def is_not_correct():
    timeout = 2
    print (f"This action correct?")
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        sys.stdin.readline()
    # print('stdin', sys.stdin, 'i', i)
    return bool(i)

def test_correct():
    while True:
        print(is_not_correct())

if __name__=='__main__':
    # test_correct()
    # step_process()
    auto_process(step_count=-1)