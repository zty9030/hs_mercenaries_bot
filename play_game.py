""""""
import time
import argparse
import sys
import select
import random
import cv2
from hs_mercenaries_bot import ios_operation, search, action


def game(img, hsaction, spell_idx):
    hsgame = search.HSContonurMatch(debug=False)
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
    location = hsgame.find_battle(img, spell_idx)
    if not isinstance(location, int):
        hsaction.action_battle(location)
        return 'paly_battle'
    location = hsgame.find_victory(img)
    if not isinstance(location, int):
        hsaction.action_idle()
        return 'paly_victory'
    if not isinstance(hsgame.find_treasure(img), int):
        hsaction.action_treassure()
        return 'play_treassure'
    if not isinstance(hsgame.find_chest(img), int):
        hsaction.action_chest()
        return 'play_chest'
    if not isinstance(hsgame.find_complete(img), int):
        hsaction.action_complete()
        return 'play_complete'
    return 'idle'


def auto_process(total_round=20):
    img_path = 'ios_game.png'
    ios_connect = ios_operation.IOS_Action()
    hsaction = action.Action(ios_connect)
    idle_count = 0
    step = 0
    start_time = time.perf_counter()
    spell_idx  = 1
    pause = False
    while True:
        if not pause:
            ios_connect.get_screenshot(img_path)
            res = game(img_path, hsaction, spell_idx)
            print(res)
            if res == 'idle':
                idle_count += 1
                if idle_count % 5 == 0:
                    hsaction.action_idle()
                if idle_count > 20:
                    total_round -= 1
            else:
                idle_count = 0
            if res == 'play_complete':
                total_round -= 1
                end_time = time.perf_counter()
                print('Round finish', total_round, 'remain')
                print('Round take', round(end_time-start_time,3))
                time.sleep(2)
                start_time = end_time
            if total_round == 0:
                save_screen(img_path, 'exit', step)
                ios_connect.client.home()
                break
            step += 1
        input_str = wait_for_input()
        if input_str in ['1','2','3']:
            print('now try to cast spell', input_str)
            spell_idx = input_str
        elif input_str == 'p':
            print('pause')
            pause = True
        elif input_str == 'r':
            pause = False
        elif input_str == 'f':
            print('stop after this round')
            total_round = 1
        elif input_str == 'q':
            break
        elif input_str:
            save_screen(img_path, res, step)


def save_screen(img_path, res, step):
    img = cv2.imread(img_path)
    file_name = f'files/debug/{res}_{step}.png'
    print(file_name)
    cv2.imwrite(file_name, img)


def wait_for_input():
    timeout = round(1.8 + random.random() * 0.4, 2)
    print("This action correct?")
    i, _, _ = select.select([sys.stdin], [], [], timeout)
    if i:
        input_str = sys.stdin.readline().strip()
    #     if input_str.strip() == 'q':
    #         sys.exit(0)
    # # print('stdin', sys.stdin, 'i', i)
    return input_str if bool(i) else ''


def test_correct():
    while True:
        print(wait_for_input())


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-r', dest='round', type=int, default=20)
    args = parse.parse_args()
    # test_correct()
    # step_process()
    auto_process(total_round=args.round)
