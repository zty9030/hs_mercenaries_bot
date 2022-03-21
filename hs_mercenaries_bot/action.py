# import ios_operation
import time
import random


class Action:
    def __init__(self, action) -> None:
        self.action = action

    def _action_sleep(self, *args, min_wait_sec=0.3):
        # print(args)
        for idx, li in enumerate(args):
            sleep_time = round(random.random() * 0.2 + min_wait_sec, 2)
            if isinstance(li[0], int):
                self.action.phone_action(li)
            else:
                self.action.phone_action(*li)
            if idx+1 < len(args):
                time.sleep(sleep_time)

    def action_bounties(self):
        # select, comfirm
        self._action_sleep([230, 140], [710, 350])

    def action_team(self):
        # select, comfirm, pop
        self._action_sleep([240, 100], [715, 355], [370, 260])

    def action_map(self, location):
        # select ,confirm
        print('click', location)
        self._action_sleep(location, [715, 355], min_wait_sec=0.5)

    def action_battle(self, location_pair):
        if location_pair == [0, 0]:
            self._action_sleep([725, 200])
        else:
            self._action_sleep(
                location_pair[0], location_pair, min_wait_sec=2)

    def action_treassure(self):
        # select, confirm
        self._action_sleep([380, 210], [540, 370])

    def action_chest(self):
        # chest 1,2,3, confirm
        self._action_sleep(
            [465, 100], [335, 280], [600, 290], [470, 200], min_wait_sec=0.2)

    def action_complete(self):
        self._action_sleep([440, 345])

    def action_idle(self):
        self._action_sleep([230, 170])
