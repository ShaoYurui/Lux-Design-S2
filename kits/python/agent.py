from my_obs import MyObs
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
import numpy as np
import sys


class Agent():
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            # bid 0 to not waste resources bidding and declare as the default faction
            return dict(faction="AlphaStrike", bid=0)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            # factory placement period

            # how much water and metal you have in your starting pool to give to new factories
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            # how many factories you have left to place
            factories_to_place = game_state.teams[self.player].factories_to_place
            # whether it is your turn to place a factory
            my_turn_to_place = my_turn_to_place_factory(game_state.teams[self.player].place_first, step)
            if factories_to_place > 0 and my_turn_to_place:
                # we will spawn our factory in a random location with 150 metal and water if it is our turn to place
                potential_spawns = np.array(list(zip(*np.where(obs["board"]["valid_spawns_mask"] == 1))))
                spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
                return dict(spawn=spawn_loc, metal=150, water=150)
            return dict()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs)

        my_obs = MyObs(game_state, self.player)

        factory_action = self.get_factory_action(game_state, my_obs)
        heavyBot_action = self.get_heavyBot_action(game_state, my_obs)
        lightBot_action = self.get_lightBot_action(game_state, my_obs)

        actions = merge(factory_action, heavyBot_action, lightBot_action)

        return actions

    def get_factory_action(self, game_state, my_obs):
        actions = dict()
        for factory in my_obs.factory_units:
            if factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                    factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST and \
                    len(my_obs.heavy_bot_units) == 0:
                actions[factory.unit_id] = factory.build_heavy()
            if factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                    factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST and \
                    not my_obs.is_occupied(factory.pos) and \
                    len(my_obs.heavy_bot_units) != 0 and len(my_obs.light_bot_units) == 0:
                actions[factory.unit_id] = factory.build_light()

            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[factory.unit_id] = factory.water()

        return actions

    def get_heavyBot_action(self, game_state, my_obs):
        actions = dict()
        for heavyBot in my_obs.heavy_bot_units:
            # print(heavyBot.action_queue, file=sys.stderr)

            if my_obs.is_at_factory(heavyBot.pos) and heavyBot.power <= 1400:
                actions[heavyBot.unit_id] = [heavyBot.pickup(4, 1500 - heavyBot.power, repeat=0, n=1)]
                continue

            # If a heavy robot is near a light robot, then he transfers the ice
            if heavyBot.cargo.ice != 0 and \
                    my_obs.is_adjacent_to_light_bot(heavyBot.pos) and \
                    my_obs.get_closest_light_bot(heavyBot.pos).cargo.ice == 0:
                trans_amount = 100
                if heavyBot.cargo.ice < 100:
                    trans_amount = heavyBot.cargo.ice
                direction = direction_to(heavyBot.pos, my_obs.get_closest_light_bot_tile(heavyBot.pos))
                actions[heavyBot.unit_id] = [heavyBot.transfer(direction, 0, trans_amount, repeat=0, n=1)]
                continue

            if np.all(heavyBot.pos == my_obs.get_closest_ice_tile(heavyBot.pos)):
                if heavyBot.power >= heavyBot.dig_cost(game_state) + heavyBot.action_queue_cost(game_state):
                    actions[heavyBot.unit_id] = [heavyBot.dig(repeat=0, n=1)]
                continue

            if game_state.board.rubble[heavyBot.pos[0]][heavyBot.pos[1]] != 0:
                if heavyBot.power >= heavyBot.dig_cost(game_state) + heavyBot.action_queue_cost(game_state):
                    actions[heavyBot.unit_id] = [heavyBot.dig(repeat=0, n=1)]
                continue

            else:
                direction = direction_to(heavyBot.pos, my_obs.get_closest_ice_tile(heavyBot.pos))
                move_cost = heavyBot.move_cost(game_state, direction)
                if direction != 0 and \
                        move_cost is not None and heavyBot.power >= move_cost + heavyBot.action_queue_cost(game_state):
                    actions[heavyBot.unit_id] = [heavyBot.move(direction, repeat=0, n=1)]
                    my_obs.unit_moving_to(heavyBot, direction)
        return actions

    def get_lightBot_action(self, game_state, my_obs):
        actions = dict()
        for lightBot in my_obs.light_bot_units:
            # if lightBot.unit_id == "unit_17":
            #     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", file=sys.stderr)
            #     print(lightBot.pos, file=sys.stderr)
            #     print(my_obs.is_adjacent_to_light_bot(lightBot.pos), file=sys.stderr)
            #     print(my_obs.get_closest_light_bot_tile(lightBot.pos), file=sys.stderr)
            #     print(my_obs.light_bot_tiles, file=sys.stderr)
            if my_obs.is_at_factory(lightBot.pos) and lightBot.power <= 140:
                actions[lightBot.unit_id] = [lightBot.pickup(4, 150 - lightBot.power, repeat=0, n=1)]
                continue

            if not my_obs.is_adjacent_to_heavy_bot(lightBot.pos) and lightBot.cargo.ice == 0 and lightBot.power > 40:
                direction = direction_to(lightBot.pos, my_obs.get_closest_ice_tile(lightBot.pos))
                direction = my_obs.get_best_directions(direction, lightBot.pos)
                move_cost = lightBot.move_cost(game_state, direction)
                if direction != 0 and \
                        move_cost is not None and lightBot.power >= move_cost + lightBot.action_queue_cost(game_state):
                    actions[lightBot.unit_id] = [lightBot.move(direction, repeat=0, n=1)]
                    my_obs.unit_moving_to(lightBot, direction)
                    continue

            # if my_obs.is_adjacent_to_light_bot(lightBot.pos) and lightBot.power > 40:
            #     if lightBot.power > my_obs.get_closest_light_bot(lightBot.pos).power:
            #         direction = direction_to(lightBot.pos, my_obs.get_closest_light_bot_tile(lightBot.pos))
            #         actions[lightBot.unit_id] = [lightBot.transfer(direction, 4, 50, repeat=0, n=1)]
            #     continue
            #
            # if my_obs.is_adjacent_to_light_bot(lightBot.pos) and lightBot.power <= 40:
            #     # direction = direction_to(lightBot.pos, my_obs.get_closest_light_bot_tile(lightBot.pos))
            #     # actions[lightBot.unit_id] = [lightBot.transfer(direction, 0, lightBot.cargo.ice, repeat=0, n=1)]
            #     continue

            if my_obs.is_adjacent_to_heavy_bot(lightBot.pos) and lightBot.power > 40:
                if not my_obs.is_unit_moving(my_obs.get_closest_heavy_bot(lightBot.pos)):
                    direction = direction_to(lightBot.pos, my_obs.get_closest_heavy_bot_tile(lightBot.pos))
                    actions[lightBot.unit_id] = [lightBot.transfer(direction, 4, lightBot.power - 40, repeat=0, n=1)]
                continue

            if my_obs.is_at_factory(lightBot.pos) and lightBot.cargo.ice != 0:
                direction = direction_to(lightBot.pos, my_obs.get_closest_factory_tile(lightBot.pos))
                actions[lightBot.unit_id] = [lightBot.transfer(direction, 0, lightBot.cargo.ice, repeat=0, n=1)]
                continue

            if (not my_obs.is_at_factory(lightBot.pos) and lightBot.cargo.ice != 0) or lightBot.power <= 40:
                direction = direction_to(lightBot.pos, my_obs.get_closest_factory_tile(lightBot.pos))
                direction = my_obs.get_best_directions(direction, lightBot.pos)
                move_cost = lightBot.move_cost(game_state, direction)

                if direction != 0 and \
                        move_cost is not None and lightBot.power >= move_cost + lightBot.action_queue_cost(game_state):
                    actions[lightBot.unit_id] = [lightBot.move(direction, repeat=0, n=1)]
                    my_obs.unit_moving_to(lightBot, direction)
                continue
        return actions


def merge(dict1, dict2, dict3):
    res = {**dict1, **dict2, **dict3}
    return res
