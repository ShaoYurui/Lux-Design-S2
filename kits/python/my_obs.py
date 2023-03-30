import sys

import numpy as np


class MyObs:
    def __init__(self, game_state, player):
        factories = game_state.factories[player]
        units = game_state.units[player]
        self.game_state = game_state
        self.move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
        self.ice_map = game_state.board.ice
        self.ice_tile_locations = np.argwhere(self.ice_map == 1)
        self.ore_map = game_state.board.ore
        self.ore_tile_locations = np.argwhere(self.ore_map == 1)
        self.moving_unit = []
        self.next_occupy_tiles = np.array([])

        self.heavy_bot_tiles = []
        self.heavy_bot_units = []
        for unit_id, unit in units.items():
            if unit.unit_type == "HEAVY":
                self.heavy_bot_tiles += [unit.pos]
                self.heavy_bot_units += [unit]
        self.heavy_bot_tiles = np.array(self.heavy_bot_tiles)

        self.light_bot_tiles = []
        self.light_bot_units = []
        for unit_id, unit in units.items():
            if unit.unit_type == "LIGHT":
                self.light_bot_tiles += [unit.pos]
                self.light_bot_units += [unit]
        self.light_bot_tiles = np.array(self.light_bot_tiles)

        self.factory_tiles = []
        self.factory_units = []
        for unit_id, unit in factories.items():
            self.factory_tiles += [unit.pos]
            self.factory_units += [unit]
        self.factory_tiles = np.array(self.factory_tiles)

    def is_at_factory(self, pos):
        if len(self.factory_tiles) > 0:
            factory_distances = np.mean((self.factory_tiles - pos) ** 2, 1)
            closest_factory_tile = self.factory_tiles[np.argmin(factory_distances)]
            closest_factory = self.factory_units[np.argmin(factory_distances)]
            return np.mean((closest_factory_tile - pos) ** 2) == 0
        return False

    def get_closest_factory_tile(self, pos):
        if len(self.factory_tiles) > 0:
            factory_distances = np.mean((self.factory_tiles - pos) ** 2, 1)
            closest_factory_tile = self.factory_tiles[np.argmin(factory_distances)]
            return closest_factory_tile
        return None

    def get_closest_ice_tile(self, pos):
        ice_tile_distances = np.mean((self.ice_tile_locations - pos) ** 2, 1)
        closest_ice_tile = self.ice_tile_locations[np.argmin(ice_tile_distances)]
        return closest_ice_tile

    def is_adjacent_to_light_bot(self, pos):
        if len(self.light_bot_tiles) > 0:
            light_bot_distances = np.mean((self.light_bot_tiles - pos) ** 2, 1)
            light_bot_distances[light_bot_distances == 0] = np.inf
            closest_light_bot_tile = self.light_bot_tiles[np.argmin(light_bot_distances)]
            return np.mean((closest_light_bot_tile - pos) ** 2) == 0.5
        return False

    def get_closest_light_bot(self, pos):
        if len(self.light_bot_tiles) > 0:
            light_bot_distances = np.mean((self.light_bot_tiles - pos) ** 2, 1)
            light_bot_distances[light_bot_distances == 0] = np.inf
            closest_light_bot = self.light_bot_units[np.argmin(light_bot_distances)]
            return closest_light_bot
        return None

    def get_closest_light_bot_tile(self, pos):
        if len(self.light_bot_tiles) > 0:
            light_bot_distances = np.mean((self.light_bot_tiles - pos) ** 2, 1)
            light_bot_distances[light_bot_distances == 0] = np.inf
            closest_light_bot_tile = self.light_bot_tiles[np.argmin(light_bot_distances)]
            return closest_light_bot_tile
        return None

    def is_adjacent_to_heavy_bot(self, pos):
        if len(self.heavy_bot_tiles) > 0:
            heavy_bot_distances = np.mean((self.heavy_bot_tiles - pos) ** 2, 1)
            heavy_bot_distances[heavy_bot_distances == 0] = np.inf
            closest_heavy_bot_tile = self.heavy_bot_tiles[np.argmin(heavy_bot_distances)]
            return np.mean((closest_heavy_bot_tile - pos) ** 2) == 0.5
        return False

    def get_closest_heavy_bot(self, pos):
        if len(self.heavy_bot_tiles) > 0:
            heavy_bot_distances = np.mean((self.heavy_bot_tiles - pos) ** 2, 1)
            heavy_bot_distances[heavy_bot_distances == 0] = np.inf
            closest_heavy_bot = self.heavy_bot_units[np.argmin(heavy_bot_distances)]
            return closest_heavy_bot
        return None

    def get_closest_heavy_bot_tile(self, pos):
        if len(self.heavy_bot_tiles) > 0:
            heavy_bot_distances = np.mean((self.heavy_bot_tiles - pos) ** 2, 1)
            heavy_bot_distances[heavy_bot_distances == 0] = np.inf
            closest_heavy_bot_tile = self.heavy_bot_tiles[np.argmin(heavy_bot_distances)]
            return closest_heavy_bot_tile
        return None

    def is_occupied(self, pos):
        for tiles in self.heavy_bot_tiles:
            if np.all(pos == tiles):
                return True
        for tiles in self.light_bot_tiles:
            if np.all(pos == tiles):
                return True
        for tiles in self.next_occupy_tiles:
            if np.all(pos == tiles):
                return True
        return False

    def get_best_directions(self, given_dir, pos):
        possible_dir = []
        if given_dir == 1:
            possible_dir = [1, 2, 4]
        if given_dir == 2:
            possible_dir = [2, 1, 3]
        if given_dir == 3:
            possible_dir = [3, 2, 4]
        if given_dir == 4:
            possible_dir = [4, 1, 3]

        free_dir = []
        rubble = []
        for dir in possible_dir:
            next_pos = pos + self.move_deltas[dir]
            if not self.is_occupied(next_pos):
                free_dir.append(dir)
                rubble.append(self.game_state.board.rubble[next_pos[0]][next_pos[1]])

        return free_dir[np.argmin(rubble)]

    def unit_moving_to(self, unit, given_dir):
        next_pos = unit.pos + self.move_deltas[given_dir]
        self.next_occupy_tiles = np.append(self.next_occupy_tiles, next_pos)
        self.moving_unit.append(unit)

    def is_unit_moving(self, unit):
        if unit in self.moving_unit:
            return True
        return False
