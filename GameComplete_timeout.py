import random
import os
import re
import turtle
import time
import matplotlib.pyplot as plt
import networkx
import seaborn as sns
import numpy as np
import pandas as pd
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from gym.spaces import Discrete
from scipy.ndimage import gaussian_filter1d
import scipy.stats as st
import warnings
warnings.filterwarnings("ignore")

visualization = False
time_limit_minutes = 1
colors_sequence = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
Q_table_plot = False

def get_action(action, last_stateX, last_stateY):
    if action == 0:  # stop
        new_stateX = last_stateX
        new_stateY = last_stateY
        actionX = 0
        actionY = 0
    elif action == 1:  # right
        sub_action = 1
        if 0 <= sub_action + last_stateX < cols:
            new_stateX = sub_action + last_stateX
            actionX = sub_action
        else:
            new_stateX = last_stateX
            actionX = 0
            action = 0
        new_stateY = last_stateY
        actionY = 0
    elif action == 2:  # left
        sub_action = -1
        if 0 <= sub_action + last_stateX < cols:
            new_stateX = sub_action + last_stateX
            actionX = sub_action
        else:
            new_stateX = last_stateX
            actionX = 0
            action = 0
        new_stateY = last_stateY
        actionY = 0
    elif action == 3:  # up
        new_stateX = last_stateX
        actionX = 0
        sub_action = 1
        if 0 <= sub_action + last_stateY < rows:
            new_stateY = sub_action + last_stateY
            actionY = sub_action
        else:
            new_stateY = last_stateY
            actionY = 0
            action = 0
    elif action == 4:  # down
        new_stateX = last_stateX
        actionX = 0
        sub_action = -1
        if 0 <= sub_action + last_stateY < rows:
            new_stateY = sub_action + last_stateY
            actionY = sub_action
        else:
            new_stateY = last_stateY
            actionY = 0
            action = 0
    """elif action == 5:  # diag_up_right
        if (0 <= 1 + last_stateY < rows) and (0 <= 1 + last_stateX < cols):
            new_stateY = 1 + last_stateY
            actionY = 1
            new_stateX = 1 + last_stateX
            actionX = 1
        else:
            new_stateY = last_stateY
            actionY = 0
            new_stateX = last_stateX
            actionX = 0
            action = 0
    elif action == 6:  # diag_down_right
        if (0 <= -1 + last_stateY < rows) and (0 <= 1 + last_stateX < cols):
            new_stateY = -1 + last_stateY
            actionY = -1
            new_stateX = 1 + last_stateX
            actionX = 1
        else:
            new_stateY = last_stateY
            actionY = 0
            new_stateX = last_stateX
            actionX = 0
            action = 0
    elif action == 7:  # diag_up_left
        if (0 <= 1 + last_stateY < rows) and (0 <= -1 + last_stateX < cols):
            new_stateY = 1 + last_stateY
            actionY = 1
            new_stateX = -1 + last_stateX
            actionX = -1
        else:
            new_stateY = last_stateY
            actionY = 0
            new_stateX = last_stateX
            actionX = 0
            action = 0
    elif action == 8:  # diag_down_left
        if (0 <= -1 + last_stateY < rows) and (0 <= -1 + last_stateX < cols):
            new_stateY = -1 + last_stateY
            actionY = -1
            new_stateX = -1 + last_stateX
            actionX = -1
        else:
            new_stateY = last_stateY
            actionY = 0
            new_stateX = last_stateX
            actionX = 0
            action = 0"""

    return new_stateX, new_stateY, action, actionX, actionY


def get_direction(x_ag, y_ag, x_en, y_en):
    deltaX = x_en - x_ag
    deltaY = y_en - y_ag

    direction_ag_en = -1

    if deltaX == 0 and deltaY == 0: # stop
        direction_ag_en = 0
    elif deltaX == 1 and deltaY == 0: # right
        direction_ag_en = 1
    elif deltaX == -1 and deltaY == 0: # left
        direction_ag_en = 2
    elif deltaX == 0 and deltaY == 1: # up
        direction_ag_en = 3
    elif deltaX == 0 and deltaY == -1: # down
        direction_ag_en = 4
    elif deltaX == 1 and deltaY == 1 and n_act_agents > 5: # diag up right
        direction_ag_en = 5
    elif deltaX == 1 and deltaY == -1 and n_act_agents > 5:  # diag down right
        direction_ag_en = 6
    elif deltaX == -1 and deltaY == 1 and n_act_agents > 5:  # diag up left
        direction_ag_en = 7
    elif deltaX == -1 and deltaY == -1 and n_act_agents > 5:  # diag down left
        direction_ag_en = 8
    else: # otherwise
        direction_ag_en = 50

    # print([x_ag, y_ag], [x_en, y_en], direction_ag_en)

    return direction_ag_en


if visualization:
    wn = turtle.Screen()
    wn.bgcolor('black')
    wn.title('Game')
    # wn.setup(700, 700)
    speed_screen = 1
    #wn.tracer(0)

    class_enemies = []
    class_agents = []
    class_goals = []
    images = ['agent.gif', 'treasure.gif',
              'wall.gif', 'enemy.gif']
    # register shapes 24x24
    for image in images:
        turtle.register_shape(image)


    class Pen(turtle.Turtle):
        def __init__(self):
            turtle.Turtle.__init__(self)
            self.shape('square')
            self.color('black')
            self.penup()
            self.speed(0)

        def __del__(self):
            pass


    class Agent(turtle.Turtle):
        def __init__(self):
            turtle.Turtle.__init__(self)
            self.shape('agent.gif')
            self.color('blue')
            self.penup()
            self.speed(speed_screen)

        def __del__(self):
            pass

        def movement(self, newX, newY, agent):
            screen_x = -288 + (newX * 24)
            screen_y = 288 - (newY * 24)
            class_agents[agent].goto(screen_x, screen_y)
            wn.update()


    class Enemy(turtle.Turtle):
        def __init__(self):
            turtle.Turtle.__init__(self)
            self.shape('enemy.gif')
            self.color('red')
            self.penup()
            self.speed(speed_screen)

        def __del__(self):
            pass

        def movement(self, newX, newY, enemy):
            screen_x = -288 + (newX * 24)
            screen_y = 288 - (newY * 24)
            class_enemies[enemy].goto(screen_x, screen_y)
            wn.update()


    class Goal(turtle.Turtle):
        def __init__(self):
            turtle.Turtle.__init__(self)
            self.shape('treasure.gif')
            self.color('gold')
            self.penup()

        def __del__(self):
            pass


class CustomEnv:

    def __init__(self, rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals):
        self.rows = rows
        self.cols = cols
        self.n_agents = n_agents
        self.n_act_agents = n_act_agents
        self.n_enemies = n_enemies
        self.n_act_enemies = n_act_enemies
        self.n_goals = n_goals
        self.n_walls = rows * 2

        # reward definitionm
        self.reward_alive = 0
        self.reward_winner = 1
        self.reward_loser = -1

        self.n_times_loser = 0

        #  game episode
        self.n_steps = 0
        self.len_actions_enemies = 100000

        self.status_col_name = 'Status'

        # grid for visualize agents and enemies positions
        self.grid_for_game = []
        # list for saving enemy' positions
        self.pos_enemies = []
        # list for saving agents' positions
        self.pos_agents = []
        # goal's position
        self.pos_goals = []

        # action space of agents
        self.action_space = Discrete(self.n_act_agents, start=0)

        # defining empyt matrices for game
        for ind_row in range(self.rows):
            row = []
            for ind_col in range(self.cols):
                row.append('-')
            self.grid_for_game.append(row)

        self.observation_space = rows * cols

        # positioning enemies
        row_pos_enemies = []
        for enemy in range(1, self.n_enemies + 1, 1):
            # check if same position
            do = True
            while (do):
                x_nem = random.randint(0, self.rows - 1)
                y_nem = random.randint(0, self.cols - 1)
                if ([x_nem, y_nem] not in row_pos_enemies):
                    do = False
            self.grid_for_game[x_nem][y_nem] = 'En' + str(enemy)
            row_pos_enemies.append([x_nem, y_nem])
            # self.pos_enemies.append([x_nem, y_nem])
        self.pos_enemies.append(row_pos_enemies)

        # positioning agents
        row_pos_agents = []
        for agent in range(1, self.n_agents + 1, 1):
            # check if same position than enemies
            do = True
            while (do):
                x_agent = random.randint(0, self.rows - 1)
                y_agent = random.randint(0, self.cols - 1)
                if ([x_agent, y_agent] not in self.pos_enemies[0]):
                    do = False
            self.grid_for_game[x_agent][y_agent] = 'Agent' + str(agent)
            row_pos_agents.append([x_agent, y_agent])
        self.pos_agents.append(row_pos_agents)

        # for reset
        self.pos_agents_for_reset = self.pos_agents[0].copy()
        self.pos_enemies_for_reset = self.pos_enemies[0].copy()

        self.reset_enemies_attached = [[False] * self.n_enemies] * self.n_agents
        self.reset_enemies_nearby = []
        for agent in range(0, self.n_agents, 1):
            singol_agent = []
            for enemy in range(0, self.n_enemies, 1):
                x_ag = self.pos_agents_for_reset[agent][0]
                y_ag = self.pos_agents_for_reset[agent][1]
                x_en = self.pos_enemies_for_reset[enemy][0]
                y_en = self.pos_enemies_for_reset[enemy][1]
                singol_agent.append(get_direction(x_ag, y_ag, x_en, y_en))
            self.reset_enemies_nearby.append(singol_agent)

        self.list_enemies_actions = []
        for enemy in range(0, self.n_enemies, 1):
            enemy_actions = []
            for act in range(self.len_actions_enemies):
                enemy_actions.append(random.randint(0, self.n_act_enemies - 1))
            self.list_enemies_actions.append(enemy_actions)

        if maze:
            self.walls = []
            for wall in range(0, self.n_walls, 1):
                # check if same position than enemies and agents
                do = True
                while (do):
                    x_wall = random.randint(0, self.rows - 1)
                    y_wall = random.randint(0, self.cols - 1)
                    if ([x_wall, y_wall] not in self.pos_enemies[0] and [x_wall, y_wall] not in self.pos_agents[0] and [
                        x_wall, y_wall]):
                        do = False
                self.grid_for_game[x_wall][y_wall] = 'W'
                self.walls.append([x_wall, y_wall])

        # positioning goal
        for goal in range(1, self.n_goals + 1, 1):
            # check if same position than enemies and agents
            do = True
            while (do):
                x_goal = random.randint(0, self.rows - 1)
                y_goal = random.randint(0, self.cols - 1)
                if ([x_goal, y_goal] not in self.pos_enemies[0] and [x_goal, y_goal] not in self.pos_agents[0]):
                    do = False
            self.grid_for_game[x_goal][y_goal] = 'Goal' + str(goal)
            self.pos_goals.append([x_goal, y_goal])

        for goal_x, goal_y in self.pos_goals:
            check_goals = 0
            vetX = [-1, 0, 1]
            vetY = [-1, 0, 1]

            if goal_x == 0:
                check_goals += 1
                vetX.remove(-1)
            if goal_y == 0:
                check_goals += 1
                vetY.remove(-1)

            if goal_x == self.cols:
                check_goals += 1
                vetX.remove(1)
            if goal_y == self.rows:
                check_goals += 1
                vetY.remove(1)

            for addX in vetX:
                for addY in vetY:
                    if 0 < goal_x + addX < self.cols and 0 < goal_y + addY < self.rows:
                        if self.grid_for_game[goal_x + addX][goal_y + addY] == 'W':
                            check_goals += 1

            if check_goals >= self.n_act_agents - 1:
                for addX in vetX:
                    for addY in vetY:
                        if 0 < goal_x + addX < self.cols and 0 < goal_y + addY < self.rows:
                            if self.grid_for_game[goal_x + addX][goal_y + addY] == 'W':
                                self.grid_for_game[goal_x + addX][goal_y + addY] = '-'
                                break

        for ag_x, ag_y in self.pos_agents_for_reset:
            check_agents = 0
            vetX = [-1, 0, 1]
            vetY = [-1, 0, 1]

            if ag_x == 0:
                check_agents += 1
                vetX.remove(-1)
            if ag_y == 0:
                check_agents += 1
                vetY.remove(-1)

            if ag_x == self.cols:
                check_agents += 1
                vetX.remove(1)
            if ag_y == self.rows:
                check_agents += 1
                vetY.remove(1)

            for addX in vetX:
                for addY in vetY:
                    if 0 < ag_x + addX < self.cols and 0 < ag_y + addY < self.rows:
                        if self.grid_for_game[ag_x + addX][ag_y + addY] == 'W':
                            check_agents += 1

            if check_agents >= self.n_act_agents - 1:
                for addX in vetX:
                    for addY in vetY:
                        if 0 < ag_x + addX < self.cols and 0 < ag_y + addY < self.rows:
                            if self.grid_for_game[ag_x + addX][ag_y + addY] == 'W':
                                self.grid_for_game[ag_x + addX][ag_y + addY] = '-'
                                break

        for ind in range(len(self.grid_for_game)):
            print(self.grid_for_game[ind])

        if visualization:
            self.setup_maze()


    def setup_maze(self):
        count_agents = 0
        count_enemies = 0
        count_goals = 0
        for y in range(len(self.grid_for_game)):
            for x in range(len(self.grid_for_game[y])):
                # get the character at each x, y coordinate
                # NOTE the order of y and x in the next line
                character = self.grid_for_game[y][x]
                # calculate the screen x, y coordinates
                screen_x = -288 + (x * 24)
                screen_y = 288 - (y * 24)

                # check if it is an X (representing a wall)
                if character == 'W':
                    pen.goto(screen_x, screen_y)
                    pen.shape('wall.gif')
                    pen.stamp()
                if x == 0 and y == 0:
                    pen.goto(screen_x - 24, screen_y + 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                    pen.goto(screen_x - 24, screen_y)
                    pen.shape('wall.gif')
                    pen.stamp()
                    pen.goto(screen_x, screen_y + 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                if x == 0:
                    pen.goto(screen_x - 24, screen_y - 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                if y == 0:
                    pen.goto(screen_x + 24, screen_y + 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                if x == len(self.grid_for_game[y]) - 1:
                    pen.goto(screen_x + 24, screen_y + 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                if y == len(self.grid_for_game) - 1:
                    pen.goto(screen_x - 24, screen_y - 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                if x == len(self.grid_for_game[y]) - 1 and y == len(self.grid_for_game) - 1:
                    pen.goto(screen_x + 24, screen_y - 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                    pen.goto(screen_x + 24, screen_y)
                    pen.shape('wall.gif')
                    pen.stamp()
                    pen.goto(screen_x, screen_y - 24)
                    pen.shape('wall.gif')
                    pen.stamp()
                    """pen.goto(screen_x - 24, screen_y + 24)
                    pen.shape('wall.gif')
                    pen.stamp()"""

                # check if it is a Agent (representing a player)
                if character.find('Ag') != -1:
                    class_agents[count_agents].goto(screen_x, screen_y)
                    count_agents += 1

                # check if it is a En (representing Enemy)
                if character.find('En') != -1:
                    class_enemies[count_enemies].goto(screen_x, screen_y)
                    count_enemies += 1

                # check if it is an E (representing Enemy)
                if character.find('Goal') != -1:
                    class_goals[count_goals].goto(screen_x, screen_y)
                    count_goals += 1


    def step_enemies(self):

        self.n_steps = self.n_steps + 1
        rewards, dones, enemies_nearby, enemies_attached = [], [], [], []
        " Enemies "
        " Setting enemies positions"
        new_enemies_pos = []
        for enemy in range(1, self.n_enemies + 1, 1):
            last_stateX_en = self.pos_enemies[-1][enemy - 1][0]
            last_stateY_en = self.pos_enemies[-1][enemy - 1][1]

            if same_enemies_actions:
                n = self.n_steps % len(self.list_enemies_actions)
                action = self.list_enemies_actions[enemy - 1][n]
            else:
                action = random.randint(0, self.n_act_enemies - 1)

            new_stateX_en, new_stateY_en, _, _, _ = get_action(action, last_stateX_en, last_stateY_en)
            if self.grid_for_game[new_stateX_en][new_stateY_en] == 'W':
                new_stateX_en = last_stateX_en
                new_stateY_en = last_stateY_en
            if abs(new_stateX_en-last_stateX_en)+abs(new_stateY_en-last_stateY_en) > 1 and self.n_act_enemies > 4:
                print('Enemy wrong movement',[last_stateX_en, last_stateY_en], '-', [new_stateX_en, new_stateY_en])
            if visualization:
                class_enemies[enemy-1].movement(new_stateX_en, new_stateY_en, enemy-1)
            new_enemies_pos.append([new_stateX_en, new_stateY_en])
        self.pos_enemies.append(new_enemies_pos)

        for agent in range(1, self.n_agents + 1, 1):
            x_ag = self.pos_agents[-1][agent - 1][0]
            y_ag = self.pos_agents[-1][agent - 1][1]
            singol_agent = []
            check_attached = False
            for enemy in range(1, self.n_enemies + 1, 1):
                x_en = self.pos_enemies[-1][enemy - 1][0]
                y_en = self.pos_enemies[-1][enemy - 1][1]
                direction_attached_nearby = get_direction(x_ag, y_ag, x_en, y_en)
                singol_agent.append(direction_attached_nearby)
                if direction_attached_nearby == 0:
                    check_attached = True

            enemies_attached.append(check_attached)
            enemies_nearby.append(singol_agent)

        return enemies_nearby, enemies_attached


    def step_agent(self, agents_actions):
        " Agents "
        rewards = []
        dones = []
        new_agents_pos = []
        for agent in range(1, self.n_agents + 1, 1):
            #print(self.pos_agents[self.n_steps-1])
            last_stateX_ag = self.pos_agents[-1][agent - 1][0]
            last_stateY_ag = self.pos_agents[-1][agent - 1][1]

            action = agents_actions

            new_stateX_ag, new_stateY_ag, res_action, _, _ = get_action(action, last_stateX_ag, last_stateY_ag)
            if self.grid_for_game[new_stateX_ag][new_stateY_ag] == 'W':
                new_stateX_ag = last_stateX_ag
                new_stateY_ag = last_stateY_ag

            #
            #print('ag inside', [last_stateX_ag, last_stateY_ag], [new_stateX_ag, new_stateY_ag])

            if abs(new_stateX_ag-last_stateX_ag)+abs(new_stateY_ag-last_stateY_ag) > 1 and self.n_act_agents > 4:
                print('Agent wrong movement', [last_stateX_ag, last_stateY_ag], '-', [new_stateX_ag, new_stateY_ag], action)
            if visualization:
                class_agents[agent-1].movement(new_stateX_ag, new_stateY_ag, agent - 1)

            # check if agent wins
            win = False
            for goal in self.pos_goals:
                goal_x = goal[0]
                goal_y = goal[1]
                if new_stateX_ag == goal_x and new_stateY_ag == goal_y:
                    reward = self.reward_winner
                    done = True
                    win = True
                    # print('winner', goal, [new_stateX_ag, new_stateY_ag])
            # check if agent loses
            if not win:
                lose = False
                for enemy in range(0, self.n_enemies, 1):
                    X_en = self.pos_enemies[self.n_steps][enemy][0]
                    Y_en = self.pos_enemies[self.n_steps][enemy][1]
                    if new_stateX_ag == X_en and new_stateY_ag == Y_en:
                        reward = self.reward_loser
                        lose = True
                        self.n_times_loser += 1
                        done = False
                        self.n_steps_enemies_actions = 0
                        # print('loser', [X_en, Y_en], [new_stateX_ag, new_stateY_ag], action, res_action)
                # otherwise agent is alive
                if not lose:
                    # print('alive')
                    reward = self.reward_alive
                    done = False

                if win or lose:
                    env.reset(reset_n_times_loser=False)

            # print('new inside', [new_stateX_ag, new_stateY_ag])
            new_agents_pos.append([new_stateX_ag, new_stateY_ag])

            rewards.append(reward)
            dones.append(done)

        self.pos_agents.append(new_agents_pos)

        return new_agents_pos, rewards, dones


    def reset(self, reset_n_times_loser):
        # print('reset')
        if reset_n_times_loser:
            self.n_times_loser = 0

        # reset agents' states
        reset_pos_agents = []
        self.n_steps = 0
        reset_rewards = [0]*self.n_agents
        reset_dones = [False] * self.n_agents

        self.pos_agents = []
        row_agents = []

        for agent in range(self.n_agents):
            newX_ag = self.pos_agents_for_reset[agent - 1][0]
            newY_ag = self.pos_agents_for_reset[agent - 1][1]
            posX_agent = newX_ag
            posY_agent = newY_ag
            reset_pos_agents.append([posX_agent, posY_agent])
            row_agents.append(self.pos_agents_for_reset[agent-1])

            if visualization:
                class_agents[agent].movement(newX_ag, newY_ag, agent-1)

        self.pos_agents.append(row_agents)

        self.pos_enemies = []
        row_enemies = []

        for enemy in range(self.n_enemies):
            row_enemies.append(self.pos_enemies_for_reset[enemy-1])
            newX_en = self.pos_enemies_for_reset[enemy-1][0]
            newY_en = self.pos_enemies_for_reset[enemy - 1][1]

            if visualization:
                class_enemies[enemy - 1].movement(newX_en, newY_en, enemy-1)

        self.pos_enemies.append(row_enemies)

        return reset_pos_agents, reset_rewards, reset_dones, self.reset_enemies_nearby, self.reset_enemies_attached


def plot_RLmetrics(rows, cols, vet_average_episodes_rewards_allGames, vet_steps_for_episode_allGames, dir,
                   table_for_saving, labels_for_plot, indexes_ok):

    n_points = 500
    gaussian_filter_order = 1

    n_games_plot = len(vet_average_episodes_rewards_allGames)
    n_algs_plot = len(vet_average_episodes_rewards_allGames[0])
    n_episodes_plot = len(vet_average_episodes_rewards_allGames[0][0])

    av_rew_final = [0]*n_algs_plot
    steps_needed_final = [[0] * n_episodes_plot] * n_algs_plot
    for select_alg in range(n_algs_plot):
        av_rew_inter = [0]*n_episodes_plot
        steps_needed_inter = [0]*n_episodes_plot
        for select_game in range(n_games_plot):
            for select_component in range(n_episodes_plot):
                av_rew_inter[select_component] += vet_average_episodes_rewards_allGames[select_game][select_alg][select_component]
                steps_needed_inter[select_component] += vet_steps_for_episode_allGames[select_game][select_alg][select_component]
        av_rew_final[select_alg] = av_rew_inter
        steps_needed_final[select_alg] = steps_needed_inter

    for select_alg in range(n_algs_plot):
        for select_component in range(n_episodes_plot):
            av_rew_final[select_alg][select_component] = av_rew_final[select_alg][select_component]/n_games_plot
            steps_needed_final[select_alg][select_component] = steps_needed_final[select_alg][select_component]/n_games_plot

    " Visualization "
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, dpi=500)
    fig.suptitle(f'{word_kindGame} {rows}x{cols} - {n_games} games averaged - {n_enemies} enemies, {word_EnemiesAct}')

    results_for_saving = []
    for select_alg in range(n_algs_plot):
        if select_alg in indexes_ok:
            average_reward_to_saving = np.mean(av_rew_final[select_alg])
            if average_reward_to_saving > 0.99:
                average_reward_to_saving = round(average_reward_to_saving, 4)
            else:
                average_reward_to_saving = round(average_reward_to_saving, 3)
            results_for_saving.append(average_reward_to_saving)

            label_plot = labels_for_plot[select_alg]
            if label_plot == 'QL':
                linewidth = 2
            else:
                linewidth = 1

            """x = np.arange(0, len(av_rew_final[select_alg]), len(av_rew_final[select_alg])/n_points, dtype=int)
            av_rew_for_plot = []
            steps_needed_for_plot = []
            for component in x:
                av_rew_for_plot.append(av_rew_final[select_alg][component])
                steps_needed_for_plot.append(steps_needed_final[select_alg][component])"""

            x = np.arange(0, len(av_rew_final[select_alg]), 1, dtype=int)

            av_rew_for_plot = gaussian_filter1d(av_rew_final[select_alg], gaussian_filter_order)
            steps_needed_for_plot = gaussian_filter1d(steps_needed_final[select_alg], gaussian_filter_order)

            confidence_interval = np.std(av_rew_final[select_alg])

            ax1.plot(x, av_rew_for_plot, linewidth=linewidth, label=f'av_rew = {average_reward_to_saving}', color=colors_sequence[select_alg])
            ax1.fill_between(x, (av_rew_for_plot - confidence_interval), (av_rew_for_plot + confidence_interval), alpha=(n_algs_plot-select_alg)/10)
            ax1.set_title('Average reward on episode steps')
            ax1.legend(fontsize = 'x-small')

            ax2.plot(x, steps_needed_for_plot, linewidth=linewidth, label=f'{label_plot}', color=colors_sequence[select_alg])
            ax2.set_yscale('log')
            ax2.set_title('Steps needed to complete the episode')
            ax2.legend(fontsize = 'x-small')
            ax2.set_xlabel('Episode', fontsize=12)
        else:
            results_for_saving.append('')

    # plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(dir + f'\\{word_kindGame}_{rows}x{cols}_Enemies{n_enemies}_{word_EnemiesAct}_AvRew.png')
    plt.show()

    table_for_saving.loc[f'{rows}x{cols}_Enemies{n_enemies}_{word_EnemiesAct}'] = results_for_saving


def plot_Time(rows, cols, vet_time_for_AllGames, dir, labels_for_plot, vet_timeouts_algs):
    width = 0.5
    fontsize_ticks = 8
    fontsize_labels = 12

    index = labels_for_plot  #np.arange(1, len(labels_for_plot)+1)

    plt.figure(dpi=500)
    for approach in range(len(labels_for_plot)):
        res_time_av = 0
        for singol_game in range(len(vet_time_for_AllGames)):
            res_time_av +=vet_time_for_AllGames[singol_game][approach]
        if vet_timeouts_algs[approach] == 0:
            plt.bar(index[approach], (res_time_av/len(vet_time_for_AllGames))/60, width=width, color=colors_sequence[approach])

    plt.title(f'{word_kindGame} {rows}x{cols} - {n_games} games averaged - {n_enemies} enemies, {word_EnemiesAct}')
    plt.xticks(fontsize=fontsize_ticks, rotation=90)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlabel('Approach used', fontsize=fontsize_labels)
    plt.ylabel('Average running time [min]', fontsize=fontsize_labels, labelpad=5)
    plt.grid()
    plt.subplots_adjust(bottom=0.17, left=0.12)
    plt.savefig(dir + f'\\{word_kindGame}_{rows}x{cols}_Enemies{n_enemies}_{word_EnemiesAct}_Times.png')
    plt.show()


def plot_n_times_loser(rows, cols, vet_losers_for_AllGames, dir, labels_for_plot, vet_timeouts_algs):
    width = 0.5
    fontsize_ticks = 8
    fontsize_labels = 12

    index = labels_for_plot

    plt.figure(dpi=500)

    for approach in range(len(labels_for_plot)):
        res_loser_av = 0
        for singol_game in range(len(vet_losers_for_AllGames)):
            res_loser_av += vet_losers_for_AllGames[singol_game][approach]
        if vet_timeouts_algs[approach] == 0:
            plt.bar(index[approach], res_loser_av / len(vet_losers_for_AllGames), width=width, color=colors_sequence[approach])

    plt.title(f'{word_kindGame} {rows}x{cols} - {n_games} games averaged - {n_enemies} enemies, {word_EnemiesAct}')
    plt.xticks(fontsize=fontsize_ticks, rotation=90)
    plt.yticks(fontsize=fontsize_ticks)
    plt.xlabel('Approach used', fontsize=fontsize_labels)
    plt.ylabel('Average number of defeats', fontsize=fontsize_labels, labelpad=5)
    plt.grid()
    plt.subplots_adjust(bottom=0.17, left=0.12)
    plt.savefig(dir + f'\\{word_kindGame}_{rows}x{cols}_Enemies{n_enemies}_{word_EnemiesAct}_Losers.png')
    plt.show()


def plot_n_timouts(rows, cols, vet_timeouts_algs, dir, labels_for_plot, indexes_ok):
    width = 0.5
    fontsize_ticks = 8
    fontsize_labels = 12

    index = labels_for_plot

    plt.figure(dpi=500)
    i = 0
    for approach in range(len(labels_for_plot)):
        res_timeout = vet_timeouts_algs[approach]
        plt.bar(index[i], res_timeout, width=width, color=colors_sequence[approach])
        i += 1

    plt.title(f'{word_kindGame} {rows}x{cols} - {n_games} games averaged - {n_enemies} enemies, {word_EnemiesAct}')
    plt.xticks(fontsize=fontsize_ticks, rotation=90)
    plt.yticks(np.arange(0, n_games+1, 1), fontsize=fontsize_ticks)
    plt.xlabel('Approach used', fontsize=fontsize_labels)
    plt.ylabel('Number of timeouts', fontsize=fontsize_labels, labelpad=5)
    plt.grid()
    plt.subplots_adjust(bottom=0.17, left=0.12)
    plt.savefig(dir + f'\\{word_kindGame}_{rows}x{cols}_Enemies{n_enemies}_{word_EnemiesAct}_Timeouts.png')
    plt.show()


def causal_model_movement(causal_table, action, oldX, oldY):
    action = 'Action_Agent1_Act' + str(action)

    row_causal_table = causal_table.loc[action, :]

    newX = oldX
    newY = oldY

    for col in range(len(row_causal_table)):
        if 'StateX_Agent' in row_causal_table.index[col]:
            if 0 <= oldX + row_causal_table[col] <= cols-1:
                newX += row_causal_table[col]
        if 'StateY_Agent' in row_causal_table.index[col]:
            if 0 <= oldY + row_causal_table[col] <= rows-1:
                newY += row_causal_table[col]

    return newX, newY


def causal_model_gameover(causal_table, possible_actions, nearbies, attached):

    for nearby in nearbies:
        if nearby != 50 and nearby != 0:
            input = ['EnemiesNearby_Agent1_Dir' + str(nearby)]

            rows_colGameOverIsTrue = causal_table[causal_table['GameOver_Agent1'] == 1]
            row_nearby = rows_colGameOverIsTrue[input] == 1
            index_colGameOver_int = np.where((row_nearby[input] == True).values == True)[0][0]
            row_GameOver = causal_table.loc[row_nearby.index[index_colGameOver_int], :]

            for action in possible_actions:
                if row_GameOver['Action_Agent1_Act' + str(action)] == 1:
                    possible_actions.remove(action)

    if attached:
        input = ['EnemiesAttached_Agent1']

        rows_colGameOverIsTrue = causal_table[causal_table['GameOver_Agent1'] == 1]
        row_nearby = rows_colGameOverIsTrue[input] == 1
        index_colGameOver_int = np.where((row_nearby[input] == True).values == True)[0][0]
        row_GameOver = causal_table.loc[row_nearby.index[index_colGameOver_int], :]

        for action in possible_actions:
            if row_GameOver['Action_Agent1_Act' + str(action)] == 1:
                possible_actions.remove(action)

    return possible_actions


""" 
Game where agents and enemies play in a custom grid.
Agents have to achieve the goal (in term of X-Y position).
If an agent position is equal to an enemy positon --> LOSE; the loser agent will restart from the original starting 
positon, instead enemies will stay in their same positions.
If an agent position is equal to a goal position -> WIN; the winner agent will restart from the original starting 
positon, instead enemies will stay in their same positions.
In the causal model there are no goal
The goals are always in the same position (inititally defined casually)
Movements of the enemies are random and also the agents movements. 
Enemies move first, then agents.
Agent actions -->  0: stop(val=0), 1: right(val_x=+1), 2: left(val_x=-1), 3: up(val_y=+1), 4: down(val_y=-1)
              -->  5: diag_up_right, 6: diag_down_right, 7: diag_up_left, 8:diag_down_left
              Pay attention because we need to specify at least actions = 1, to obtain stopped agents/enemies

Dataframe is generated only for training the BN; in the custom environment the system works step by step.

The causal model generates a Causal_Q_Table where do-calculus probability distribution are reported.
"""

causal_table = pd.read_pickle('final_causal_table.pkl')
for maze in [True]:
    for same_enemies_actions in [False]:

        """ ************************************************************************************************************* """
        " ENVIRONMENT DEFINITION"
        # CQL1: more exploration, no Q-Table update
        # CQL2: more exploration, Q-Table update
        # CQL3: nearby and attached with more exploration, Q-Table update
        # CQL4: nearby and attached without more exploration, Q-Table update
        approaches_choices = ['QL',
                              'CQL1', 'CQL2', 'CQL3', 'CQL4',
                              'CQL1*', 'CQL2*', 'CQL3*', 'CQL4*']
        columms_table_for_saving = ['av_rew_QL',
                                    'av_rew_CQL1', 'av_rew_CQL2', 'av_rew_CQL3', 'av_rew_CQL4',
                                    'av_rew_CQL1*', 'av_rew_CQL2*', 'av_rew_CQL3*', 'av_rew_CQL4*']
        table_for_saving = pd.DataFrame(columns=columms_table_for_saving)

        grids_dim = [10]
        n_episodes = 1000
        n_games = 1

        # exploration reward
        reward_exploration_pos = 0
        reward_exploration_neg = 0

        # thresholds game over and alive
        threshold = 0.7

        # modify folder
        if maze:
            word_kindGame = 'Maze'
        else:
            word_kindGame = 'Grid'

        if same_enemies_actions:
            word_EnemiesAct = 'SameEnemiesActions'
        else:
            word_EnemiesAct = 'RandomEnemiesAction'
        # folder
        dir_principle = f"{word_kindGame}_{len(approaches_choices)}alg_Averaged{n_games}Games_{word_EnemiesAct}_simpleEnv"

        for n_agents in [1]:
            for n_enemies in [5]:
                for n_act_agents in [5]:
                    for n_act_enemies in [5]:

                        # folder for saving plots
                        if not os.path.exists(dir_principle):
                            os.mkdir(dir_principle)
                        dir_secondary = dir_principle + f"\\{n_agents}ag_{n_act_agents}agAct_{n_enemies}en_{n_act_enemies}enAct"
                        if not os.path.exists(dir_secondary):
                            os.mkdir(dir_secondary)

                        for rows in grids_dim:
                            cols = rows
                            n_goals = 1
                            if n_enemies > rows * 2:
                                print(f'No: {n_enemies} enemies in {rows}x{rows} env')
                            else:
                                " RL ENVIRONMENT DEFINITION"
                                vet_average_episodes_rewards_AllGames, vet_steps_for_episode_AllGames, vet_time_for_AllGames, vet_losers_for_AllGames = [], [], [], []
                                vet_timeouts_algs = [0] * len(approaches_choices)
                                for game in range(n_games):

                                    if visualization:
                                        pen = Pen()
                                        class_agents = []
                                        for agent in range(n_agents):
                                            class_agents.append(Agent())
                                        class_enemies = []
                                        for enemy in range(n_enemies):
                                            class_enemies.append(Enemy())
                                        class_goals = []
                                        for goal in range(n_goals):
                                            class_goals.append(Goal())

                                    env = CustomEnv(rows=rows, cols=cols, n_agents=n_agents, n_act_agents=n_act_agents, n_enemies=n_enemies,
                                                n_act_enemies=n_act_enemies, n_goals=n_goals)

                                    vet_average_episodes_rewards_singleGame, vet_steps_for_episode_singleGame, vet_time_for_singleGame, vet_losers_for_singleGame =  [], [], [], []

                                    for approach_choice in approaches_choices:
                                        print(f'Game {game+1}/{n_games}, Algorithm: {approach_choice}')
                                        # initialize the Q-Table
                                        Q_table = np.zeros((rows, cols, n_act_agents))  # x, y, actions
                                        # initialize table to keep track of the explored states
                                        Q_table_track = np.zeros((rows, cols))  # x, y, actions
                                        # initialize the exploration probability to 1
                                        exploration_proba = 1
                                        # minimum of exploration proba
                                        min_exploration_proba = 0.01
                                        # exploration decreasing decay for exponential decreasing
                                        exploration_decreasing_decay = -np.log(min_exploration_proba)/(0.6*n_episodes)
                                        # discounted factor
                                        gamma = 0.99
                                        # learning rate
                                        lr = 0.01
                                        # how many tries the system can do in the exploration
                                        exploration_actions_threshold = 10
                                        # start simulation time
                                        start_time = time.time()

                                        average_episodes_rewards = []
                                        steps_for_episode = []

                                        for e in range(n_episodes):
                                            # intilization
                                            if e == 0:
                                                res_loser = True
                                            else:
                                                res_loser = False
                                            current_state, rewards, dones, enemies_nearby_all_agents, enemies_attached_all_agents = env.reset(res_loser)
                                            current_stateX = current_state[0][0]
                                            current_stateY = current_state[0][1]
                                            # sum the rewards that the agent gets from the environment
                                            total_episode_reward = 0
                                            step_for_episode = 0
                                            done = False
                                            agent = 0

                                            timeout = False
                                            while not done:
                                                actual_time = time.time() - start_time
                                                if actual_time >= time_limit_minutes * 60:
                                                    timeout = True
                                                    break

                                                current_stateX = env.pos_agents[-1][agent][0]
                                                current_stateY = env.pos_agents[-1][agent][1]

                                                # print('current acquired', [current_stateX, current_stateY])
                                                step_for_episode += 1
                                                enemies_nearby_all_agents, enemies_attached_all_agents = env.step_enemies()
                                                # print('pos_en', env.pos_enemies[-1], enemies_nearby_all_agents, enemies_attached_all_agents)
                                                # print('pos ag', env.pos_agents[-1])
                                                " epsiilon-greedy "
                                                if np.random.uniform(0, 1) < exploration_proba: # exploration
                                                    if approach_choice == 'CQL1' or approach_choice == 'CQL1*':
                                                        new = False
                                                        check_tries = 0
                                                        while not new:
                                                            action = env.action_space.sample()
                                                            next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX)

                                                            if Q_table_track[next_stateX, next_stateY] == 0:
                                                                new = True
                                                                reward = reward_exploration_pos
                                                                Q_table_track[next_stateX, next_stateY] = 1

                                                            else:
                                                                check_tries += 1
                                                                if check_tries == exploration_actions_threshold:
                                                                    new = True
                                                                    reward = reward_exploration_neg
                                                                    action = env.action_space.sample()

                                                    elif approach_choice == 'CQL2' or approach_choice == 'CQL2*':
                                                        new = False
                                                        check_tries = 0
                                                        while not new:
                                                            action = env.action_space.sample()
                                                            next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX)

                                                            if Q_table_track[next_stateX, next_stateY] == 0:
                                                                new = True
                                                                reward = reward_exploration_pos
                                                                Q_table_track[next_stateX, next_stateY] = 1
                                                            else:
                                                                check_tries += 1
                                                                if check_tries == exploration_actions_threshold:
                                                                    new = True
                                                                    reward = reward_exploration_neg
                                                                    action = env.action_space.sample()
                                                        # additional Q-Table udpate
                                                        Q_table[current_stateX, current_stateY, action] = (1 - lr) * Q_table[
                                                            current_stateX, current_stateY, action] + lr * (reward + gamma * max(
                                                            Q_table[next_stateX, next_stateY, :]))

                                                    elif approach_choice == 'CQL3' or approach_choice == 'CQL3*':
                                                        enemies_nearby_agent = enemies_nearby_all_agents[agent]
                                                        enemies_attached_agent = enemies_attached_all_agents[agent]
                                                        possible_actions = [s for s in range(n_act_agents)]

                                                        possible_actions =  causal_model_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                                              enemies_attached_agent)

                                                        new = False
                                                        check_tries = 0
                                                        if len(possible_actions) > 0:
                                                            while not new:
                                                                action = random.sample(possible_actions, 1)[0]
                                                                next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX)

                                                                if Q_table_track[next_stateX, next_stateY] == 0:
                                                                    new = True
                                                                    reward = reward_exploration_pos
                                                                    Q_table_track[next_stateX, next_stateY] = 1
                                                                else:
                                                                    check_tries += 1
                                                                    if check_tries == exploration_actions_threshold:
                                                                        new = True
                                                                        reward = reward_exploration_neg
                                                                        action = random.sample(possible_actions, 1)[0]
                                                        else:
                                                            action = env.action_space.sample()

                                                        next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX)

                                                        if enemies_nearby_agent[0] == action and len(possible_actions) > 0:
                                                            print('Problem in action selection with nearby model: action taken', action)
                                                        # additional Q-Table udpate
                                                        Q_table[current_stateX, current_stateY, action] = (1 - lr) * Q_table[
                                                            current_stateX, current_stateY, action] + lr * (reward + gamma * max(
                                                            Q_table[next_stateX, next_stateY, :]))

                                                    elif approach_choice == 'CQL4' or approach_choice == 'CQL4*':
                                                        enemies_nearby_agent = enemies_nearby_all_agents[agent]
                                                        enemies_attached_agent = enemies_attached_all_agents[agent]
                                                        possible_actions = [s for s in range(n_act_agents)]

                                                        possible_actions = causal_model_gameover(causal_table,
                                                                                                 possible_actions,
                                                                                                 enemies_nearby_agent,
                                                                                                 enemies_attached_agent)

                                                        if len(possible_actions) > 0:
                                                            action = random.sample(possible_actions, 1)[0]
                                                        else:
                                                            action = env.action_space.sample()

                                                        next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX)
                                                        reward = 0
                                                        # additional Q-Table udpate
                                                        Q_table[current_stateX, current_stateY, action] = (1 - lr) * Q_table[
                                                            current_stateX, current_stateY, action] + lr * (reward + gamma * max(Q_table[next_stateX, next_stateY, :]))

                                                    else: # classic Q-Learning
                                                        action = env.action_space.sample()
                                                else:  # exploitation
                                                    if approach_choice == 'QL' or approach_choice == 'CQL1' or approach_choice == 'CQL2' or approach_choice == 'CQL3' or approach_choice == 'CQL4':
                                                        action = np.argmax(Q_table[current_stateX, current_stateY, :])
                                                    else:
                                                        enemies_nearby_agent = enemies_nearby_all_agents[agent]
                                                        enemies_attached_agent = enemies_attached_all_agents[agent]
                                                        possible_actions = [s for s in range(n_act_agents)]

                                                        possible_actions = causal_model_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                                              enemies_attached_agent)

                                                        if len(possible_actions) > 0:
                                                            if np.mean(Q_table[current_stateX, current_stateY, possible_actions]) == 0:
                                                                action = np.argmax(Q_table[current_stateX, current_stateY, possible_actions])
                                                            else:
                                                                max_value = max(Q_table[current_stateX, current_stateY, possible_actions])
                                                                possibilities = np.array(np.where(Q_table[current_stateX, current_stateY, :] == max_value)[0])
                                                                action = random.sample(list(possibilities), k=1)[0]
                                                        else:
                                                            action = env.action_space.sample()

                                                result = env.step_agent(action)
                                                # print('result:', result)
                                                next_stateX = int(result[0][agent][0])
                                                next_stateY = int(result[0][agent][1])
                                                reward = int(result[1][agent])
                                                done = result[2][agent] # If agent wins, end loop and restart

                                                # Q-Table update
                                                Q_table[current_stateX, current_stateY, action] = (1 - lr) * Q_table[
                                                    current_stateX, current_stateY, action] + lr * (reward + gamma * max(Q_table[next_stateX, next_stateY, :]))
                                                total_episode_reward = total_episode_reward + reward

                                                if abs(current_stateX-next_stateX)+abs(current_stateY-next_stateY)>1:
                                                    print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

                                                # current_stateX = next_stateX
                                                # current_stateY = next_stateY

                                            if not timeout:
                                                average_episodes_rewards.append(total_episode_reward)
                                                steps_for_episode.append(step_for_episode)
                                                # updating the exploration proba using exponential decay formula
                                                exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay * e))
                                            else:
                                                for ep in range(n_episodes-e):
                                                    average_episodes_rewards.append(-n_enemies)
                                                    steps_for_episode.append(0)
                                                vet_timeouts_algs[approaches_choices.index(approach_choice)] += 1
                                                print('**timeout**')
                                                break

                                        if Q_table_plot and game == 0:
                                            Q_table_for_plot = np.reshape(Q_table, (Q_table.shape[2], Q_table.shape[0] * Q_table.shape[1]))
                                            fig = plt.figure(dpi=500)
                                            plt.title(f'Updated Q-Table of {approach_choice}, {word_kindGame} {rows}x{cols}, {n_enemies}enemies')
                                            sns.heatmap(Q_table_for_plot)
                                            approach_choice = approach_choice.replace('*', 'super')
                                            plt.savefig(dir_secondary + f'\\QT_{approach_choice}_{word_kindGame}_{rows}x{cols}_{n_enemies}enemies.png')
                                            plt.show()

                                        vet_losers_for_singleGame.append(env.n_times_loser)
                                        vet_average_episodes_rewards_singleGame.append(average_episodes_rewards)
                                        vet_steps_for_episode_singleGame.append(steps_for_episode)
                                        final_time = time.time()
                                        vet_time_for_singleGame.append(final_time-start_time)

                                        if visualization:
                                            for agent in range(n_agents):
                                                class_agents[agent].__del__()
                                            for enemy in range(n_enemies):
                                                class_enemies[enemy].__del__()
                                            for goal in range(n_goals):
                                                class_goals[goal].__del__()

                                    vet_time_for_AllGames.append(vet_time_for_singleGame)
                                    vet_average_episodes_rewards_AllGames.append(vet_average_episodes_rewards_singleGame)
                                    vet_steps_for_episode_AllGames.append(vet_steps_for_episode_singleGame)
                                    vet_losers_for_AllGames.append(vet_losers_for_singleGame)

                                indexes_to_remove = []
                                for alg_to_remove in range(len(vet_timeouts_algs)):
                                    if vet_timeouts_algs[alg_to_remove] > 0:
                                        indexes_to_remove.append(alg_to_remove)
                                        print(f'{approaches_choices[alg_to_remove]} removed')
                                indexes_to_remove.sort()
                                indexes_ok = np.arange(0, len(approaches_choices), 1)
                                indexes_ok = np.delete(indexes_ok, indexes_to_remove)

                                if len(indexes_ok) > 0:
                                    plot_RLmetrics(rows, cols, vet_average_episodes_rewards_AllGames, vet_steps_for_episode_AllGames, dir_secondary, table_for_saving, approaches_choices, indexes_ok)
                                    plot_Time(rows, cols, vet_time_for_AllGames, dir_secondary, approaches_choices, vet_timeouts_algs)
                                    plot_n_times_loser(rows, cols, vet_losers_for_AllGames, dir_secondary, approaches_choices, vet_timeouts_algs)
                                if len(indexes_ok) < len(approaches_choices):
                                    plot_n_timouts(rows, cols, vet_timeouts_algs, dir_secondary, approaches_choices, indexes_ok)

                table_for_saving.to_excel(dir_principle + f'\\resume_results.xlsx')
