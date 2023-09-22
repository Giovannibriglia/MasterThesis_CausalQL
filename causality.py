import random
import os
import re
import turtle
import time
import operator
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


def get_action(action, last_stateX, last_stateY, if_en):

    if if_en:  # enemy movement
        if action == 0:  # stop
            new_stateX = last_stateX
            new_stateY = last_stateY
            actionX = 0
            actionY = 0
        elif action == 1:  # up
            new_stateX = 0
            actionX = 0
            sub_action = 1
            if 0 <= sub_action + last_stateY < rows:
                new_stateY = sub_action + last_stateY
                actionY = sub_action
            else:
                new_stateY = last_stateY
                actionY = 0
                action = 0
        elif action == 2:  # down
            new_stateX = 0
            actionX = 0
            sub_action = -1
            if 0 <= sub_action + last_stateY < rows:
                new_stateY = sub_action + last_stateY
                actionY = sub_action
            else:
                new_stateY = last_stateY
                actionY = 0
                action = 0
        elif action == 3:  # right
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
        elif action == 4:  # left
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
        elif action == 5:  # diag_up_right
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
                action = 0
    else:  # agent movement
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
        elif action == 5:  # diag_up_right
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
                action = 0

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

    return direction_ag_en


if visualization:
    wn = turtle.Screen()
    wn.bgcolor('black')
    wn.title('Game')
    # wn.setup(700, 700)
    speed_screen = 1
    wn.tracer(0)

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


class CreateDf:

    def __init__(self, rows, cols, n_agents, n_act_agents, n_enemies, n_act_enemies, n_goals):
        self.rows = rows
        self.cols = cols
        self.n_agents = n_agents
        self.n_act_agents = n_act_agents
        self.n_enemies = n_enemies
        self.n_act_enemies = n_act_enemies
        self.n_goals = n_goals

        # grid for visualize agents and enemies positions
        self.grid_for_game = []
        # list for saving enemy' positions
        self.pos_enemies_for_reset = []
        # list for saving agents' positions
        self.pos_agents_for_reset = []
        # goal's position
        self.pos_goal = []
        # predefined action state
        self.predefined_action_state = 50
        # vectors to save win or lose for each agent
        self.agents_win = [False] * self.n_agents
        self.agents_lose = [False] * self.n_agents

        # defining empyt matrices for game
        for ind_row in range(self.rows):
            row = []
            for ind_col in range(self.cols):
                row.append('-')
            self.grid_for_game.append(row)

        # positioning enemies
        for enemy in range(1, self.n_enemies+1, 1):
            # check if same position
            do = True
            while(do):
                x_nem = random.randint(0, self.rows - 1)
                y_nem = random.randint(0, self.cols - 1)
                if([x_nem, y_nem] not in self.pos_enemies_for_reset):
                    do = False
            self.grid_for_game[x_nem][y_nem] = 'En' + str(enemy)
            self.pos_enemies_for_reset.append([x_nem, y_nem])

        # positioning agents
        for agent in range(1, self.n_agents + 1, 1):
            # check if same position than enemies
            do = True
            while(do):
                x_agent = random.randint(0, self.rows - 1)
                y_agent = random.randint(0, self.cols - 1)
                if([x_agent, y_agent] not in self.pos_enemies_for_reset):
                    do = False
            self.grid_for_game[x_agent][y_agent] = 'Agent' + str(agent)
            self.pos_agents_for_reset.append([x_agent, y_agent])

        # positioning goal
        for goal in range(1, self.n_goals + 1, 1):
            # check if same position than enemies and agents
            do = True
            while(do):
                x_goal = random.randint(0, self.rows - 1)
                y_goal = random.randint(0, self.cols - 1)
                if([x_goal, y_goal] not in self.pos_enemies_for_reset and [x_goal, y_goal] not in self.pos_agents_for_reset and [x_goal, y_goal] != [0, 0]):
                    do = False
            self.grid_for_game[x_goal][y_goal] = 'Goal' + str(goal)
            self.pos_goal.append([x_goal, y_goal])

        """for ind in range(len(self.grid_for_game)):
            print(self.grid_for_game[ind])
        print('\n')"""


    def create_df(self, n_episodes, binary_df):
        print('creating dataframe...')
        # n_col_df = self.n_agents*self.n_act_agents*2 + self.n_enemies*self.n_act_enemies*2
        # Defining columnns names in this way: agent-1, agent0, agent1....en-1, en0, en1...
        self.col_names = []
        self.col_names_bin = []
        " Enemies "
        for enem in range(1, self.n_enemies + 1, 1):
            self.col_names.append('Action_' + 'Enemy' + str(enem))
            self.col_names.append('StateX_' + 'Enemy' + str(enem))
            self.col_names.append('StateY_' + 'Enemy' + str(enem))
            self.col_names_bin.append('StateX_' + 'Enemy' + str(enem))
            self.col_names_bin.append('StateY_' + 'Enemy' + str(enem))
            for act_enem in range(0, self.n_act_enemies, 1):
                name = 'Enemy' + str(enem) + '_Act' + str(act_enem).replace('-','_')
                self.col_names_bin.append('Action_' + name)
                self.col_names_bin.append('Var_' + name)
        " Agents "
        for agent in range(1, self.n_agents + 1, 1):
            self.col_names.append('Action_' + 'Agent' + str(agent))
            self.col_names.append('StateX_' + 'Agent' + str(agent))
            self.col_names.append('StateY_' + 'Agent' + str(agent))
            self.col_names.append('GameOver_' + 'Agent' + str(agent))
            self.col_names.append('EnemiesAttached_' + 'Agent' + str(agent))
            self.col_names.append('Alive_Agent' + str(agent))

            self.col_names_bin.append('StateX_' + 'Agent' + str(agent))
            self.col_names_bin.append('StateY_' + 'Agent' + str(agent))
            self.col_names_bin.append('GameOver_' + 'Agent' + str(agent))
            self.col_names_bin.append('EnemiesAttached_' + 'Agent' + str(agent))
            self.col_names_bin.append('Alive_Agent' + str(agent))

            for act_agent in range(0, self.n_act_agents, 1):
                name = 'Agent' + str(agent) + '_Act' + str(act_agent).replace('-','_')
                self.col_names_bin.append('Action_' + name)
                self.col_names_bin.append('Var_' + name)
                if act_agent >= 0:
                    self.col_names.append('EnemiesNearby_Agent' + str(agent) + '_Dir'+ str(act_agent))
                    self.col_names.append('Contact_Agent' + str(agent) + '_Dir'+ str(act_agent))
                    self.col_names_bin.append('EnemiesNearby_Agent' + str(agent) + '_Dir'+ str(act_agent))
                    self.col_names_bin.append('Contact_Agent' + str(agent) + '_Dir'+ str(act_agent))
            self.col_names.append('GeneralContact_Agent' + str(agent))
            self.col_names.append('Reset')
            self.col_names_bin.append('Reset')
            self.col_names_bin.append('GeneralContact_Agent' + str(agent))

        " Create first row: episode 0 "
        first_row = []
        for enemy in range(0, self.n_enemies, 1):
            first_row.append(0)                          # action
            first_row.append(self.predefined_action_state) # stateX
            first_row.append(self.predefined_action_state) # stateY
        for agent in range(0, self.n_agents, 1):
            first_row.append(0)                            # action
            first_row.append(self.predefined_action_state) # stateX
            first_row.append(self.predefined_action_state) # stateY
            first_row.append(0)                            # game over
            first_row.append(0)                            # alert if there is an enemy in the same cell of the agent
            first_row.append(1)                            # agent alive
            #first_row.append(0)                           # contact: agent went against enemy

            for enemy in range(0, self.n_enemies, 1):
                pos_enX = self.pos_enemies_for_reset[enemy][0]
                pos_enY = self.pos_enemies_for_reset[enemy][1]
                pos_agX = self.pos_agents_for_reset[agent][0]
                pos_agY = self.pos_agents_for_reset[agent][1]
                direction = get_direction(pos_agX, pos_agY, pos_enX, pos_enY)
                for act_agent in range(0, self.n_act_agents, 1):
                    if act_agent >= 0:
                        if direction == act_agent:            # action which brings the agent on the enemies
                            first_row.append(1)
                        else:
                            first_row.append(0)
                        first_row.append(0)                   # contact direction
        first_row.append(0)                                   # general contact
        first_row.append(0)                                   # reset

        last_states_en = [[0, 0]]*self.n_enemies
        last_states_ag = [[0, 0]]*self.n_agents

        " Dataframe "
        df = pd.DataFrame([first_row], columns=self.col_names)
        # Defining random actions for each agent and enemy and update the states
        for episode in range(1, n_episodes, 1):
            " Enemies "
            for enemy in range(1, self.n_enemies+1, 1):
                last_stateX_en = last_states_en[enemy-1][0]
                last_stateY_en = last_states_en[enemy-1][1]

                action = random.randint(0, self.n_act_enemies-1)

                new_stateX_en, new_stateY_en, action, new_deltaX_en, new_deltaY_en  = get_action(action, last_stateX_en, last_stateY_en, if_en=True)
                df.loc[episode, 'Action_' + 'Enemy' + str(enemy)] = action
                df.loc[episode, 'StateX_' + 'Enemy' + str(enemy)] = new_deltaX_en
                df.loc[episode, 'StateY_' + 'Enemy' + str(enemy)] = new_deltaY_en

                last_states_en[enemy-1] = [new_stateX_en, new_stateY_en]

                # print(f'Enemy--> Last: {last_stateX_en, last_stateY_en}; Action: {action}, New: {new_stateX_en, new_stateY_en}')
            " Check same cell enemies-agents "
            for enemy in range(1, self.n_enemies+1, 1):
                val_enX = last_states_en[enemy-1][0]
                val_enY = last_states_en[enemy-1][1]
                for agent in range(1, self.n_agents+1, 1):
                    val_agX = last_states_ag[agent-1][0]
                    val_agY = last_states_ag[agent-1][1]
                    if (val_agX == val_enX) and (val_agY == val_enY):
                        df.loc[episode, 'EnemiesAttached_' + 'Agent' + str(agent)] = 1
                        # print('Alert', [val_agX, val_agY], [val_enX, val_enY])
                    else:
                        df.loc[episode, 'EnemiesAttached_' + 'Agent' + str(agent)] = 0
                        # print('No alert', [val_agX, val_agY], [val_enX, val_enY])
            " Check if there are enemies near agents"
            for agent in range(1, self.n_agents+1, 1):
                for enemy in range(0, self.n_enemies, 1):
                    pos_enX = last_states_en[enemy-1][0]
                    pos_enY = last_states_en[enemy-1][1]
                    pos_agX = last_states_ag[agent-1][0]
                    pos_agY = last_states_ag[agent-1][1]
                    direction = get_direction(pos_agX, pos_agY, pos_enX, pos_enY)
                    for act_agent in range(0, self.n_act_agents, 1):
                        if direction == act_agent and act_agent >= 0:
                            df.loc[episode, 'EnemiesNearby_Agent' + str(agent) + '_Dir'+ str(act_agent)] = 1
                        else:
                            df.loc[episode, 'EnemiesNearby_Agent' + str(agent) + '_Dir'+ str(act_agent)] = 0

            " Agents "
            for agent in range(1, self.n_agents+1, 1):
                # if agent lot --> restart from the same previous start position
                if self.agents_win[agent-1] or self.agents_lose[agent-1]:
                    df.loc[episode, 'Action_' + 'Agent' + str(agent)] = 0
                    new_stateX_ag = self.pos_agents_for_reset[agent - 1][0]
                    new_stateY_ag = self.pos_agents_for_reset[agent - 1][1]
                    #print('Pos targ', self.pos_agents[agent-1])
                    df.loc[episode, 'StateX_' + 'Agent' + str(agent)] = self.predefined_action_state
                    df.loc[episode, 'StateY_' + 'Agent' + str(agent)] = self.predefined_action_state
                    df.loc[episode, 'GameOver_' + 'Agent' + str(agent)] = 0
                    df.loc[episode, 'EnemiesAttached_Agent' + str(agent)] = 0
                    # df.loc[episode, 'GameOver_Agent' + str(agent) + 'Stop'] = 0
                    self.agents_win[agent - 1] = 0
                    self.agents_lose[agent - 1] = 0
                    for enemy in range(1, self.n_enemies + 1, 1):
                        df.loc[episode, 'StateX_' + 'Enemy' + str(enemy)] = self.pos_enemies_for_reset[enemy - 1][0]
                        df.loc[episode, 'StateY_' + 'Enemy' + str(enemy)] = self.pos_enemies_for_reset[enemy - 1][0]
                        df.loc[episode, 'Action_' + 'Enemy' + str(enemy)] = 0
                    df.loc[episode, 'Reset'] = 1
                    df.loc[episode, 'Alive_Agent' + str(agent)] = 1
                    for act_agent in range(0, self.n_act_agents, 1):
                        if act_agent >= 0:
                            df.loc[episode, 'Contact_Agent' + str(agent) + '_Dir'+ str(act_agent)] = 0
                    df.loc[episode, 'GeneralContact_Agent' + str(agent)] = 0
                else:
                    win, lose = False, False

                    last_stateX_ag = last_states_ag[agent-1][0]
                    last_stateY_ag = last_states_ag[agent-1][1]

                    action = random.randint(0, self.n_act_agents -1)
                    new_stateX_ag, new_stateY_ag, action, new_deltaX_ag, new_deltaY_ag = get_action(action, last_stateX_ag, last_stateY_ag, if_en=False)

                    df.loc[episode, 'Action_' + 'Agent' + str(agent)] = action
                    df.loc[episode, 'StateX_' + 'Agent' + str(agent)] = new_deltaX_ag
                    df.loc[episode, 'StateY_' + 'Agent' + str(agent)] = new_deltaY_ag

                    last_states_ag[agent-1] = [new_stateX_ag, new_stateY_ag]

                    # check if the agent won
                    for goal in self.pos_goal:
                        goal_x = goal[0]
                        goal_y = goal[1]
                        if new_stateX_ag == goal_x and new_stateY_ag == goal_x:
                            win = True
                            self.agents_win[agent-1] = True
                            # print('win', new_stateX_ag, new_stateY_ag, goal_x, goal_y)
                    if not win:
                        # check if the agent lost
                        self.agents_win[agent-1] = False
                        for enemy_check in range(1, self.n_enemies+1, 1):
                            val_enX = last_states_en[enemy - 1][0]
                            val_enY = last_states_en[enemy - 1][1]
                            if (new_stateX_ag == val_enX) and (new_stateY_ag == val_enY):
                                lose = True
                                # print('GameOver', [new_stateX_ag, new_stateY_ag], [int(val_enX), int(val_enY)])
                    if lose:
                        df.loc[episode, 'GameOver_' + 'Agent' + str(agent)] = 1
                        self.agents_lose[agent - 1] = True
                        df.loc[episode, 'Reset'] = 0
                        df.loc[episode, 'Alive_Agent'+str(agent)] = 0
                        for act_agent in range(0, self.n_act_agents, 1):
                            if action == act_agent and action >= 0:
                                df.loc[episode, 'Contact_Agent' + str(agent) + '_Dir' + str(act_agent)] = 1
                                df.loc[episode, 'GeneralContact_Agent' + str(agent)] = 1
                            else:
                                df.loc[episode, 'Contact_Agent' + str(agent) + '_Dir' + str(act_agent)] = 0
                                df.loc[episode, 'GeneralContact_Agent' + str(agent)] = 0
                    else:
                        df.loc[episode, 'GameOver_' + 'Agent' + str(agent)] = 0
                        self.agents_lose[agent - 1] = False
                        df.loc[episode, 'Reset'] = 0
                        df.loc[episode, 'Alive_Agent'+str(agent)] = 1
                        for act_agent in range(0, self.n_act_agents, 1):
                            if act_agent >= 0:
                                df.loc[episode, 'Contact_Agent' + str(agent) + '_Dir' + str(act_agent)] = 0
                                df.loc[episode, 'GeneralContact_Agent' + str(agent)] = 0
                        # df.loc[episode, 'GameOver_Agent' + str(agent) + 'Stop'] = 0
                    # otherwise agent is alive

                # print(f'Agent--> Last: {last_stateX_ag, last_stateY_ag}; Action: {action}, New: {new_stateX_ag, new_stateY_ag}')

        # removing columns without values changes
        for col in df.columns:
            if df[col].std() == 0:
                del df[col]
                col_to_del = [s for s in self.col_names_bin if col in s or col.replace('Action', 'Var') in s]
                for element_to_del in col_to_del:
                    self.col_names_bin.remove(element_to_del)

        if not binary_df:
            for col in df.columns:
                df[str(col)] = df[str(col)].astype(str).str.replace(',', '').astype(float)

            return df
        else:
            df_bin = pd.DataFrame([], columns=self.col_names_bin)

            " Agents "
            actions_agent_cols_index = [df.columns.to_list().index(s) for s in df.columns.to_list() if
                                        'Action_Agent' in s]
            actions_agents = np.arange(0, self.n_act_agents)
            for agent in range(self.n_agents):
                for action in actions_agents:
                    for n_col in actions_agent_cols_index:
                        col_name = df.columns.to_list()[n_col]
                        old_series = df[col_name]
                        new_series = []
                        for n_row in range(len(df)):
                            if df.iloc[n_row, n_col] == action:
                                new_series.append(1)
                            else:
                                new_series.append(0)
                        df_bin[col_name + '_Act' + str(action).replace('-','_')] = new_series
                        df_bin[col_name.replace('Action', 'Var') + '_Act' + str(action).replace('-','_')] = new_series

            " Agents and Enemies states "
            states_agent_cols_names = [s for s in df.columns.to_list() if 'State' in s or 'GameOver' in s
                                       or 'Reset' in s or 'EnemiesAttached_Agent' in s or 'Nearby_' in s
                                       or 'Alive' in s or 'Contact_Agent' in s]
            for col_name in states_agent_cols_names:
                df_bin[col_name] = df[col_name]

            " Enemies "
            actions_enemy_cols_index = [df.columns.to_list().index(s) for s in df.columns.to_list() if
                                        'Action_Enemy' in s]
            actions_enemies = np.arange(0, self.n_act_enemies)
            for enemy in range(self.n_enemies):
                for action in actions_enemies:
                    for n_col in actions_enemy_cols_index:
                        col_name = df.columns.to_list()[n_col]
                        old_series = df[col_name]
                        new_series = []
                        for n_row in range(len(df)):
                            if df.iloc[n_row, n_col] == action:
                                new_series.append(1)
                            else:
                                new_series.append(0)
                        df_bin[col_name + '_Act' + str(action).replace('-','_')] = new_series
                        df_bin[col_name.replace('Action', 'Var') + '_Act' + str(action).replace('-','_')] = new_series


            for col in df_bin.columns:
                df_bin[str(col)] = df_bin[str(col)].astype(str).str.replace(',', '').astype(float)

            feat_to_del = ['Reset', 'GeneralContact_Agent1']
            df_bin.drop(feat_to_del, axis=1, inplace=True)

            return df_bin


"""class Causality:

    def __init__(self, df):
        " You must use 3x3 grid "
        # input dataframe
        self.df = df
        self.col_names = self.df.columns.to_list()
        self.models = [] # name, df, features, parents, childs, (specials, targets)

        # UNDERSTANDING MOVEMENTS
        model = []
        model.append('Movement')

        feat_to_del_mov = [s for s in self.col_names if 'Var' not in s and 'State' not in s and 'Action' not in s]
        df = self.df.drop(feat_to_del_mov, axis=1)
        model.append(df)
        features = df.columns.to_list()
        model.append(features)
        model.append([s for s in features if 'Var' in s or 'State' in s or 'GameOver' in s])
        model.append([s for s in features if 'Action' in s])
        self.models.append(model)

        # UNDERSTANDING CONTACT
        model = []
        model.append('Contact')

        feat_to_del_contact = [s for s in self.col_names if 'Var' in s or 'State' in s or 'Action' in s or 'Reset' in s or 'Nearby' in s]
        df = self.df.drop(feat_to_del_contact, axis=1)
        model.append(df)
        features = df.columns.to_list()
        model.append(features)
        model.append([s for s in features if 'GameOver' in s or 'Alive' in s])
        model.append([s for s in features if 'Attached' in s or 'Contact' in s])
        self.models.append(model)

        # UNDERSTANDING ATTACHED
        model = []
        model.append('Attached')

        feat_to_del_attached = [s for s in self.col_names if 'Var' in s or 'State' in s or 'Action_Enemy' in s or 'Reset' in s or 'Nearby' in s or 'Contact' in s]
        df = self.df.drop(feat_to_del_attached, axis=1)
        model.append(df)
        features = df.columns.to_list()
        model.append(features)
        model.append([s for s in features if 'GameOver' in s or 'Alive' in s])
        model.append([s for s in features if 'Action' in s or 'Attached' in s or 'Nearby' in s])
        model.append([s for s in features if 'Attached' in s or 'Nearby' in s])
        model.append([s for s in features if 'GameOver' in s or 'Alive' in s])
        self.models.append(model)

        # UNDERSTANDING NEARBY
        model = []
        model.append('Nearby')

        feat_to_del_nearby = [s for s in self.col_names if 'Action_Agent' not in s and 'Contact' not in s and 'Nearby' not in s and 'Alive' not in s and 'GameOver' not in s and 'Attached' not in s]
        df = self.df.drop(feat_to_del_nearby, axis=1)
        model.append(df)
        features = df.columns.to_list()
        model.append(features)
        model.append([s for s in features if 'GameOver' in s or 'Alive' in s])
        model.append([])
        model.append([s for s in features if 'Contact' in s])
        model.append([s for s in features if 'GameOver' in s or 'Alive' in s or 'Contact' in s])
        self.models.append(model)


    def training(self):
        graphs = []
        for model in self.models:
            # name, df, features, parents, childs, (specials, targets)
            print(f'structuring {model[0]} model...')
            # Training the model
            structureModel = from_pandas(model[1], tabu_child_nodes=model[4], tabu_parent_nodes=model[3])
            structureModel.remove_edges_below_threshold(0.3)

            " ********************************************************** "
            " Plot structure "
            plt.figure(dpi=500)
            plt.title(f'{model[0]} model')
            networkx.draw(structureModel, pos=networkx.circular_layout(structureModel),
                          with_labels=True, font_size=6, edge_color='orange')
            # networkx.draw(structureModel, with_labels=True, font_size=3, edge_color='orange')
            plt.savefig(f'{model[0]} model.png')
            plt.show()

            " ********************************************************** "
            " Find the subgraphs "
            for component in networkx.weakly_connected_components(structureModel):
                subgraph = structureModel.subgraph(component).copy()
                if len(subgraph.edges) > 1:
                      graphs.append(list(subgraph.edges))

        return graphs


    def counterfactual(self, graphs):
        final_causalTable = pd.DataFrame()
        for model in range(1, len(graphs)):
            # name, df, features, parents, childs, (specials, targets)
            print('\n**********************************')
            " Defining substructure model "
            sub_structureModel = StructureModel()
            sub_structureModel.add_edges_from(graphs[model])

            " ********************************************************** "
            " Bayesian network instantiation "
            print('training BN...')
            df_work = self.models[model][1]
            bn = BayesianNetwork(sub_structureModel)
            bn = bn.fit_node_states_and_cpds(df_work)

            bad_nodes = [node for node in bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
            if bad_nodes:
                print('Bad nodes: ', bad_nodes)

            " ********************************************************** "
            " Inference "
            print('inference...')
            ie = InferenceEngine(bn)

            " ********************************************************** "
            " Doing Do-Calculus and Defining Causal Q-Table "
            if len(self.models[model]) > 5:
                specials = self.models[model][5]
                targets = self.models[model][6]
                childs = self.models[model][4]
                if self.models[model][0] == 'Nearby':
                    childs_do_calculus = [s for s in self.models[model][2] if 'Action' in s]
                else:
                    childs_do_calculus = [s for s in childs if s not in specials]

                # print('specials: ', specials)
                # print('targets: ', targets)
                # print('childs: ', childs_do_calculus)

                columns_causal_table = specials+targets+childs_do_calculus
                columns_causal_table[:] = [x for i, x in enumerate(columns_causal_table) if i == columns_causal_table.index(x)]
                causal_table = pd.DataFrame(columns=columns_causal_table)

                row = 0
                for special in specials:
                    for true_false_special in [True, False]:
                        for child in childs_do_calculus:
                            if special.find('Attached') == -1 or child.find('Enemy') == -1 or special != child:
                                ie.do_intervention(special, true_false_special)
                                for true_false_child in [True, False]:
                                    ie.do_intervention(child, true_false_child)
                                    elements = [0] * len(columns_causal_table)
                                    for target in targets:
                                        after_do = round(ie.query()[target][1],3)
                                        print(f'p({target} | do=[{special}={true_false_special} and {child}={true_false_child}]) = ',after_do)  # P(parent|do(child))

                                        for other_target in targets:
                                            col = columns_causal_table.index(other_target)
                                            if other_target == target:
                                                elements[col] = after_do

                                    for other_special in specials:
                                        col = columns_causal_table.index(other_special)
                                        if other_special == special:
                                            elements[col] = str(true_false_special)
                                        else:
                                            elements[col] = 'ind'

                                    for other_child in childs_do_calculus:
                                        col = columns_causal_table.index(other_child)
                                        if other_child == child:
                                            elements[col] = str(true_false_child)
                                        else:
                                            elements[col] = 'ind'

                                    causal_table.loc[row] = elements
                                    row = row+1

                                    ie.reset_do(child)
                                ie.reset_do(special)

                causal_table.to_excel(f'causal_table_{n_act_agents}agAct_{self.models[model][0]}.xlsx')
                # causal_table.to_pickle(f'causal_table_{n_act_agents}agAct_{self.models[model][0]}.pkl')
            else:
                childs = self.models[model][4]
                parents = self.models[model][3]
                columns_causal_table = parents + childs
                columns_causal_table[:] = [x for i, x in enumerate(columns_causal_table) if
                                           i == columns_causal_table.index(x)]
                causal_table = pd.DataFrame(columns=columns_causal_table)

                row = 0
                for child in childs:
                    for true_false_child in [True, False]:
                        ie.do_intervention(child, true_false_child)
                        elements = [0] * len(columns_causal_table)
                        for parent in parents:
                            after_do = round(ie.query()[parent][1], 3)
                            print(f'p({parent} | do=[{child}={true_false_child}]) = ', after_do)  # P(parent|do(child))

                            for other_parent in parents:
                                col = columns_causal_table.index(other_parent)
                                if other_parent == parent:
                                    elements[col] = after_do

                        for other_child in childs:
                            col = columns_causal_table.index(other_child)
                            if other_child == child:
                                elements[col] = str(true_false_child)
                            else:
                                elements[col] = 'ind'

                        causal_table.loc[row] = elements
                        row = row + 1

                        ie.reset_do(child)

            # causal_table.to_excel(f'causal_table_{n_act_agents}agAct_{self.models[model][0]}.xlsx')
            causal_table.to_pickle(f'causal_table_{n_act_agents}agAct_{self.models[model][0]}.pkl')
"""
class Causality:

    def __init__(self, df):
        # input dataframe
        self.df = df
        self.parents_names = ['GameOver', 'Alive', 'Var', 'State']


    def training(self):
        # Training the model
        parents = []
        for col in self.df.columns.to_list():
            for par_name in self.parents_names:
                if par_name in col:
                    parents.append(col)

        self.structureModel = from_pandas(self.df, tabu_parent_nodes=parents)
        self.structureModel.remove_edges_below_threshold(0.3)

        " ********************************************************** "
        " Plot structure "
        plt.figure(dpi=500)
        plt.title(f'Before: {self.structureModel}', fontsize=16)
        networkx.draw(self.structureModel, pos=networkx.circular_layout(self.structureModel), with_labels=True, font_size=6, edge_color='orange')
        plt.show()

        self.interesting_features_gameover_directly = []
        for child, parent, weight in self.structureModel.edges(data="weight"):
            if 'GameOver' in parent:
                self.interesting_features_gameover_directly.append(child)

        self.interesting_features = []
        for child, parent, weight in self.structureModel.edges(data="weight"):
            if child in self.interesting_features_gameover_directly:
                self.interesting_features.append(parent)

        self.interesting_features[:] = list(set(self.interesting_features[:]))

        print('Directly interesting features for game over: \n', self.interesting_features_gameover_directly)
        print('Interesting features for the causal model before cleaning: \n', self.interesting_features)
        " ********************************************************************* "

        nodes_to_remove = []
        initial_nodes = self.structureModel.nodes
        for feat in initial_nodes:
            if feat not in self.interesting_features and feat not in self.interesting_features_gameover_directly and feat not in 'GameOver_Agent1' and feat not in 'Alive_Agent1':
                nodes_to_remove.append(feat)
        print('Removed nodes: ', nodes_to_remove)
        self.structureModel.remove_nodes_from(nodes_to_remove)

        self.interesting_features_final = self.structureModel.nodes

        copy_structureModel = self.structureModel.copy()
        for child, parent, weight in copy_structureModel.edges(data="weight"):
            if not(child in self.interesting_features_final):
                self.structureModel.remove_edge(child, parent)

        graphs = []
        for component in networkx.weakly_connected_components(self.structureModel):
            subgraph = self.structureModel.subgraph(component).copy()
            if len(subgraph.edges) > 1:
                elements = list(subgraph.edges)

                weights = []
                for child, parent, weight in subgraph.edges(data="weight"):
                    weights.append(weight)

                new_elements = []
                for index_in_elements in range(len(elements)):
                    val = weights[index_in_elements]
                    new_elements.append(elements[index_in_elements] + (val,))

                graphs.append(new_elements)
            else:
                print('Removed node: ', subgraph.nodes)
                pass

        self.structureModel = StructureModel()
        self.structureModel.add_weighted_edges_from(graphs[0])

        plt.figure(dpi=500)
        plt.title(f'After: {self.structureModel}', fontsize=16)
        networkx.draw(self.structureModel, pos=networkx.circular_layout(self.structureModel), with_labels=True, font_size=6, edge_color='orange')
        plt.show()


    def counterfactual(self):
        features_work = self.structureModel.nodes
        init_time = time.time()
        " ********************************************************** "
        " Bayesian network instantiation "
        print('training BN...')
        bn = BayesianNetwork(self.structureModel)
        bn = bn.fit_node_states_and_cpds(self.df)

        bad_nodes = [node for node in bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
           print('Bad nodes: ', bad_nodes)

        " ********************************************************** "
        " Inference "
        print('inference...')
        ie = InferenceEngine(bn)

        " ********************************************************** "
        " Defining Causal Q-Tables "
        print('do-calculus...')
        rows_causal_Qtable = []
        for row in features_work:
            rows_causal_Qtable.append(row)

        causal_table = pd.DataFrame(data=[[0] * len(features_work)] * len(rows_causal_Qtable),
                                             columns=features_work, index=rows_causal_Qtable)
        # most probably value is selected
        for action in self.interesting_features_final:
            try:
                ie.do_intervention(action, True)
                after_do1 = ie.query()
                # print('*****************************************')
                for goal in features_work:
                    best = max(after_do1[goal].items(), key=operator.itemgetter(1))
                    # print(f'{goal} after do {action} is equal to {int(best[0])} with prob {round(best[1]*100, 2)} %')
                    causal_table.loc[str(action)][goal] = best[0]
                ie.reset_do(action)
            except:
                # print('no ', action)
                pass
        print(f'Time needed: {round((time.time() - init_time)/60, 2)} min')

        rows_to_remove = []
        for row in causal_table.index:
            row_values = list(causal_table.loc[row, :])
            if np.std(row_values) == 0:
                rows_to_remove.append(row)

        cols_to_remove = []
        for col in causal_table.columns.to_list():
            col_values = list(causal_table.loc[:, col])
            if np.std(col_values) == 0:
                cols_to_remove.append(col)

        causal_table.drop(rows_to_remove, axis=0, inplace=True)
        causal_table.drop(cols_to_remove, axis=1, inplace=True)

        causal_table.to_excel('complete.xlsx')
        causal_table.to_pickle('final_causal_table.pkl')


def get_CausaltablesXY(causal_table, rows, cols):

    rows_table = causal_table.index.to_list()
    cols_agents_actions = [s for s in causal_table.columns.to_list() if 'Agent' in s]
    actions = len(cols_agents_actions)
    Causal_TableX = np.zeros((rows, actions))
    Causal_TableY = np.zeros((cols, actions))

    rows_agents_actionsX_consequences = [s for s in rows_table if 'StateX' in s and 'Agent' in s]
    rows_agents_actionsY_consequences = [s for s in rows_table if 'StateY' in s and 'Agent' in s]

    for stateX in range(rows):
        for stateY in range(cols):
            for col_agent_action in cols_agents_actions:
                if rows_agents_actionsX_consequences != []:
                    value_most_probX = 0
                    for row_actX in rows_agents_actionsX_consequences:
                        valX = causal_table.loc[row_actX, col_agent_action]
                        if valX >= value_most_probX:
                            value_most_probX = valX
                            ind_for_cutting = row_actX.index('val')
                            action_most_probX_in = row_actX.replace(row_actX[:ind_for_cutting + 3], "")
                            if action_most_probX_in.find('_') != -1:
                                action_most_probX = -int(action_most_probX_in.replace('_', ''))
                            else:
                                action_most_probX = int(action_most_probX_in)
                else:
                    action_most_probX = 0

                if 0 <= stateX + action_most_probX < cols-1:
                    update_stateX = stateX + action_most_probX
                else:
                    update_stateX = stateX

                if rows_agents_actionsY_consequences != []:
                    value_most_probY = 0
                    for row_actY in rows_agents_actionsY_consequences:
                        valY = causal_table.loc[row_actY, col_agent_action]
                        if valY >= value_most_probY:
                            value_most_probY = valY
                            ind_for_cutting = row_actY.index('val')
                            action_most_probY = row_actY.replace(row_actY[:ind_for_cutting+3], "")
                            if action_most_probY.find('_') != -1:
                                action_most_probY = -int(action_most_probY.replace('_',''))
                            else:
                                action_most_probY = int(action_most_probY)
                else:
                    action_most_probY = 0

                if 0 <= stateY + action_most_probY < rows-1:
                    update_stateY = stateY + action_most_probY
                else:
                    update_stateY = stateY

                Causal_TableX[stateX, cols_agents_actions.index(col_agent_action)] = update_stateX
                Causal_TableY[stateY, cols_agents_actions.index(col_agent_action)] = update_stateY

    return Causal_TableX, Causal_TableY


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
Enemy actions -->  1: stop, 2: up(val_y=+1), 3: down(val_y=-1), 4: right(val_x=+1), 5: left(val_x=-1)
              -->  6: diag_up_right, 7: diag_down_right, 8: diag_up_left, 9:diag_down_left
Agent actions -->  1: stop(val=0), 2: right(val_x=+1), 3: left(val_x=-1), 4: up(val_y=+1), 5: down(val_y=-1)
              -->  6: diag_up_right, 7: diag_down_right, 8: diag_up_left, 9:diag_down_left

Dataframe is generated only for training the BN; in the custom environment the system works step by step.

The causal model generates a Causal_Q_Table where do-calculus probability distribution are reported.
"""


for maze in [False]:
    for same_enemies_actions in [False]:

        """ ************************************************************************************************************* """
        " ENVIRONMENT DEFINITION"
        # CQL1: more exploration, no Q-Table update
        # CQL2: more exploration, Q-Table update
        # CQL3: nearby and attached with more exploration, Q-Table update
        # CQL4: nearby and attached without more exploration, Q-Table update
        approaches_choices = ['QL',
                              'CQL1_noExpMod', 'CQL2_noExpMod', 'CQL3_noExpMod', 'CQL4_noExpMod',
                              'CQL1_ExpMod', 'CQL2_ExpMod', 'CQL3_ExpMod', 'CQL4_ExpMod',]
        columms_table_for_saving = ['av_rew_QL',
                                    'av_rew_CQL1_noExpMod', 'av_rew_CQL2_noExpMod', 'av_rew_CQL3_noExpMod', 'av_rew_CQL4_noExpMod',
                                    'av_rew_CQL1_ExpMod', 'av_rew_CQL2_ExpMod', 'av_rew_CQL3_ExpMod', 'av_rew_CQL4_ExpMod']
        table_for_saving = pd.DataFrame(columns=columms_table_for_saving)

        grids_dim = [3]
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
        dir_principle = f"{word_kindGame}_{len(approaches_choices)}alg_Averaged{n_games}Games_{word_EnemiesAct}"


        for n_agents in [1]:
            for n_enemies in [1]:
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
                            # number of nodes = n_agents*(n_act_agents*2+2) + n_enemies*(n_act_enemies*2+2)
                            """ ************************************************************************************************************* """
                            " Dataframe "
                            create = CreateDf(rows=rows, cols=cols, n_agents=1, n_act_agents=n_act_agents,
                                             n_enemies=1, n_act_enemies=n_act_enemies, n_goals=0)
                            df = create.create_df(n_episodes=10000, binary_df=True)
                            df.to_pickle('mario.pkl')
                            " Causal Model "
                            causal_model = Causality(df)
                            causal_model.training()
                            causal_model.counterfactual()

