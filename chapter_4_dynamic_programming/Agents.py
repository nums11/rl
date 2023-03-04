import numpy as np
import pprint

class State(object):
    """
    Represents a state or a point in the grid.

    coord: coordinate in grid world
    """
    def __init__(self, coord, is_terminal):
        self.coord = coord
        self.action_state_transitions = self._getActionStateTranstions()
        self.is_terminal = is_terminal
        self.reward = 5 if is_terminal else -1

    # Returns a dictionary mapping each action to the following state
    # it would put the agent in from the currrent state
    def _getActionStateTranstions(self):
        action_state_transitions = {}
        # Action 0 - up
        if self._isFirstRowState():
            action_state_transitions[0] = self.coord
        else:
            # prev row, same col
            action_state_transitions[0] = (self.coord[0]-1, self.coord[1])

        # Action 1 - right
        if self._isLastColState():
            action_state_transitions[1] = self.coord
        else:
            # same row, next col
            action_state_transitions[1] = (self.coord[0], self.coord[1]+1)

        # Action 2 - down
        if self._isLastRowState():
            action_state_transitions[2] = self.coord
        else:
            # next row, same col
            action_state_transitions[2] = (self.coord[0]+1, self.coord[1])

        # Action 3 - left
        if self._isFirstRowState():
            action_state_transitions[3] = self.coord
        else:
            # same row, prev col
            action_state_transitions[3] = (self.coord[0], self.coord[1]-1)

        return action_state_transitions

    def _isFirstRowState(self):
        return self.coord[0] == 0

    def _isLastRowState(self):
        return self.coord[0] == 3

    def _isFirstColState(self):
        return self.coord[1] == 0

    def _isLastColState(self):
        return self.coord[1] == 3

    # Returns if the current state is a terminal state
    def isTerminal(self):
        return self.is_terminal

    # Gets the action required to move the agent from the current state
    # to some state s2. If the agent cannot move to s2 it returns None
    def getActionTransiton(self, s2):
        for action, next_state in self.action_state_transitions.items():
            if next_state == s2.coord:
                return action
        return None

    # Returns the likelihood of ending up in state s_prime after taking
    # action a from the current state
    def getNextStateLikelihood(self, a, s_prime):
        if self.action_state_transitions[a] == s_prime.coord:
            return 1
        else:
            return 0

    # Returrn the reward for stepping into this state
    def getReward(self):
        return self.reward


class DynamicProgrammingAgent(object):
    """
    Base implementation of a Dynamic Programming Agent for the Grid World Problem

    env: Gym env the agent will be trained on
    """
    def __init__(self, gamma):
        self.gamma = gamma

        # of states and actions for the grid world problem
        self.num_states = 16
        self.num_actions = 4

    # Prints the values of each state on the grid
    def _printStateValues(self, V):
        grid = np.zeros([4,4])

        for state, value in V.items():
            x = state.coord[0]
            y = state.coord[1]
            grid[x,y] = value

        print("Value Function--------------------------")
        pprint.pprint(grid)
        print('\n')

    # Prints the policy as a grid of arrows
    def _printPolicy(self, pi):
        grid = np.zeros([4,4])

        for state, actions in pi.items():
            x = state.coord[0]
            y = state.coord[1]
            action = np.argmax(actions)
            grid[x,y] = action

        # Convert actions to arrows
        arrow_grid = []
        for row_index, row in enumerate(grid):
            arrow_grid_row = []
            for col_index, action in enumerate(row):
                arrow_char = ''
                if (row_index == 0 and col_index == 0) or (row_index == 3 and col_index == 3):
                    arrow_grid_row.append(arrow_char)
                else:
                    if action == 0:
                        arrow_char = '↑'
                    elif action == 1:
                        arrow_char = '→'
                    elif action == 2:
                        arrow_char = '↓'
                    elif action == 3:
                        arrow_char = '←'
                    arrow_grid_row.append(arrow_char)
            arrow_grid.append(arrow_grid_row)

        print("Policy--------------------------")
        pprint.pprint(arrow_grid)
        print('\n')

    # # Initialize the states (S), state value function (V), and the policy (pi)
    def initSVAndPi(self):
        self.S = []
        V = {}
        pi = {}
        for r in range(4):
            for c in range(4):
                # Create the state
                is_terminal = False
                if (r == 0 and c == 0) or (r == 3 and c == 3):
                    is_terminal = True
                s = State((r,c,), is_terminal)
                self.S.append(s)
                # Initialize the value of every state to 0
                V[s] = 0
                # Begin with a policy that selects every  action with equal probability
                pi[s] = self.num_actions * [0.25]
        return V, pi

    # Gets the action values for a state by getting the expected return
    # of taking each action
    def getActionValuesForState(self, s, V):
        action_values = []
        for action in range(self.num_actions):
            action_value = 0
            for s_prime in self.S:
                p = s.getNextStateLikelihood(action, s_prime)
                action_value += p * (s_prime.getReward() + self.gamma * V[s_prime])
            action_values.append(action_value)
        return action_values


class PolicyIterationAgent(DynamicProgrammingAgent):
    """
    Implementation of an agent that uses policy iteration to derive
    the optimal policy (pi*) and state value function (v*) on the 4x4 grid world.

    gamma: discount factor
    """
    def __init__(self, gamma):
        # Call base class constructor
        super().__init__(gamma)

    def policyIterate(self):
        V, pi = self.initSVAndPi()

        policy_stable = False
        i = 1
        while not policy_stable:
            print("Policy Iteration", i)
            V = self._iterPolicyEval(pi, V)
            pi, V, policy_stable = self.policyImprove(pi, V)
            self._printPolicy(pi)
            i += 1

    # Evaluate the policy pi by iteratively updating it's state value
    # function V
    def _iterPolicyEval(self, pi, V):
        # threshold for determing the end of the policy eval loop
        theta = 0.01

        while True:
            # change in state values
            delta = 0

            for s in self.S:
                # current state value
                v = V[s]

                # set the new state value
                V[s] = 0

                if s.isTerminal():
                    continue

                for s_prime in self.S:
                    p = self._p(s, s_prime, pi)
                    V[s] += p * (s_prime.getReward() + self.gamma * V[s_prime])

                delta = max(delta, abs(v - V[s]))

            self._printStateValues(V)

            if delta < theta:
                break

        return V

    # Gets the likelihood of ending up in s_prime from s under the
    # current policy pi using a perfect model of the environment
    def _p(self, s, s_prime, pi):
        # Get the action that would take the agent from s to s_prime
        transition_action = s.getActionTransiton(s_prime)

        # if the agent cannot move from s to s_prime it would never be selected
        if transition_action == None:
            return 0

        # return probability of selecting this action under the policy
        return pi[s][transition_action]

    # Improve the policy pi by making it greedy with respect to it's
    # value function V
    def policyImprove(self, pi, V):
        policy_stable = True

        for s in self.S:
            old_best_action = np.argmax(pi[s])
            # print(old_action)

            if s.isTerminal():
                continue

            action_values = self.getActionValuesForState(s, V)

            new_best_action = np.argmax(action_values)

            # Set the likelihood of selecting the new best action in the policy to 1
            # for all other actions make it 0
            for action in range(self.num_actions):
                if action != new_best_action:
                    pi[s][action] = 0
                else:
                    pi[s][action] = 1

            if old_best_action != new_best_action:
                policy_stable = False

        return pi, V, policy_stable


class ValueIterationAgent(DynamicProgrammingAgent):
    """
    Implementation of an agent that uses value iteration to derive
    the optimal policy (pi*) and state value function (v*) on the 4x4 grid world.

    gamma: discount factor
    """
    def __init__(self, gamma):
        # Call base class constructor
        super().__init__(gamma)

    def valueIterate(self):
        V, pi = self.initSVAndPi()

        # threshold for determing the end of the policy eval loop
        theta = 0.01

        i = 1
        while True:
            print("Value Iteration", i)
            # change in state values
            delta = 0

            for s in self.S:
                # current state value
                v = V[s]

                # set the new state value
                V[s] = 0

                if s.isTerminal():
                    continue

                action_values = self.getActionValuesForState(s, V)

                V[s] = max(action_values)
                new_best_action = np.argmax(action_values)

                # Set the likelihood of selecting the new best action in the policy to 1
                # for all other actions make it 0
                for action in range(self.num_actions):
                    if action != new_best_action:
                        pi[s][action] = 0
                    else:
                        pi[s][action] = 1

                delta = max(delta, abs(v - V[s]))

            self._printStateValues(V)

            if delta < theta:
                break

            i += 1

        self._printPolicy(pi)
