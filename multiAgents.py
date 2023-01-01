# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # distance to ghosts
        ghost_positions = [ghostState.getPosition() for ghostState in newGhostStates]
        ghost_distances = [util.manhattanDistance(position, newPos) for position in ghost_positions]
        min_ghost_distance = min(ghost_distances)
        # distance to food
        food_distances = [util.manhattanDistance(position, newPos) for position in newFood.asList()]
        max_food_distance = max(food_distances) if len(food_distances) != 0 else 1

        # no stopping
        stop = -1000 if action == "Stop" else 0

        return childGameState.getScore() + min_ghost_distance - max_food_distance + stop


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    def get_value_action(self, gameState, index, num_moves, alpha=None, beta=None):
        # if terminal state --> maximum depth reached or we are on a leaf (self.depth == 0)
        if num_moves == 0 or gameState.getLegalActions(index) == []:
            return self.evaluationFunction(gameState), ""
        # if agent is pacman (index == 0)
        if index == 0:
            return self.pacman_value_action(gameState, index, num_moves - 1, alpha, beta)
        # if agent is ghost (index >= 1)
        else:
            return self.ghost_value_action(gameState, index, num_moves - 1, alpha, beta)

    def pacman_value_action(self, gameState, index, num_moves, alpha=None, beta=None):
        pacman_value = - math.inf
        pacman_action = None
        legal_actions = gameState.getLegalActions(index)
        for action in legal_actions:
            succesor_state = gameState.getNextState(index, action)
            succesor_player = self.next_player(index, gameState.getNumAgents())
            succesor_value, succesor_action = self.get_value_action(succesor_state, succesor_player, num_moves, alpha,
                                                                    beta)
            if succesor_value > pacman_value:
                pacman_value = succesor_value
                pacman_action = action

            # In case we are running an expectimax
            if None not in (alpha, beta):
                if pacman_value > beta:
                    return pacman_value, pacman_action

                alpha = max(alpha, pacman_value)

        return pacman_value, pacman_action

    def ghost_value_action(self, gameState, index, num_moves, alpha=None, beta=None):
        ghost_value = math.inf
        ghost_action = None
        legal_actions = gameState.getLegalActions(index)

        for action in legal_actions:
            succesor_state = gameState.getNextState(index, action)
            succesor_player = self.next_player(index, gameState.getNumAgents())
            succesor_value, succesor_action = self.get_value_action(succesor_state, succesor_player, num_moves, alpha,
                                                                    beta)

            if succesor_value < ghost_value:
                ghost_value = succesor_value
                ghost_action = action

            # In case we are running an expectimax
            if None not in (alpha, beta):
                if ghost_value < alpha:
                    return ghost_value, ghost_action
                beta = min(beta, ghost_value)

        return ghost_value, ghost_action

    def next_player(self, current_player, num_players):
        return (current_player + 1) % num_players


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        num_moves = self.depth * gameState.getNumAgents()
        minimax_value, minimax_action = self.get_value_action(gameState, self.index,
                                                              num_moves)  # self.get_minimax_action(gameState, self.index, num_moves)
        return minimax_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_moves = (self.depth * gameState.getNumAgents())
        minimax_value, minimax_action = self.get_value_action(gameState, self.index, num_moves, -math.inf, math.inf)
        return minimax_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_moves = (self.depth * gameState.getNumAgents())
        expectimax_value, expectimax_action = self.get_value_action(gameState, self.index, num_moves)
        return expectimax_action


    def ghost_value_action(self, gameState, index, num_moves, alpha=None, beta=None):
        """Return the weighted average (expectation) of children"""
        expected_value = 0
        legal_actions = gameState.getLegalActions(index)
        probability = 1 / len(legal_actions)
        for action in legal_actions:
            succesor_state = gameState.getNextState(index, action)
            succesor_player = self.next_player(index, gameState.getNumAgents())
            succesor_value, succesor_action = self.get_value_action(succesor_state, succesor_player, num_moves)
            expected_value += probability * succesor_value

        return expected_value, ""


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    food_positions = currentGameState.getFood()
    pacman_position = currentGameState.getPacmanPosition()
    food_distances = [util.manhattanDistance(food_position, pacman_position) for food_position in
                      food_positions.asList()]
    min_food_distance = min(food_distances) if len(food_distances) != 0 else 1

    return currentGameState.getScore() - min_food_distance


# Abbreviation
better = betterEvaluationFunction
