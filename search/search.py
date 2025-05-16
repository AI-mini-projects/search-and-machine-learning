# search.py
# ---------
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

import util
import heapq


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def depthFirstSearch(problem):
    currentState = problem.getStartState()
    Path = []
    if problem.isGoalState(currentState):
        return Path

    frontier = []
    visited = []
    frontier.append((currentState, Path))
    while not len(frontier) == 0:
        currentState, Path = frontier.pop()
        if problem.isGoalState(currentState):
            return Path
        visited.append(currentState)
        for successors in problem.getSuccessors(currentState):
            if successors[0] not in visited:
                frontier.append((successors[0], Path + [successors[1]]))


def breadthFirstSearch(problem):
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def uniformCostSearch(problem):
    visited = {}
    frontier = util.PriorityQueue()
    current_state = problem.getStartState()
    frontier.push((current_state, [],0 ), 0)
    visited[current_state] = 1

    while not frontier.isEmpty():
        current_state, path, current_cost = frontier.pop()
        if problem.isGoalState(current_state):
            return path
        if visited[current_state] < current_cost:
            continue

        for state, action, new_cost in problem.getSuccessors(current_state):
            state_cost = new_cost + current_cost
            if state not in visited.keys() or visited[state] > state_cost:
                visited[state] = state_cost
                frontier.push((state, path+[action],state_cost), state_cost)
    return path


def nullHeuristic(state, problem=None):
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    startState = problem.getStartState()
    frontier = util.PriorityQueue()
    visited = {}

    frontier.push((startState, [], 0), heuristic(startState, problem))
    visited[startState] = 0

    while not frontier.isEmpty():
        currentState, path, current_cost = frontier.pop()

        if problem.isGoalState(currentState):
            return path

        if visited[currentState] < current_cost:
            continue

        for successor, action, step_cost in problem.getSuccessors(currentState):
            new_cost = current_cost + step_cost

            if successor not in visited or new_cost < visited[successor]:
                visited[successor] = new_cost
                priority = new_cost + heuristic(successor, problem)
                frontier.push((successor, path + [action], new_cost), priority)

    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
