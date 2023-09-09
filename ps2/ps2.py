""" CS2109S Problem Set 2: Informed Search"""

from collections import deque, namedtuple
import copy
import heapq
import itertools
import math
import random
import time
from typing import List, Tuple
import numpy as np

import cube
import utils

""" ADD HELPER FUNCTION HERE """

"""
We provide implementations for the Node and PriorityQueue classes in utils.py, but you can implement your own if you wish
"""
from utils import Node
from utils import PriorityQueue


#TODO Task 1.1: Implement your heuristic function, which takes in an instance of the Cube and
#   the State class and returns the estimated cost of reaching the goal state from the state given.

# One possible heuristic: sum of the number of tiles that are not in the correct position
# EDIT: THIS IS NOT ADMISSIBLE
# COUNTER EXAMPLE:
# Goal:
# 123
# 456
# 789

# Current:
# 372
# 156
# 489

# Actions: [{0, "left"}, {0, "up"}]
# Cost: 2
# min(rows_diff, cols_diff) = 3
def heuristic_func2(problem: cube.Cube, state: cube.State) -> float:
    r"""
    Computes the heuristic value of a state
    
    Args:
        problem (cube.Cube): the problem to compute
        state (cube.State): the state to be evaluated
        
    Returns:
        h_n (float): the heuristic value 
    """
    h_n = 0.0
    goals = problem.goal
    shape = goals.shape
    # Convert the flat layouts to 2D numpy arrays
    goal_array = np.array(goals.layout).reshape(shape)
    state_array = np.array(state.layout).reshape(shape)

    # We look the the number of rows in the state that are different from the goal, same with columns
    different_rows = np.sum(np.any(state_array != goal_array, axis=1))
    different_columns = np.sum(np.any(state_array != goal_array, axis=0))
    # The inspiration behind this heuristic is the example given:
    #  State  Goal
    #  RRR    BRR
    #  GGG    RGG
    #  BBB    GBB
    # We cannot use number of tiles not aligned, since that would give us a cost of 3, which is an overestimate of the true
    # cost of 1 needed to just rotate the left column up
    # We can then generalize this. For this example, the heuristic cannot be more than 1 for it to be admissible.
    # By eyepowering, we see that the easiest way is to take the number of columns out of sync
    # Of course, we can also generalize this to the rows
    # In this case the are 3 rows out of sync, and since heuristics cannot be more than the true cost,
    # We take the min of columns and rows out of sync
    h_n = min(different_rows, different_columns)
    return h_n

# For example:
# Goal:
# RRR
# GGG
# BBB
# 
# Current:
# GRR
# BGG
# RBB
# 
# sum of distance to nearest correct position (1 + 1 + 1)
# Explanation: the nearest B in the goal state is at position (0, 0). The misaligned B is at position (1, 0). Distance is 1. Same reasoning for R and G.
# total misaligned tiles (RBG of left column so total 3)
# heuristic = 3/3 = 1
def heuristic_func(problem: cube.Cube, state: cube.State) -> float:
    goals = problem.goal
    shape = goals.shape
    goal_array = np.array(goals.layout).reshape(shape)
    state_array = np.array(state.layout).reshape(shape)
    
    total_distance = 0
    misaligned_count = 0
    
    for i in range(shape[0]):
        for j in range(shape[1]):
            if state_array[i, j] != goal_array[i, j]:
                correct_positions = np.argwhere(goal_array == state_array[i, j])
                distances = [abs(i-x) + abs(j-y) for x, y in correct_positions]
                min_distance = min(distances)
                total_distance += min_distance
                misaligned_count += 1

    if misaligned_count == 0:
        return 0.0

    h_n = total_distance / misaligned_count
    return h_n

# Test
def wrap_test(func):
    def inner(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return f'FAILED, error: {type(e).__name__}, reason: {str(e)}'
    return inner

# Test case for Task 1.1
@wrap_test
def test_heuristic(case):

    input_dict = case['input_dict']
    answer = case['answer']
    problem = cube.Cube(input_dict = input_dict)

    assert heuristic_func(problem, problem.goal) == 0, "Heuristic is not 0 at the goal state"
    assert heuristic_func(problem, problem.initial) <= answer['cost'], "Heuristic is not admissible"

    return "PASSED"

if __name__ == '__main__':
    
    cube1 = {'input_dict': {"initial": {'shape': [3, 3], 'layout': ['N', 'U',   
        'S', 'N','U', 'S', 'N', 'U', 'S']}, 'goal': {'shape': [3, 3], 'layout': 
        ['N', 'U', 'S', 'N', 'U', 'S', 'N', 'U', 'S']}}, 'answer': {'solution': 
        [], 'cost': 0}}

    cube2 = {'input_dict': {"initial": {'shape': [3, 3], 'layout': ['S', 'O', 
        'C', 'S', 'O', 'C', 'S', 'O', 'C']}, 'goal': {'shape': [3, 3], 
        'layout': ['S', 'S', 'S', 'O', 'O', 'O', 'C', 'C', 'C']}}, 'answer': 
        {'solution': [[2, 'right'], [1, 'left'], [1, 'down'], 
        [2, 'up']], 'cost': 4}}

    cube3 = {'input_dict': {"initial": {'shape': [3, 3], 'layout': ['N', 'U',   
        'S', 'N', 'U', 'S', 'N', 'U', 'S']}, 'goal': {'shape': [3, 3], 'layout': 
        ['S', 'U', 'N', 'N', 'S', 'U', 'U', 'N', 'S']}}, 'answer': {'solution': 
        [[0, 'left'], [1, 'right'], [0, 'up'], [1, 'down']], 'cost': 4}}

    cube4 = {'input_dict':{"initial": {'shape': [3, 4], 'layout': [1, 1, 9, 0,
        2, 2, 0, 2, 9, 0, 1, 9]}, 'goal': {'shape': [3, 4], 'layout': [ 1, 0,
        9, 2, 2, 1, 0, 9, 2, 1, 0, 9]}}, 'answer': {'solution': [[1, 'down'],
        [3, 'up'], [2, 'left']], 'cost': 3}}

    print('Task 1.1:')
    print('cube1: ' + test_heuristic(cube1))
    print('cube2: ' + test_heuristic(cube2))
    print('cube3: ' + test_heuristic(cube3))
    print('cube4: ' + test_heuristic(cube4))
    print('\n')

Node = namedtuple('Node', ['state', 'parent', 'action', 'cost'])
State = namedtuple('State', ['layout', 'shape'])
#TODO Task 1.2: Implement A* search which takes in an instance of the Cube 
#   class and returns a list of actions [(2,'left'), ...] from the provided action set.
def astar_search(problem: cube.Cube):
    r"""
    A* Search finds the solution to reach the goal from the initial.
    If no solution is found, return False.
    
    Args:
        problem (cube.Cube): Cube instance

    Returns:
        solution (List[Action]): the action sequence
    """
    fail = True
    solution = []
    pq = []
    counter = 0
    heapq.heapify(pq)
    initial_node = Node(state=problem.initial, parent=None, action=None, cost=0)
    heapq.heappush(pq, (initial_node.cost, counter, initial_node))
    counter += 1
    shape = problem.initial.shape
    r, c = shape

    # possible actions
    # {0, "left"}, {0, "right"}, {1, "left"}, {1, "right"}, {2, "left"}, {2, "right"} .... {r, "left"}, {r, "right"}
    # {0, "up"}, {0, "down"}, {1, "up"}, {1, "down"}, {2, "up"}, {2, "down"} .... {c, "up"}, {c, "down"}
    # There would be total of 2 * (r + c) actions
    

    while pq:
        cur_node =  heapq.heappop(pq)[2]
        if np.array_equal(np.array(cur_node.state.layout), np.array(problem.goal.layout)):
            fail = False
            return get_solution(cur_node)
        for i in range(r):
            for direction in ["left", "right"]:
                next_state = rubik_transition(cur_node.state, (i, direction), problem.initial.shape)
                cost = heuristic_func(problem, next_state) + cur_node.cost + 1
                next_node = Node(state=next_state, parent=cur_node, action=(i, direction), cost=cost)
                print(next_state)
                print(next_node.action)
                print(next_node.cost)
                heapq.heappush(pq, (next_node.cost, counter, next_node))
                counter += 1
        for j in range(c):
            for direction in ["up", "down"]:
                print("passed")
                next_state = rubik_transition(cur_node.state, (j, direction), problem.initial.shape)
                cost = heuristic_func(problem, next_state) + cur_node.cost + 1
                next_node = Node(state=next_state, parent=cur_node, action=(j, direction), cost=cost)
                print(next_state)
                print(next_node.action)
                print(next_node.cost)
                heapq.heappush(pq, (next_node.cost, counter, next_node))
                counter += 1
    if fail:
        return False
    return solution

def rubik_transition(state, action, shape):
    state_array = np.array(state.layout).reshape(shape)
    idx = action[0]
    # consider all cases of action using switch case below
    match action[1]:
        case "left":
            # turn the idx-th row left
            state_array[idx, :] = np.roll(state_array[idx, :], -1)
        case "right":
            # turn the idx-th row right
            state_array[idx, :] = np.roll(state_array[idx, :], 1)
        case "up":
            # turn the idx-th col up
            state_array[:, idx] = np.roll(state_array[:, idx], -1)
        case "down":
            # turn the idx-th col down
            state_array[:, idx] = np.roll(state_array[:, idx], 1)
    
    print(state_array)

    flattened_array = state_array.flatten()
    return State(layout=flattened_array, shape=shape)

def get_solution(node):
    solution = []
    while node.parent:
        solution.append(node.action)
        node = node.parent
    solution.reverse()
    print(solution)
    return solution

@wrap_test
def test_astar(case):

    input_dict = case['input_dict']
    answer = case['answer']
    problem = cube.Cube(input_dict = input_dict)
    
    start = time.time()
    solution = astar_search(problem)
    print(f"Time lapsed: {time.time() - start}")

    if solution is False:
        assert answer['solution'] is False, "Solution is not False"
    else:
        correctness, cost = problem.verify_solution(solution, _print=False)
        assert correctness, f"Fail to reach goal state with solution {solution}"
        assert cost <= answer['cost'], f"Cost is not optimal."
    return "PASSED"


if __name__ == '__main__':
    
    cube1 = {'input_dict': {"initial": {'shape': [3, 3], 'layout': ['N', 'U',   
        'S', 'N','U', 'S', 'N', 'U', 'S']}, 'goal': {'shape': [3, 3], 'layout': 
        ['N', 'U', 'S', 'N', 'U', 'S', 'N', 'U', 'S']}}, 'answer': {'solution': 
        [], 'cost': 0}}

    cube2 = {'input_dict': {"initial": {'shape': [3, 3], 'layout': ['S', 'O', 
        'C', 'S', 'O', 'C', 'S', 'O', 'C']}, 'goal': {'shape': [3, 3], 
        'layout': ['S', 'S', 'S', 'O', 'O', 'O', 'C', 'C', 'C']}}, 'answer': 
        {'solution': [[2, 'right'], [1, 'left'], [1, 'down'], 
        [2, 'up']], 'cost': 4}}

    cube3 = {'input_dict': {"initial": {'shape': [3, 3], 'layout': ['N', 'U',   
        'S', 'N', 'U', 'S', 'N', 'U', 'S']}, 'goal': {'shape': [3, 3], 'layout': 
        ['S', 'U', 'N', 'N', 'S', 'U', 'U', 'N', 'S']}}, 'answer': {'solution': 
        [[0, 'left'], [1, 'right'], [0, 'up'], [1, 'down']], 'cost': 4}}

    cube4 = {'input_dict':{"initial": {'shape': [3, 4], 'layout': [1, 1, 9, 0,
        2, 2, 0, 2, 9, 0, 1, 9]}, 'goal': {'shape': [3, 4], 'layout': [ 1, 0,
        9, 2, 2, 1, 0, 9, 2, 1, 0, 9]}}, 'answer': {'solution': [[1, 'down'],
        [3, 'up'], [2, 'left']], 'cost': 3}}

    #print('Task 1.2:')
    #print('cube1: ' + test_astar(cube1)) 
    #print('cube2: ' + test_astar(cube2)) 
    #print('cube3: ' + test_astar(cube3)) 
    #print('cube4: ' + test_astar(cube4)) 
    #print('\n')


#TODO Task 1.3: Explain why the heuristic you designed for Task 1.1 is {consistent} 
#   and {admissible}.


#TODO Task 2.1: Propose a state representation for this problem if we want to formulate it
#   as a local search problem.


#TODO Task 2.2: What are the initial and goal states under your proposed representation?


#TODO Task 2.3: Implement a reasonable transition function to generate new routes by applying
#   minor "tweaks" to the current route. It should return a list of new routes to be used in 
#   the next iteration in the hill-climbing algorithm.
def transition(route: List[int]):
    r"""
    Generates new routes to be used in the next iteration in the hill-climbing algorithm.

    Args:        
        route (List[int]): The current route as a list of cities in the order of travel

    Returns:
        new_routes (List[List[int]]): New routes to be considered
    """
    new_routes = []
    for _ in range(len(route)):
        new_route = copy.deepcopy(route)
        idx1, idx2 = random.sample(range(len(route)), 2)
        new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
        new_routes.append(new_route)

    return new_routes

# Test
@wrap_test
def test_transition(route: List[int]):
    for new_route in transition(route):
        assert sorted(new_route) == list(range(len(route))), "Invalid route"

    return "PASSED"

if __name__ == '__main__':
    print('Task 2.3:')
    print(test_transition([1, 3, 2, 0]))
    print(test_transition([7, 8, 6, 3, 5, 4, 9, 2, 0, 1]))
    print('\n')


#TODO Task 2.4: Implement an evaluation function `evaluation_func(cities, distances, route)` that 
#   would be helpful in deciding on the "goodness" of a route, i.e. an optimal route should 
#   return a higher evaluation score than a suboptimal one.
def evaluation_func(cities: int, distances: List[Tuple[int]], route: List[int]) -> float:
    r"""
    Computes the evaluation score of a route

    Args:
        cities (int): The number of cities to be visited

        distances (List[Tuple[int]]): The list of distances between every two cities
            Each distance is represented as a tuple in the form of (c1, c2, d), where
                c1 and c2 are the two cities and d is the distance between them.
            The length of the list should be equal to cities * (cities - 1)/2.

        route (List[int]): The current route as a list of cities in the order of travel

    Returns:
        h_n (float): the evaluation score
    """
    h_n = 0.0
    dict = {}
    for distance in distances:
        dict[(distance[0], distance[1])] = distance[2]
    for i in range(0, len(route), 1):
        if i == len(route) - 1:
            cur_city, next_city = route[i], route[0]
        else:
            cur_city, next_city = route[i], route[i+1]
        if (cur_city, next_city) in dict:
            dist = dict[(cur_city, next_city)]
        else:
            dist = dict[(next_city, cur_city)]
        h_n += dist

    return -h_n

if __name__ == '__main__':
    cities = 4
    distances = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]

    route_1 = evaluation_func(cities, distances, [0, 1, 2, 3])
    route_2 = evaluation_func(cities, distances, [2, 1, 3, 0])
    route_3 = evaluation_func(cities, distances, [1, 3, 2, 0])

    print('Task 2.4:')
    print(route_1 == route_2)  # True
    print(route_1 > route_3)  # True
    print('\n')


#TODO Task 2.5: Explain why your evaluation function is suitable for this problem.


#TODO Task 2.6: Implement hill-climbing which takes in the number of cities and the list of
#   distances, and returns the shortest route as a list of cities.
def hill_climbing(cities: int, distances: List[Tuple[int]]):
    r"""
    Hill climbing finds the solution to reach the goal from the initial.
    
    Args:
        cities (int): The number of cities to be visited
        
        distances (List[Tuple[int]]): The list of distances between every two cities
            Each distance is represented as a tuple in the form of (c1, c2, d), where
                c1 and c2 are the two cities and d is the distance between them.
            The length of the list should be equal to cities * (cities - 1)/2.

    Returns:
        route (List[int]): The shortest route, represented by a list of cities
            in the order to be traversed.
    """

    route = list(range(cities))
    random.shuffle(route)
    print(route)
    cur_max = evaluation_func(cities, distances, route)
    routes = transition(list(range(cities)))
    for r in routes:
        cur = evaluation_func(cities, distances, r)
        if cur > cur_max:
            cur_max = cur
            route = r
    
    return route

# Test
@wrap_test
def test_hill_climbing(cities: int, distances: List[Tuple[int]]):
    start = time.time()
    route = hill_climbing(cities, distances)
    print(f"Time lapsed: {time.time() - start}")

    assert sorted(route) == list(range(cities)), "Invalid route"

    return "PASSED"

if __name__ == '__main__':
    
    cities_1 = 4
    distances_1 = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]

    cities_2 = 10
    distances_2 = [(2, 7, 60), (1, 6, 20), (5, 4, 70), (9, 8, 90), (3, 7, 54), (2, 5, 61),
        (4, 1, 106), (0, 6, 51), (3, 1, 45), (0, 5, 86), (9, 2, 73), (8, 4, 14), (0, 1, 51),
        (9, 7, 22), (3, 2, 22), (8, 1, 120), (5, 7, 92), (5, 6, 60), (6, 2, 10), (8, 3, 78),
        (9, 6, 82), (0, 2, 41), (2, 8, 99), (7, 8, 71), (0, 9, 32), (4, 0, 73), (0, 3, 42),
        (9, 1, 80), (4, 2, 85), (5, 9, 113), (3, 6, 28), (5, 8, 81), (3, 9, 72), (9, 4, 81),
        (5, 3, 45), (7, 4, 60), (6, 8, 106), (0, 8, 85), (4, 6, 92), (7, 6, 70), (7, 0, 22),
        (7, 1, 73), (4, 3, 64), (5, 1, 80), (2, 1, 22)]

    print('Task 2.6:')
    print('cities_1: ' + test_hill_climbing(cities_1, distances_1))
    print('cities_2: ' + test_hill_climbing(cities_2, distances_2))
    print('\n')


#TODO Task 2.7: Implement hill_climbing_with_random_restarts by repeating hill climbing
#   at different random locations.
def hill_climbing_with_random_restarts(cities: int, distances: List[Tuple[int]], repeats: int = 10):
    r"""
    Hill climbing with random restarts finds the solution to reach the goal from the initial.
    
    Args:
        cities (int): The number of cities to be visited
        
        distances (List[Tuple[int]]): The list of distances between every two cities
            Each distance is represented as a tuple in the form of (c1, c2, d), where
                c1 and c2 are the two cities and d is the distance between them.
            The length of the list should be equal to cities * (cities - 1)/2.

        repeats (int): The number of times hill climbing to be repeated. The default
            value is 10.

    Returns:
        route (List[int]): The shortest route, represented by a list of cities
            in the order to be traversed.
    """

    route = []
    max_score = float('-infinity')
    for _ in range(repeats):
        cur_route = hill_climbing(cities, distances)
        eval_score = evaluation_func(cities, distances, cur_route)
        if eval_score > max_score:
            max_score = eval_score
            route = cur_route
    """ YOUR CODE HERE """

    """ END YOUR CODE HERE """
    
    return route

# Test
@wrap_test
def test_random_restarts(cities: int, distances: List[Tuple[int]], repeats: int = 10):
    start = time.time()
    route = hill_climbing_with_random_restarts(cities, distances, repeats)
    print(f"Time lapsed: {time.time() - start}")

    assert sorted(route) == list(range(cities)), "Invalid route"

    return "PASSED"

if __name__ == '__main__':
    
    cities_1 = 4
    distances_1 = [(1, 0, 10), (0, 3, 22), (2, 1, 8), (2, 3, 30), (1, 3, 25), (0, 2, 15)]

    cities_2 = 10
    distances_2 = [(2, 7, 60), (1, 6, 20), (5, 4, 70), (9, 8, 90), (3, 7, 54), (2, 5, 61),
        (4, 1, 106), (0, 6, 51), (3, 1, 45), (0, 5, 86), (9, 2, 73), (8, 4, 14), (0, 1, 51),
        (9, 7, 22), (3, 2, 22), (8, 1, 120), (5, 7, 92), (5, 6, 60), (6, 2, 10), (8, 3, 78),
        (9, 6, 82), (0, 2, 41), (2, 8, 99), (7, 8, 71), (0, 9, 32), (4, 0, 73), (0, 3, 42),
        (9, 1, 80), (4, 2, 85), (5, 9, 113), (3, 6, 28), (5, 8, 81), (3, 9, 72), (9, 4, 81),
        (5, 3, 45), (7, 4, 60), (6, 8, 106), (0, 8, 85), (4, 6, 92), (7, 6, 70), (7, 0, 22),
        (7, 1, 73), (4, 3, 64), (5, 1, 80), (2, 1, 22)]

    print('Task 2.7:')
    print('cities_1: ' + test_random_restarts(cities_1, distances_1))
    print('cities_2: ' + test_random_restarts(cities_2, distances_2, 20))


#TODO Task 2.8: Compared to previous search algorithms you have seen (uninformed search, A*), 
#   why do you think local search is more suitable for this problem?
