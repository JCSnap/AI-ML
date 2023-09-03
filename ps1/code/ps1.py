from collections import deque, namedtuple
from enum import Enum

class Position(Enum):
    LEFT = "left"
    RIGHT = "right"

Node = namedtuple('Node', ['state', 'parent', 'action_from_parent', 'depth'])

# Task 1.6
def mnc_tree_search(m, c):
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    actions = [(1, 0), (2, 0), (1, 1), (0, 1), (0, 2)]
    upper_bound = 2 * (m + 1) + (c + 1)
    left_state = (m, c, Position.LEFT)
    queue = deque([Node(left_state, None, None, 0)])

    while queue:
        node = queue.popleft()
        left_m, left_c, boat = node.state
        cur_depth = node.depth

        if cur_depth >= upper_bound:
            print("Upper bound reached")
            break

        if left_m == 0 and left_c == 0 and boat == Position.RIGHT:
            return get_path(node)

        for action in actions:
            new_state = get_new_state(node.state, action)
            if is_valid(new_state[0], new_state[1], m - new_state[0], c - new_state[1]):
                new_node = Node(new_state, node, action, cur_depth + 1)
                queue.append(new_node)
    return False

def is_valid(left_m, left_c, right_m, right_c):
    return all(val >= 0 for val in [left_m, left_c, right_m, right_c]) and \
           (left_m >= left_c or left_m == 0) and \
           (right_m >= right_c or right_m == 0)

def get_path(node):
    path = []
    while node.parent:
        path.insert(0, node.action_from_parent)
        node = node.parent
    print(path)
    return tuple(path)

def get_new_state(state, action):
    if state[2] == Position.LEFT:
        return (state[0] - action[0], state[1] - action[1], Position.RIGHT)
    else:
        return (state[0] + action[0], state[1] + action[1], Position.LEFT)

# Test cases for Task 1.6
def test_16():
    expected = ((2, 0), (1, 0), (1, 1))
    #assert(mnc_tree_search(2,1) == expected)

    expected = ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    #assert(mnc_tree_search(2,2) == expected)

    #expected = ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    #assert(mnc_tree_search(3,3) == expected)   

    assert(mnc_tree_search(4, 4) == False)

#test_16()

# Task 1.7
def mnc_graph_search(m, c):
    '''
    Graph search requires to deal with the redundant path: cycle or loopy path.
    Modify the above implemented tree search algorithm to accelerate your AI.

    Parameters
    ----------    
    m: no. of missionaries
    c: no. of cannibals
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a tuple of two numbers x and y, indicating the number of missionaries and cannibals on the boat respectively as the boat moves from one side of the river to another. If there is no solution, return False.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    if m == 0 and c == 0:
        return ()
    actions = [(1, 0), (2, 0), (1, 1), (0, 1), (0, 2)]
    upper_bound = 2 * (m + 1) + (c + 1)
    left_state = (m, c, Position.LEFT)
    queue = deque([Node(left_state, None, None, 0)])
    visited = set()
    visited.add(left_state)

    while queue:
        node = queue.popleft()
        cur_depth = node.depth

        if cur_depth >= upper_bound:
            print("Upper bound reached")
            break

        for action in actions:
            new_state = get_new_state(node.state, action)
            if new_state in visited:
                continue
            if is_valid(new_state[0], new_state[1], m - new_state[0], c - new_state[1]) and new_state not in visited:
                new_node = Node(new_state, node, action, cur_depth + 1)
                if is_goal(new_state):
                    return get_path(new_node)
                queue.append(new_node)
                visited.add(new_state)
    return False

def is_goal(state):
    return state[0] == 0 and state[1] == 0 and state[2] == Position.RIGHT

# Test cases for Task 1.7
def test_17():
    # Your existing test cases
    expected = ((2, 0), (1, 0), (1, 1))
    assert(mnc_graph_search(2, 1) == expected)

    expected = ((1, 1), (1, 0), (2, 0), (1, 0), (1, 1))
    assert(mnc_graph_search(2, 2) == expected)

    expected = ((1, 1), (1, 0), (0, 2), (0, 1), (2, 0), (1, 1), (2, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    assert(mnc_graph_search(3, 3) == expected)

    assert(mnc_graph_search(4, 4) == False)
    
    assert(mnc_graph_search(3, 4) == False)

    # Edge Cases
    assert(mnc_graph_search(0, 0) == ())  # No missionaries or cannibals

    expected = ((2, 0),)  # Only missionaries, no cannibals
    print((mnc_graph_search(2, 0)))
    assert(mnc_graph_search(2, 0) == expected)

    expected = ((0, 2),)  # Only cannibals, no missionaries
    assert(mnc_graph_search(0, 2) == expected)

    expected = ((1, 1),)  # Equal number of missionaries and cannibals but fewer than 3
    assert(mnc_graph_search(1, 1) == expected)

    assert(mnc_graph_search(1, 2) == False)  # More cannibals than missionaries

#test_17()


Pitchers = namedtuple('Pitchers', ['state', 'parent', 'action_from_parent', 'depth'])
# Task 2.3
def pitcher_search(p1,p2,p3,a):
    '''
    Solution should be the action taken from the root node (initial state) to 
    the leaf node (goal state) in the search tree.

    Parameters
    ----------    
    p1: capacity of pitcher 1
    p2: capacity of pitcher 2
    p3: capacity of pitcher 3
    a: amount of water we want to measure
    
    Returns
    ----------    
    Returns the solution to the problem as a tuple of steps. Each step is a string: "Fill Pi", "Empty Pi", "Pi=>Pj". 
    If there is no solution, return False.
    '''
    # TODO: add your solution here and remove `raise NotImplementedError`
    initial_state = (0, 0, 0)
    # Each state change, we want to interate through filling each pitcher, emptying each pitcher, and pouring from one pitcher to another, considering all possible combinations
    actions = [fill, empty, pour]
    pitchers = [p1, p2, p3]
    upper_bound = (p1 + 1) + (p2 + 1) + (p3 + 1)
    queue = deque([Pitchers(initial_state, None, None, 0)])
    visited = set()
    visited.add(initial_state)
    # loop through actions
    while queue:
        node = queue.popleft()
        cur_state = node.state
        cur_depth = node.depth
        if a in cur_state:
            return get_pitcher_path(node)
        if cur_depth >= upper_bound:
            break
        for i in range(3):
            # Fill pitcher i
            new_state = fill(cur_state, i, pitchers[i])
            if new_state not in visited:
                queue.append(Pitchers(new_state, node, f"Fill P{i+1}", cur_depth + 1))
                visited.add(new_state)
            
            # Empty pitcher i
            new_state = empty(cur_state, i)
            if new_state not in visited:
                queue.append(Pitchers(new_state, node, f"Empty P{i+1}", cur_depth + 1))
                visited.add(new_state)
        
            # Pour from pitcher i to j
            for j in range(3):
                if i != j:
                    new_state = pour(cur_state, i, j, pitchers)
                    if new_state not in visited:
                        queue.append(Pitchers(new_state, node, f"P{i+1}=>P{j+1}", cur_depth + 1))
                        visited.add(new_state)
    return False

def fill(state, pitcher_index, capacity):
    state_list = list(state)
    state_list[pitcher_index] = capacity
    return tuple(state_list)

def empty(state, pitcher_index):
    state_list = list(state)
    state_list[pitcher_index] = 0
    return tuple(state_list)

def pour(state, from_index, to_index, capacities):
    state_list = list(state)
    total = state_list[from_index] + state_list[to_index]
    if total <= capacities[to_index]:
        state_list[to_index] = total
        state_list[from_index] = 0
    else:
        state_list[from_index] = total - capacities[to_index]
        state_list[to_index] = capacities[to_index]
    return tuple(state_list)

def get_pitcher_path(node):
    path = []
    while node.parent:
        path.insert(0, node.action_from_parent)
        node = node.parent
    return tuple(path)

# Test cases for Task 2.3
def test_23():
    expected = ('Fill P2', 'P2=>P1')
    assert(pitcher_search(2,3,4,1) == expected)

    expected = ('Fill P3', 'P3=>P1', 'Empty P1', 'P3=>P1')
    assert(pitcher_search(1,4,9,7) == expected)

    assert(pitcher_search(2,3,7,8) == False)

#test_23()

test_23()