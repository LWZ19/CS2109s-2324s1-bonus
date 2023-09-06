graph = {
    'S': {('A', 1), ('B', 5),  ('C', 15)},
    'A': {('G', 10), ('S', 1)},
    'B': {('G', 5), ('S', 5)},
    'C': {('G', 5), ('S', 15)},
    'G': set()
}

from priority_queue import PriorityQueue
from collections import defaultdict


def print_frontier(frontier):
    frontier_temp = []
    while len(frontier) != 0:
        frontier_temp.append(frontier.pop())
    frontier_str = ''
    for cost, state in frontier_temp:
        frontier_str += state + '(' + str(cost) + ') '
        frontier.append(state, cost)
    print(frontier_str)


# Return the path found
def uniform_cost_search(graph, inital_node, goal_test, is_tree, is_update):

    frontier = PriorityQueue('min')
    frontier.append(inital_node, 0)
    visited = set()
    parents = defaultdict()
    parents[(0, inital_node)] = None
    curr = None
    path = ''
    while len(frontier) != 0:
        print_frontier(frontier)
        curr_cost, state = frontier.pop()
        if goal_test(state):
            curr = (curr_cost, state)
            break
        if not is_tree and state not in visited:
            visited.add(state)

        for next_state, cost in graph[state]:
            next_cost = curr_cost + cost
            if not is_tree and next_state in visited:
                continue

            if is_update and next_state in frontier:
                prev_cost = frontier[next_state]
                if prev_cost > next_cost:
                    del frontier[next_state]
                    del parents[(prev_cost, next_state)]
                    parents[(next_cost, next_state)] = (curr_cost, state)
                else:
                    continue

            if (next_cost, next_state) not in parents:
                parents[(next_cost, next_state)] = (curr_cost, state)
            frontier.append(next_state, next_cost)

    if curr is not None:
        while curr is not None:
            path = curr[1] + path
            curr = parents[curr]
        return path
    else:
        return False


print("=====")
print("Tree")
print("=====")
print(p:=uniform_cost_search(graph, 'S', lambda n: n=='G', is_tree=True, is_update=False))
assert(p=="SBG")

print("=====")
print("Graph")
print("=====")
print(p:=uniform_cost_search(graph, 'S', lambda n: n=='G', is_tree=False, is_update=True))
assert(p=="SBG")
