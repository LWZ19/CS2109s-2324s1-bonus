from T01.priority_queue import PriorityQueue

h = {
    'S':7,
    'B':0,
    'A':3,
    'G':0
}
graph = {
    'S': {('A',2), ('B',4)},
    'A': {('S',2), ('B',1)},
    'B': {('S',4), ('A',1), ('G',4)},
    'G': {('B',4)},
}

def print_frontier(frontier):
    frontier_temp = []
    while len(frontier) != 0:
        frontier_temp.append(frontier.pop())
    frontier_str = ''
    for fn, state in frontier_temp:
        frontier_str += state[-1] + '(' + str(fn) + '-' + state[:-1] + ') '
        frontier.append(state, fn)
    print(frontier_str)

def astar(graph, inital_node, goal_test, heuristics, is_tree, is_update):
    frontier = PriorityQueue('min')
    frontier.append(inital_node, heuristics[inital_node])
    visited = set()
    while len(frontier) != 0:
        print_frontier(frontier)
        curr_fn, path = frontier.pop()
        if goal_test(path[-1]):
            return path
        if not is_tree and path[-1] not in visited:
            visited.add(path[-1])

        for next_state, cost in graph[path[-1]]:
            next_cost = curr_fn - heuristics[path[-1]] + cost + heuristics[next_state]
            if not is_tree and next_state in visited:
                continue

            frontier.append(path + next_state, next_cost)
    return False

# You might get a different trace due to popping different nodes.

print("=====")
print("Tree")
print("=====")
print(p:=astar(graph, 'S', lambda n: n=='G', h, is_tree=True, is_update=False))
assert(p=="SABG")

print("=====")
print("Graph")
print("=====")
print(p:=astar(graph, 'S', lambda n: n=='G', h, is_tree=False, is_update=False))
assert(p=="SBG")
