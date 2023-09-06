from treelib import Tree
import queue

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

def astar(graph, inital_node, goal_test, heuristics, is_tree, is_update):
    tree = Tree()
    priorityQueue = queue.PriorityQueue()

    raise NotImplementedError

    tree.save2file('astar.txt', line_type='ascii')

    return ''.join(path)

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