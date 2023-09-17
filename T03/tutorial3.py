import numpy as np


class State:
    def __init__(self, val, is_max_player=False, children=[]):
        self.children = children
        self.is_max_player = is_max_player
        self.val = val

    def get_children(self):
        return self.children

    def is_terminal(self):
        return len(self.children) == 0


def print_trace(isEnter, state, alpha, beta, depth, isPruned=False, val=0):
    symbol = '< ' if isEnter else '> '
    space = '    '
    result = ''
    for i in range(depth):
        result += space
    result += symbol
    result += str(state.val)
    if isPruned:
        result += ' Pruned val ' + ('>= beta: ' if state.is_max_player else '<= alpha: ')
        result += str(val) + (' >= ' if state.is_max_player else ' <= ')
        result += str(beta) if state.is_max_player else str(alpha)
    else:
        result += ' ' + str(alpha) + ' ' + str(beta)
    print(result)


def alpha_beta_search(state, alpha=-np.infty, beta=np.infty, depth=0):
    if state.is_terminal():
        return state.val

    v = np.infty * (-1 if state.is_max_player else 1)
    for next_state in state.children:
        print_trace(True, state, alpha, beta, depth)
        next_val = alpha_beta_search(next_state, alpha, beta, depth + 1)
        if state.is_max_player and next_val > v \
                or not state.is_max_player and next_val < v:
            v = next_val

        if (state.is_max_player and v >= beta) \
                or (not state.is_max_player and v <= alpha):
            print_trace(False, state, alpha, beta, depth, True, v)
            return v

        alpha = max(alpha, v) if state.is_max_player else alpha
        beta = min(beta, v) if not state.is_max_player else beta
        print_trace(False, state, alpha, beta, depth)

    return v


s = State('MAX0', True,
          [
              State('MIN1.1', False,
                    [
                        State('MAX2.1', True, [
                            State('MIN3.1', False, [State(8), State(7)]),
                            State('MIN3.2', False, [State(3), State(9)]),
                        ]),
                        State('MAX2.2', True, [
                            State('MIN3.3', False, [State(9), State(8)]),
                            State('MIN3.4', False, [State(2), State(4)]),
                        ]),
                    ]),
              State('MIN1.2', False,
                    [
                        State('MAX2.3', True, [
                            State('MIN3.5', False, [State(1), State(8)]),
                            State('MIN3.6', False, [State(8), State(9)]),
                        ]),
                        State('MAX2.4', True, [
                            State('MIN3.7', False, [State(9), State(9)]),
                            State('MIN3.8', False, [State(3), State(4)]),
                        ]),
                    ])
          ])

q4 = State('S', True, [
    State('a2', False, [
        State(9),
        State(7),
        State('b2', True, [State(5), State(4)])
    ]),
    State('a1', False, [
        State(9),
        State('b1', True, [
            State(3),
            State(1),
            State(9),
            State(4)
        ]),
        State(6)
    ])
])

print("Tutorial 3, Bonus:")
print(alpha_beta_search(s), '\n')
print("Tutorial 3, Q4:")
print(alpha_beta_search(q4))

'''
How can we benefit from α-β’s efficiency?

Faster Decision Making: Alpha-beta pruning allows you to explore fewer
nodes in the game tree, which means you can make decisions faster.

Improved Performance: In games where the branching factor is high.
Without this optimization, searching all possible moves would be too slow.

Reduced Memory Usage: By not having to store and evaluate as many nodes
in the game tree, you can reduce memory usage.

Adaptability: Alpha-beta pruning can be combined with various enhancements,
such as transposition tables and iterative deepening, to make the algorithm
even more efficient and adaptable to different game scenarios.
'''
