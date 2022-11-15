
def replace(idx, token):
    def _action(tokens):
        tokens[idx] = token
        return tokens
    return _action, ("R", idx, token)

def insert(idx, token):
    def _action(tokens):
        tokens.insert(idx, token)
        return tokens
    return _action, ("I", idx, token)

def delete(idx):
    def _action(tokens):
        del tokens[idx]
        return tokens
    return _action, ("D", idx, None)


def levenshtein_distance(initial_sequence, target_sequence, memory=None):
    if memory is None:
        memory = {}

    if len(initial_sequence) == 0:
        return len(target_sequence)
    if len(target_sequence) == 0:
        return len(initial_sequence)

    key = (tuple(initial_sequence), tuple(target_sequence))

    if key in memory:
        return memory[key]
        
    distance = 0
    if initial_sequence[0] == target_sequence[0]:
        distance = levenshtein_distance(initial_sequence[1:], target_sequence[1:], memory)
    else:
        distance =  1 + min(
            levenshtein_distance(initial_sequence[1:], target_sequence, memory),
            levenshtein_distance(initial_sequence, target_sequence[1:], memory),
            levenshtein_distance(initial_sequence[1:], target_sequence[1:], memory)
        )
    memory[key] = distance
    return distance


def iter_actions(initial_sequence, target_sequence):
    for i in range(len(initial_sequence)):
        yield delete(i)
        for token in set(target_sequence):
            yield insert(i, token)
            if token != initial_sequence[i]:
                yield replace(i, token)
    
    for token in set(target_sequence):
            yield insert(len(initial_sequence), token)


def find_actions(initial_sequence, target_sequence):
    memory = {}
    initial_distance = levenshtein_distance(initial_sequence, target_sequence, memory)
    actions = []
    action_tups = []

    for action, action_tup in iter_actions(initial_sequence, target_sequence):
        new_sequence = action(initial_sequence.copy())
        new_distance = levenshtein_distance(new_sequence, target_sequence, memory)
        if new_distance < initial_distance:
            actions.append(action)
            action_tups.append(action_tup)

    return action_tups



