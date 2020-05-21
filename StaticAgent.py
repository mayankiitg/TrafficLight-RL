t = 0
def static_rl_actions(state):
    global t
    t += 1
    return [t%30 == 0]    

