import numpy as np

def normalize_CardPole(state):
    """
    This function will normalize a state given from
    the CardPole environment so it can be used by a quantum circuit
    using Angle Encoding.

    Observation Space:
    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |
    
    Source: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
    """
    # TODO: Normalize velocity?

    if len(state.shape) == 1:
        assert len(state) == 4, f"Is this state from the CardPole environment? It should have 4 elements.\n State: {state}"

        # Normalize the state to be between -pi and pi
        state[0] = (state[0] / 4.8) * np.pi
        #state[1] = (state[1] / 4.8) * np.pi
        state[2] = (state[2] / 0.418) * np.pi
        #state[3] = (state[3] / 0.418) * np.pi

    elif len(state.shape) == 2:
        assert state.shape[1] == 4, f"Is this state from the CardPole environment? It should have 4 elements.\n State: {state}"

        # Normalize the state to be between -pi and pi
        state[:,0] = (state[:,0] / 4.8) * np.pi
        #state[:,1] = (state[:,1] / 4.8) * np.pi
        state[:,2] = (state[:,2] / 0.418) * np.pi
        #state[:,3] = (state[:,3] / 0.418) * np.pi


    return state



def normalize_AcroBot(state):
    """
    This function will normalize a state given from
    the AcroBot environment so it can be used by a quantum circuit
    using Angle Encoding.

    Observation Space:
    | Num | Observation                | Min                 | Max               |
    |-----|----------------------------|---------------------|-------------------|
    | 0   | Cosine of theta1           | -1                  | 1                 |
    | 1   | Sine of theta1             | -1                  | 1                 |
    | 2   | Cosine of theta2           | -1                  | 1                 |
    | 3   | Sine of theta2             | -1                  | 1                 |
    | 4   | Angular velocity of theta1 | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
    | 5   | Angular velocity of theta2 | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |

    
    Source: https://www.gymlibrary.dev/environments/classic_control/acrobot/
    """


    if len(state.shape) == 1:
        assert len(state) == 6, f"Is this state from the AcroBot environment? It should have 4 elements.\n State: {state}"

        # Normalize the state to be between -pi and pi
        state[0] = state[0] * np.pi
        state[1] = state[1] * np.pi
        state[2] = state[2] * np.pi
        state[3] = state[3] * np.pi
        state[4] = state[4] / 4
        state[5] = state[5] / 9

    elif len(state.shape) == 2:
        assert state.shape[1] == 6, f"Is this state from the AcroBot environment? It should have 4 elements.\n State: {state}"

        # Normalize the state to be between -pi and pi
        state[:,0] = state[:,0] * np.pi
        state[:,1] = state[:,1] * np.pi
        state[:,2] = state[:,2] * np.pi
        state[:,3] = state[:,3] * np.pi
        state[:,4] = state[:,4] / 4
        state[:,5] = state[:,5] / 9

    return state
