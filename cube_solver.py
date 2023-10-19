import numpy as np
import tensorflow as tf

from utils import CubeState, get_possible_moves


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')
        ])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(state.reshape(1, -1))[0]
            return np.argmax(q_values)

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.random.choice(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        next_q_values = self.model.predict(next_states.reshape(batch_size, -1))
        for i, state, action, reward, next_state, done in enumerate(minibatch):
            target = reward
            if not done:
                target += self.gamma * np.max(next_q_values[i])

            self.model.fit(states.reshape(batch_size, -1), np.array([[target]]), epochs=1, verbose=0)
            self.epsilon *= self.epsilon_decay

    def train(self, n_episodes, batch_size):
        for episode in range(n_episodes):
            state = CubeState.from_scramble(scramble)
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done = state.apply_move(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state

            self.replay(batch_size)

    def solve(self, scramble):
        state = CubeState.from_scramble(scramble)
        moves = []
        while not state.is_solved():
            action = self.act(state.to_array())
            moves.append(action)
            state = state.apply_move(action)
        return moves


def main():
    env = DQNAgent(state_size=CubeState.state_size(), action_size=len(get_possible_moves()))
    env.train(n_episodes=100000, batch_size=128)

    scramble = "R U R' U' L' D R D'"
    moves = env.solve(scramble)
    print(moves)


if __name__ == "__main__":
    main()
