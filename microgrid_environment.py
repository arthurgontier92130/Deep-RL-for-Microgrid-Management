import numpy as np
import gym
from gym import spaces

class MicrogridEnv(gym.Env):
    def __init__(self,
                 E_max=100.0,
                 P_max=50.0,
                 eta_c=0.95,
                 eta_d=0.95,
                 SoC_min=0.1,
                 SoC_max=0.9,
                 alpha1=0.33,
                 alpha2=0.66,
                 lambda_deg=0.01,
                 lambda_soc=10.0,
                 delta_t=1.0):
        super().__init__()

        # Actions discrètes (7 niveaux)
        self.action_levels = np.array([-1.0, -alpha2, -alpha1, 0.0, alpha1, alpha2, 1.0])
        self.action_space = spaces.Discrete(len(self.action_levels))

        # États : [SoC, P_load, price, P_solar]
        self.observation_space = spaces.Box(
            low=np.array([-0.2, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.2, 100.0, 1.0, 100.0], dtype=np.float32)
        )

        # Paramètres physiques
        self.E_max = E_max
        self.P_max = P_max
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.SoC_min = SoC_min
        self.SoC_max = SoC_max
        self.lambda_deg = lambda_deg
        self.lambda_soc = lambda_soc
        self.delta_t = delta_t

        # Variables internes
        self.state = None
        self.current_step = 0

        self.reset()

    def step(self, action):
        a = self.action_levels[action] * self.P_max
        SoC, P_load, price, P_solar = self.state

        # Bilan de puissance
        P_grid = P_load - (P_solar + a)

        # Mise à jour du SoC
        SoC_next = SoC + ((self.eta_c * max(-a, 0) - (1/self.eta_d) * max(a, 0)) * self.delta_t / self.E_max)
       

        # Coût et récompense
        penalty_soc = self.lambda_soc * ((SoC_next < self.SoC_min) or (SoC_next > self.SoC_max))
        cost = price * P_grid * self.delta_t + self.lambda_deg * abs(a) * self.delta_t + penalty_soc
        reward = -cost

        # Mise à jour aléatoire (provisoire)
        P_load_next = np.random.uniform(10, 60)
        price_next = np.random.uniform(0.1, 0.3)
        P_solar_next = np.random.uniform(0, 40)

        self.state = np.array([SoC_next, P_load_next, price_next, P_solar_next], dtype=np.float32)

        done = (SoC_next <= 0.0) or (SoC_next >= 1.2)
        self.current_step += 1

        return self.state, reward, done, {}

    def reset(self):
        SoC_init = np.random.uniform(self.SoC_min, self.SoC_max)
        P_load = np.random.uniform(10, 60)
        price = np.random.uniform(0.1, 0.3)
        P_solar = np.random.uniform(0, 40)
        self.state = np.array([SoC_init, P_load, price, P_solar], dtype=np.float32)
        self.current_step = 0
        return self.state

    def render(self, mode='human'):
        pass

env = MicrogridEnv()
state = env.reset()

for t in range(10):
    action = env.action_space.sample()   # action aléatoire
    next_state, reward, done, info = env.step(action)
    print(f"Step {t}: Action={action}, Reward={reward:.3f}, SoC={next_state[0]:.3f}")
    if done:
        break
