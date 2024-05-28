from gymnasium import Env
import torch
import math
from deepqlearning.replay_memory import ReplayMemory, Transition
from networks.train_wrapper import TorchTrainWrapper
import pickle

class QLearner():
    def __init__(self, env : Env, model, device, **params):
        self.env                = env
        self.policy_net         = model
        self.replay_buffer      = ReplayMemory(params['replay_buffer_size'])
        self.batch_size         = params['batch_size']
        self.eps_start          = params['eps_start']
        self.eps_end            = params['eps_end']
        self.eps_decay          = params['eps_decay']
        self.learning_rate      = params['learning_rate']
        self.gamma              = params['gamma']
        self.tau                = params['tau']
        self.device             = device
        self.steps_done         = 0
        self.target_net         = None
        self.train_wrapper      = None

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def get_current_eps(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * self.steps_done / self.eps_decay)

    def increment_steps(self):
        self.steps_done += 1

    def select_action(self, state, in_test=False):
        if in_test or torch.rand(1).item() > self.get_current_eps():
            self.policy_net.eval()
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[self.env.action_space.sample()]]).to(self.device)
    
    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


    def optimize_model(self):
        if self.target_net is None:
            self.target_net = pickle.loads(pickle.dumps(self.policy_net))

        if self.train_wrapper is None:
            self.train_wrapper = TorchTrainWrapper(self.policy_net, 
                                                   torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True),
                                                    torch.nn.SmoothL1Loss())

        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = self.replay_buffer.sample(self.batch_size)

        batch = Transition(*zip(*transitions))

        state_batch  = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        self.policy_net.train()
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        self.target_net.eval()
        with torch.no_grad():
            for idx, state in enumerate(batch.next_state):
                if state is not None:
                    next_state_values[idx] = self.target_net(state).max(1).values


        expected_state_action_values = (reward_batch + self.gamma * next_state_values).unsqueeze(1)
        self.train_wrapper.train_step(state_action_values, expected_state_action_values)
        

    def episode(self, test = False):
        state, _ = self.env.reset()

        done = False
        total_reward = 0
        while not done:
            action = self.select_action(state, test)
            self.increment_steps()
            observation, reward, terminated, truncated, _ = self.env.step(action.item())

            reward = torch.tensor([reward], device=self.device, dtype=torch.float32) # TODO wrap in the environment
            total_reward += reward.item()

            if terminated:
                next_state = None
            else:
                next_state = observation

            if not test:
                self.replay_buffer.push(state, action, next_state, reward)
                
                self.optimize_model()
                self.update_target_net()

            state = next_state
            done  = terminated or truncated

        return total_reward
    

    def train(self, n_episodes, reward_break_condition):
        for i in range(n_episodes):
            reward = self.episode()
            print(f'Train Episode {i} reward: {reward}')
            if reward > reward_break_condition:
                break
        
    def test(self, n_episodes):
        total_reward = 0
        for i in range(n_episodes):
            reward = self.episode(test=True)
            print(f'Test {i} reward: {reward}')
            total_reward += reward

        print(f'Average Test Reward: {total_reward/n_episodes}')

        return total_reward/n_episodes