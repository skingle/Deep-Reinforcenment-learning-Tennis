import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim


CONFIG={
    'buffer_size' : int(1e5),  # replay buffer size
    'batch_size' : 128,        # minibatch size
    'gamma' : 0.99,            # discount factor
    'tau' : 1e-3,              # for soft update of target parameters
    'lr_actor' : 1e-3,         # learning rate of the actor 
    'lr_critic' : 1e-3,        # learning rate of the critic
    'weight_decay' : 0,        # L2 weight decay
    'update_every' : 1,        # update the network after every UPDATE_EVERY timestep
    'update_times' : 1,        # update UPDATE_TIME for every update
    'epsilon' : 1,             # epsilon noise parameter
    'epsilon_decay' : 0,       # decay parameter of epsilon
    'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'OUNoise':{'mu':0,'theta':0.15,'sigma':0.2}
}

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, n_agents, random_seed,config=CONFIG):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        random.seed(random_seed)
        self.n_agents = n_agents
        self.config = config
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.config['device'])
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.config['device'])
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.config['lr_actor'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(self.config['device'])
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.config['device'])
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.config['lr_critic'], weight_decay=self.config['weight_decay'])

        # Noise process
        self.noise = OUNoise((n_agents, action_size), random_seed, \
        self.config['OUNoise']['mu'],\
        self.config['OUNoise']['theta'],\
        self.config['OUNoise']['sigma'])

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.config['buffer_size'], self.config['batch_size'], random_seed,self.config['device'])
        
        # Epsilon
        self.epsilon = self.config['epsilon']
        
#         # Make sure target is with the same weight as the source
#         self.hard_update(self.actor_target, self.actor_local)
#         self.hard_update(self.critic_target, self.critic_local)
    
    def step(self, states, actions, rewards, next_states, dones, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        #self.memory.add(states, actions, rewards, next_states, dones)
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.config['batch_size'] and timestep % self.config['update_every'] == 0:
            for _ in range(self.config['update_times']):
                experiences = self.memory.sample()
                self.learn(experiences, self.config['gamma'])

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.config['device'])
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            
            # epsilon decay 
            self.epsilon -= self.config['epsilon_decay']
            self.epsilon=np.maximum(self.epsilon,0.001)
            action += self.epsilon*self.noise.sample()
            #action += self.epsilon* np.random.randn(self.n_agents, self.action_size)
            
            
        
        return np.clip(action, -1, 1)
        

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # perform gradient clipping
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config['tau'])
        self.soft_update(self.actor_local, self.actor_target, self.config['tau'])   
        
        
 

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    def load_weights(self,cp_actor,cp_critic):
        self.critic_local.load_state_dict(torch.load(cp_critic))
        self.actor_local.load_state_dict(torch.load(cp_actor))
    
    def eval_act(self,state):
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        return action
        
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.reset()
        self.size = size

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        #dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        #dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size[0], self.size[1])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed,device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)