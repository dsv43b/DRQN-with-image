import torch
import numpy as np
from collections import deque
from typing import Union, Tuple, List


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    '''
    Буфер памяти

    size: размер буфера.
    images: архив с изображениями.
    '''

    def __init__(self, size: int, images: torch.tensor) -> None:
        self.state = deque(maxlen=size)
        self.action = deque(maxlen=size)
        self.reward = deque(maxlen=size)
        self.next_state = deque(maxlen=size)
        self.done = deque(maxlen=size)
        self.all_mem = [self.state, self.action, self.reward, self.next_state, self.done]
        self.images = images

    def add(self, *args):
        # Добавление новых данных в буфер
        for loc_memory, arg in zip(self.all_mem, args):
            loc_memory.append(arg)

    def sample(self, batch_size: int) -> Tuple[torch.tensor]:
        index = np.random.choice(len(self.action), batch_size)

        state_t = torch.cat([self.state[i][0] for i in index]).to(DEVICE)
        action_t = torch.LongTensor([self.action[i] for i in index]).unsqueeze(-1).to(DEVICE)
        reward_t = torch.FloatTensor([self.reward[i] for i in index]).unsqueeze(-1).to(DEVICE)
        next_state_t = torch.cat([self.next_state[i][0] for i in index]).to(DEVICE)
        done_t = torch.FloatTensor([self.done[i] for i in index]).unsqueeze(-1).to(DEVICE)
        num_img_state = [self.state[i][1] for i in index]
        num_img_nstate = [self.next_state[i][1] for i in index]
        image_state = torch.cat([self.images[i].unsqueeze(0) for i in num_img_state]).to(DEVICE)
        image_nstate = torch.cat([self.images[i].unsqueeze(0) for i in num_img_nstate]).to(DEVICE)
        return state_t, action_t, reward_t, next_state_t, done_t, image_state, image_nstate

    def __len__(self):
        return len(self.action)
