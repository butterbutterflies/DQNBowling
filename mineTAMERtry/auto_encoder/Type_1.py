import os
import pathlib
import random
import warnings

import gym
import matplotlib.pyplot as plt
import numpy as np

# warnings.filterwarnings("ignore")

from collections import deque
from itertools import count

import torch
import torchvision
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim

torch.manual_seed(0)

'''
This code is to make auto-encoder train till episode is DONE.
each state is 1 RGB image.
'''


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(1)

        self.linear_1 = nn.Linear(64, 100)

    def forward(self, x):
        """
        x = x
        x = F.max_pool2d(self.conv_bn1(self.conv1(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn2(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        # encoded states might come in "-ve" so no Relu or softmax
        return x
        """
        x = self.conv_bn1(torch.relu(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.conv_bn1(torch.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.conv_bn1(torch.relu(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.conv_bn2(torch.relu(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.linear_1(x))

        return x


class Decoder(nn.Module):
    # Linear - Reshape - Upsample - Deconv + BatchNorm
    def __init__(self):
        super(Decoder, self).__init__()

        self.linear_1 = nn.Linear(100, 64)

        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_2 = nn.Upsample(scale_factor=2.03)

        self.deconv_1 = nn.ConvTranspose2d(1, 64, 3)
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 3)
        self.deconv_3 = nn.ConvTranspose2d(64, 3, 3)

        self.deconv_bn1 = nn.BatchNorm2d(64)
        self.deconv_bn2 = nn.BatchNorm2d(3)

    def forward(self, x):
        """
        x = x
        x = F.relu(self.linear_1(x))
        x = x.view(x.shape[0], 1, 8, 8)
        x = self.deconv_bn1(self.deconv_1(self.upsample_1(x)))
        x = self.deconv_bn1(self.deconv_2(self.upsample_1(x)))
        x = self.deconv_bn1(self.deconv_2(self.upsample_2(x)))
        x = self.deconv_bn2(self.deconv_3(self.upsample_1(x)))
        return x
        """
        x = torch.relu(self.linear_1(x))
        x = x.view(x.shape[0], 1, 8, 8)
        x = torch.relu(self.deconv_1(self.upsample_1(x)))
        x = self.deconv_bn1(x)
        x = torch.relu(self.deconv_2(self.upsample_1(x)))
        x = self.deconv_bn1(x)
        x = torch.relu(self.deconv_2(self.upsample_2(x)))
        x = self.deconv_bn1(x)
        x = torch.relu(self.deconv_3(self.upsample_1(x)))
        x = self.deconv_bn2(x)
        x = torch.sigmoid(x)  # sigmoid
        return x


class BufferArray():
    def __init__(self, size):
        self.memory = torch.zeros(size, dtype=torch.float32)
        self._mem_loc = 0
        self.push_count = 0

    def __len__(self):
        return self.push_count

    def push(self, tensor):
        type(self._mem_loc)
        if self._mem_loc == len(self.memory) - 1:
            self._mem_loc = 0
        self.memory[self._mem_loc] = tensor.cpu()
        self._mem_loc += 1
        self.push_count += 1

    def random_sample(self, batch_size):
        rand_batch = np.random.randint(len(self.memory), size=batch_size)
        if len(rand_batch.shape) == 3:
            return torch.unsqueeze(self.memory[rand_batch], axis=1)
        return self.memory[rand_batch]


class BufferDeque():
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        if isinstance(index, tuple, list):
            pointer = list(self.memory)
            return [pointer[i] for i in index]
        return list(self.memory[index])

    def push(self, tensor):
        self.memory.append(tensor.cpu())

    def random_sample(self, batch_size):
        rand_batch = np.random.randint(len(self.memory), size=batch_size)
        return torch.stack([self.memory[b] for b in rand_batch])


def main():
    env = gym.make("Bowling-v0").unwrapped
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    no_of_episodes = 40  # 2000
    batch_size = 32
    img_dims = (3, 160, 160)
    buffer_size = (20000,) + img_dims

    def select_action():
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return action

    def get_screen():
        screen = env.render(mode="rgb_array").transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((img_dims[1:])),
                            T.ToTensor()])
        screen = resize(screen).to("cuda")  # .unsqueeze(0)
        return screen  # Returns Grayscale, PILIMAGE, TENSOR

    # buffer = BufferDeque(buffer_size[0])
    buffer = BufferArray(buffer_size)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_history = []

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4, weight_decay=1e-1)

    def image_visual():
        batch = 3
        plt.figure(figsize=(25, 30))
        i = buffer.random_sample(batch).to(device)
        o = decoder(encoder(i))

        interleaved_shape = (len(i) + len(o),) + i.shape[1:]
        interleaved = torch.empty(interleaved_shape)
        interleaved[0::2] = i
        interleaved[1::2] = o

        img_grid = torchvision.utils.make_grid(interleaved, nrow=2)
        plt.imshow(img_grid.permute(1, 2, 0))
        plt.savefig(os.path.join(pathlib.Path().absolute(), 'Type_1/result_lili'))

    def loss_visual(x):
        # plt.plot(x.detach().numpy())
        plt.plot(x)
        plt.show()
        plt.savefig(os.path.join(pathlib.Path().absolute(), 'Type_1/loss_history_lili'))

    def optimize():
        # Sample batch and preprocess
        state_batch = buffer.random_sample(batch_size)

        # Move state_batch to GPU and run through auto-encoder
        state_batch = state_batch.to(device)

        output_batch = decoder(encoder(state_batch))

        # Compute loss
        loss = loss_fn(state_batch, output_batch).cpu()

        # Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss

    for episode in range(no_of_episodes):
        env.reset()
        state = get_screen()

        done = False
        for step in count():
            action = select_action()
            _, _, done, _ = env.step(action.item())

            if not done:
                next_state = get_screen()
            else:
                next_state = None

            # Store transition
            buffer.push(state)
            # Set next state 
            state = next_state

            # Optimization/training
            # Don't start training until there are enough samples in memory
            if len(buffer.memory) > 1000:
                # Only train every certain number of steps
                if step % 16 == 0:
                    loss = optimize()
                    loss_history.append(loss.cpu().item())
                    print('Episode: {} Step: {} Loss: {:5f}'.format(episode, step, loss))
            # Check if done
            if done:
                break

    image_visual()
    print(loss_history)
    loss_visual(loss_history)

    torch.save(encoder.state_dict(), 'Type_1//encoder1.pt')
    torch.save(decoder.state_dict(), 'Type_1//decoder1.pt')

if __name__ == "__main__":
    main()

# SBATCH --mail-type=all
# SBATCH --mail-user=<sarrabel@uncc.edu>
