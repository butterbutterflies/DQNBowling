"""
When training, use 'A' and 'D' keys for positive and negative rewards
"""
import gym
import asyncio
import torch
from tamer.agent import DeepTamer, Encoder, Head


async def main():
    env = gym.make("Bowling-v0", mode=2).unwrapped
    env = gym.wrappers.TimeLimit(env, max_episode_steps=5000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    head = Head().to(device)
    encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))

    for name, params in encoder.named_parameters():
        params.requires_grad = False

    # hyperparameters
    epsilon = 0.9
    min_eps = 0.3
    num_episodes = 20
    batch = 160
    tamer_training_time_step = 0.1
    agent = DeepTamer(env, encoder, head, num_episodes, batch,
                      tamer_training_time_step, epsilon, min_eps)

    agent.train(model_file_to_save='auto_save')

if __name__ == '__main__':
    asyncio.run(main())




