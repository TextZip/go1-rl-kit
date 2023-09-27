import torch
from actor_critic import ActorCritic

num_obs = 44
path = 'weights/day5/walk8_sm45_model_2050.pt'
save = 'OLDwalk8_sm45_model_2050.pth.tar'
actor_critic = ActorCritic(num_actor_obs=num_obs,num_critic_obs=num_obs,num_actions=12,actor_hidden_dims = [128, 128],critic_hidden_dims = [128, 128],activation = 'elu',init_noise_std = 1.0)
loaded_dict = torch.load(path)
actor_critic.load_state_dict(loaded_dict['model_state_dict'])
torch.save(actor_critic.state_dict(),save,_use_new_zipfile_serialization=False)
# actor_critic.eval()
# self.policy = actor_critic.act_inference