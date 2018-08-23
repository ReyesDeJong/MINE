from gaussian_dataset import generate_dataset
from simple_mine_refact import SimpleMINE

# Dataset
dimension = 20
corr_factor = 0.99
n_samples = 100000

inputs_x, inputs_z, mut_info = generate_dataset(dimension, corr_factor, n_samples)
print("Mutual Information for this dataset is " + str(mut_info))

# Model
params = {
    "batch_size": 256,
    "learning_rate": 1e-4,
    "input_dim": dimension,
    "ema_decay": 0.999
}

model = SimpleMINE(params)

# Train
max_it = 30000
stat_every = 100
model.train(inputs_x, inputs_z, max_it, stat_every)