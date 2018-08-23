#import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from gaussian_dataset import generate_dataset
from simple_mine_refact import SimplMINE

# Name
name = "simple_MINE"

# Dataset params
dimension = 20
corr_factors = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]#[0, 0.3, 0.99]
corr_neg = [-corr for corr in corr_factors][::-1] 
corr_factor_ls = corr_neg[0:len(corr_neg)-1]+corr_factors
n_samples_train = 100000
n_samples_test = 30000

# Model params
params = {
    "batch_size": 256,
    "learning_rate": 1e-4,
    "input_dim": dimension,
    "ema_decay": 0.999
}

# Train params
max_it = 50000
stat_every = 1000

# To build list
theoric_I = []
estimated_I = []
#%%
for i in range(len(corr_factor_ls)):
    # Data set
    inputs_x, inputs_z, _ = generate_dataset(dimension, corr_factor_ls[i], n_samples_train)
    test_inputs_x, test_inputs_z, mut_info = generate_dataset(dimension, corr_factor_ls[i], n_samples_test)
    test_inputs_z_hat =  np.copy(test_inputs_z)
    np.random.shuffle(test_inputs_z_hat)
    print("Mutual Information for %ith test dataset is: %s" % (i,str(mut_info)))
    theoric_I.append(mut_info)
    # Model
    model = SimpleMINE(params)
    # Train
    try:
        model.train(inputs_x, inputs_z, max_it, stat_every)
    except Exception as e:
        print("type error: " + str(e))
        pass
    print("Mutual Information for %ith test dataset is: %s" % (i,str(mut_info)))
    try:
        test_I = model.sess.run(model.loss, feed_dict={model.x: test_inputs_x, model.z: test_inputs_z, model.z_hat: test_inputs_z_hat})
    except Exception as e:
        print("type error: " + str(e))
        pass
    print("Estimated Mutual Information for %ith test dataset is: %s" % (i,str(test_I)))
    estimated_I.append(test_I)

#%%    

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(corr_factor_ls, estimated_I, 'b-', label=name)
ax.plot(corr_factor_ls, theoric_I, 'b--', label='True MI')
ax.set_ylabel(r'$I(X_a;X_b)$', fontsize=15)
ax.set_xlabel(r'$\rho$', fontsize=15)
ax.set_title('Mutual Information of %i-dimensional variables' % (dimension))
ax.grid(True)


legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')

fig.tight_layout()
fig.savefig('results/'+name+'.png', bbox_inches='tight')
#plt.show()

    
    
    