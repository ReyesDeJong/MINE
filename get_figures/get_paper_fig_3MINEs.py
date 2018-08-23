#import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from gaussian_dataset import generate_dataset
from simple_mine_refact import SimpleMINE
from grad_corrected_mine import GradMINE
from simple_mine_f import MINEf
import time
import datetime

# Name
date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


# Dataset params
dimension = 180
corr_factors = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]#[0, 0.3, 0.99]
corr_neg = [-corr for corr in corr_factors][::-1] 
corr_factor_ls = corr_neg[0:len(corr_neg)-1]+corr_factors
n_samples_train = 100000
n_samples_test = 30000

name = date + "_all_MINE_" + str(dimension)

# Model params
params = {
    "batch_size": 256,
    "learning_rate": 1e-4,
    "input_dim": dimension,
    "ema_decay": 0.999
}

# Train params
max_it = 80000
stat_every = 10000

# To build list
theoric_I = []
estimated_I = []
estimated_I_grad = []
estimated_I_f = []
#%%
for i in range(len(corr_factor_ls)):
    # Data set
    inputs_x, inputs_z, _ = generate_dataset(dimension, corr_factor_ls[i], n_samples_train)
    test_inputs_x, test_inputs_z, mut_info = generate_dataset(dimension, corr_factor_ls[i], n_samples_test)
    test_inputs_z_hat =  np.copy(test_inputs_z)
    np.random.shuffle(test_inputs_z_hat)
    print("Mutual Information for %ith test dataset is: %s" % (i,str(mut_info)))
    theoric_I.append(mut_info)
    # Models
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
    
    # Models
    model_grad = GradMINE(params)
    # Train
    try:
        model_grad.train(inputs_x, inputs_z, max_it, stat_every)
    except Exception as e:
        print("type error: " + str(e))
        pass
    try:
        test_I_grad = model_grad.sess.run(model_grad.loss, feed_dict={model_grad.x: test_inputs_x, model_grad.z: test_inputs_z, model_grad.z_hat: test_inputs_z_hat})
    except Exception as e:
        print("type error: " + str(e))
        pass
    
    # Models
    model_f = MINEf(params)
    # Train
    try:
        model_f.train(inputs_x, inputs_z, max_it, stat_every)
    except Exception as e:
        print("type error: " + str(e))
        pass
    try:
        test_I_f = model_f.sess.run(model_f.loss, feed_dict={model_f.x: test_inputs_x, model_f.z: test_inputs_z, model_f.z_hat: test_inputs_z_hat})
    except Exception as e:
        print("type error: " + str(e))
        pass
    
    print("Estimated Simple Mutual Information for %ith test dataset is: %s" % (i,str(test_I)))
    print("Estimated Grad Mutual Information for %ith test dataset is: %s" % (i,str(test_I_grad)))
    print("Estimated f Mutual Information for %ith test dataset is: %s" % (i, str(test_I_f)))
    estimated_I.append(test_I)
    estimated_I_grad.append(test_I_grad)
    estimated_I_f.append(test_I_f)

#%%    

print("IM teorica")
print(theoric_I)
print("IM simple MINE")
print(estimated_I)
print("IM grad MINE")
print(estimated_I_grad)
print("IM MINE_f")
print(estimated_I_f)

print("IM teorica", flush=True, file=open('results/'+name+'_train.log', 'a'))
print(theoric_I, flush=True, file=open('results/'+name+'_train.log', 'a'))
print("IM simple MINE", flush=True, file=open('results/'+name+'_train.log', 'a'))
print(estimated_I, flush=True, file=open('results/'+name+'_train.log', 'a'))
print("IM grad MINE", flush=True, file=open('results/'+name+'_train.log', 'a'))
print(estimated_I_grad, flush=True, file=open('results/'+name+'_train.log', 'a'))
print("IM MINE_f", flush=True, file=open('results/'+name+'_train.log', 'a'))
print(estimated_I_f, flush=True, file=open('results/'+name+'_train.log', 'a'))

#%%
#import matplotlib
#matplotlib.use('Agg')
#import numpy as np
#import matplotlib.pyplot as plt

# Create plots with pre-defined labels.
#dimension = 20
#name = "all_mine"
#corr_factors = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]#[0, 0.3, 0.99]
#corr_neg = [-corr for corr in corr_factors][::-1]
#corr_factor_ls = corr_neg[0:len(corr_neg)-1]+corr_factors
#theoric_I = [39.170355472516874, 16.607312068216512, 6.733445532637656, 2.8768207245178083, 0.9431067947124133, 0.10050335853501455, -0.0, 0.10050335853501455, 0.9431067947124133, 2.8768207245178083, 6.733445532637656, 16.607312068216512, 39.170355472516874]
#estimated_I = [np.nan, np.nan, np.nan, 2.3143282, 0.8775661, 0.069535024, -0.026147725, 0.07246518, 0.86943215, 2.5712926, np.nan, np.nan, np.nan]
#estimated_I_grad = [4.4733887, 2.7001667, 6.024006, 2.699387, 0.8863009, 0.074488044, -0.021217227, 0.07841396, 0.86781573, 2.5571303, 5.5366993, 11.926083, 13.178387]
#estimated_I_f = [18.260582, 12.114112, 3.2008615, 2.779146, 0.8839741, 0.07127565, -0.023192942, 0.076206625, 0.8925297, 2.5285978, 6.1364875, 11.3494215, 19.390987]

#dimension=180
#IM teorica
#[352.53319925265197, 149.4658086139486, 60.60100979373875, 25.89138652066023, 8.48796115241175, 0.9045302268151266, -0.0, 0.9045302268151266, 8.48796115241175, 25.89138652066023, 60.60100979373875, 149.4658086139486, 352.53319925265197]
#IM simple MINE
#[nan, nan, nan, nan, 2.0991511, -0.0598675, -0.30358177, -0.06523633, 1.5352504, nan, nan, nan, nan]
#IM grad MINE
#[nan, 3.0846558, 8.46356, 5.5037537, 2.1676087, 0.013367176, -0.28315628, 0.02129507, 2.3358266, 4.492078, 5.1288576, 7.180626, nan]
#IM MINE_f
#[15.281321, 14.355327, 3.0857368, 3.2455711, 1.9532733, -0.027747154, -0.2884177, -0.016965985, 2.1113505, 2.9029574, 7.950784, 6.2512007, 12.364767]


fig, ax = plt.subplots()

ax.plot(corr_factor_ls, theoric_I, 'b--', label='True MI')
ax.plot(corr_factor_ls, estimated_I_f, 'g*')
ax.plot(corr_factor_ls, estimated_I_f, 'g-', label='MINE_f MI')
ax.plot(corr_factor_ls, estimated_I_grad, 'r-', label='Grad MINE')
ax.plot(corr_factor_ls, estimated_I_grad, 'r*')
ax.plot(corr_factor_ls, estimated_I, 'b-', label='Simple MINE')
ax.plot(corr_factor_ls, estimated_I, 'b*')

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



    
    
    