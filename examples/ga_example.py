from ydata_synthetic.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from ydata_synthetic.genetic_algorithm.solution import Solution
import tensorflow as tf
tf.debugging.set_log_device_placement(True)

### Temporary
from ydata_synthetic.genetic_algorithm.population import Population
trying_pop = Population(20,Solution())
trying_pop.save('pop_prova.pkl')
new_pop = Population
new_pop = new_pop.load('./pops/modified_pop.pkl')



df = pd.read_csv('data.csv',index_col=0)
columns = df.columns
df.dropna(inplace=True)
# df_quant = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
# df = df.loc[df.index.intersection(df_quant.index)]
scaler = MinMaxScaler()
df = scaler.fit_transform(df)
df = pd.DataFrame(df,columns=columns)
dimension = [128,10,128] #Noise_dim,data_dim,dim
train_args = ['',300,250]

# SET GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


ga_params = {"population_size": 250,
    "cross_probability": 0.6,
    "mutation_probability": 0.1,
    "number_generations": 20}

limits = {
        "batch_limits": [20,500],
        "lr_limits": [0.00001,0.005],
        "beta1_limits": [0.01,0.99],
        "beta2_limits": [0.001,0.99],
        "n_critic_limits": [2,5],
        "weight_gp_limits": [7,12]
    }


ga = GeneticAlgorithm(dimension=dimension,data=df,train_args=train_args,\
                      params=ga_params,limits=limits)

best, top5 = ga.search()

best.representation

from tensorflow.python.client import device_lib
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)





from ydata_synthetic.synthesizers.regular.wgangp.model import WGAN_GP

noise_dim = 243
dim = 10
batch_size = 243

log_step = 100
epochs = 300
learning_rate = 0.00324
beta_1 = 0.25
beta_2 = 0.852


gan_args = [batch_size, learning_rate, beta_1, beta_2, noise_dim, df.shape[1], dim]
train_args = ['', epochs, log_step]


gan = WGAN_GP(gan_args, n_critic=2,gradient_penalty_weight=11)
gan.train(df,train_args)

for i,solution in enumerate(new_pop.solutions):
    if np.isnan(solution.fitness):
        new_pop.solutions[i].set_fitness(1000)


a_pop= Population
a_pop = a_pop.load('./pops/pop_number_14.pkl')
