import os
# get rid of the warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import logging
from environment.omni_channel_retailer import OmniChannelRetailerEnv
from agent.ppo_agent import PPOAgent
from data.parameters import parameters

def test_train():
    env = OmniChannelRetailerEnv(parameters)

    state_shape_level_I = (env.get_state_level_I().shape[0],)
    state_shape_level_II = (env.get_state_level_II().shape[0],)

    agent = PPOAgent(
        input_shape_level_I=state_shape_level_I,
        input_shape_level_II=state_shape_level_II,
        # Define the maximum number of actions for level I and level II based on the paper's constraints
        # just so shapes don't change (easier to implement), action space still changes dynamically
        # adjust according to demand (higher demand requires higher action space)
        output_shape_level_I=183,
        output_shape_level_II=183,
        B_hyppar=512, # Number of Periods, normally 512
        K_MAX_hyppar=1000, # Number of Episodes, normaly 1000
        ETA_hyppar=4, # Number of mini-batches
        MU_hyppar=10,   # Number of epochs
        ALPHA_hyppar=1e-4, # Learning rate
        BETA_E_hyppar=1e-5, # Entropy regularization
        EPSILON_hyppar=0.2, # Clipping parameter
        DELTA_hyppar=5, # Huber loss constant
        GAMMA_hyppar=0.99, # Discount future profits
    )


    agent.train(env, T=parameters['T'])

if __name__ == '__main__':
    logging.debug("Starting model training...")
    test_train()
    logging.debug("Model training completed.")


