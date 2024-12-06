from keras import models, layers, optimizers
import tensorflow as tf
import logging
import numpy as np



class PPOAgent:
    def __init__(self, input_shape_level_I, input_shape_level_II, output_shape_level_I, output_shape_level_II, B_hyppar=512, ETA_hyppar=4, MU_hyppar=10, K_MAX_hyppar=1000,
                 ALPHA_hyppar=1e-4, BETA_E_hyppar=1e-5, EPSILON_hyppar=0.2, DELTA_hyppar=5, GAMMA_hyppar=0.99):
        """
        Initialize the PPOAgent with two sets of actor and critic networks for a hierarchical MDP and hyperparameters.
        Hyperparameters initialized with default values for the PPO algorithm.
        """
        self.actor_level1 = self.build_actor(input_shape_level_I, output_shape_level_I)
        self.actor_level2 = self.build_actor(input_shape_level_II, output_shape_level_II)

        self.critic_level1 = self.build_critic(input_shape_level_I)
        self.critic_level2 = self.build_critic(input_shape_level_II)

        # Hyperparameters
        self.B_hyppar = B_hyppar # Number Periods per sampling run
        self.ETA_hyppar = ETA_hyppar # Number of mini-batches
        self.MU_hyppar = MU_hyppar # Number of epochs
        self.K_MAX_hyppar = K_MAX_hyppar  # Episodes
        self.ALPHA_hyppar = ALPHA_hyppar # Learning rate
        self.BETA_E_hyppar = BETA_E_hyppar # Entropy regularization
        self.EPSILON_hyppar = EPSILON_hyppar # Clipping parameter
        self.DELTA_hyppar = DELTA_hyppar # Huber loss constant
        self.GAMMA_hyppar = GAMMA_hyppar  # to discount future profits

        # Optimizers for actor and critic with specified learning rate
        self.actor_optimizer_level1 = optimizers.Adam(learning_rate=self.ALPHA_hyppar)
        self.critic_optimizer_level1 = optimizers.Adam(learning_rate=self.ALPHA_hyppar)
        self.actor_optimizer_level2 = optimizers.Adam(learning_rate=self.ALPHA_hyppar)
        self.critic_optimizer_level2 = optimizers.Adam(learning_rate=self.ALPHA_hyppar)
 

    def build_actor(self, input_shape, max_output_shape):
        """
        Creates a neural network for the actor which outputs action probabilities.
        
        Args:
            input_shape (tuple): The shape of the input to the network.
            output_shape (int): The size of the action space.
        
        Returns:
            tf.keras.Model: The actor model.
        """
        state_input = layers.Input(shape=input_shape)
        mask_input = layers.Input(shape=(max_output_shape,), dtype=tf.float32)
        
        x = layers.Dense(256, activation='tanh', use_bias=True)(state_input)
        x = layers.Dense(256, activation='tanh', use_bias=True)(x)
        
        # As defined in Paper, mask actions before softmax function
        # Logits before softmax, as described in Huang, S., & Ontañón, S. (2020).
        logits = layers.Dense(max_output_shape, activation=None)(x)
        
        # Apply action mask to logits
        masked_logits = layers.Lambda(lambda x: x[0] + (-1e8 * (1 - x[1]))) ([logits, mask_input])

        # Apply softmax to get action probabilities
        action_probs = layers.Softmax()(masked_logits)
        model = tf.keras.Model(inputs=[state_input, mask_input], outputs=action_probs)
        
        #logging.debug(f"Actor Level 1 Model Summary: \n{model.summary()}")
        return model
    

    def build_critic(self, input_shape):
        """
        Creates a neural network for the critic which predicts state value.
        
        Args:
            input_shape (tuple): The shape of the input to the network.
        
        Returns:
            tf.keras.Model: The critic model.
        """
        model = models.Sequential([
            layers.InputLayer(input_shape=input_shape),
            layers.Dense(256, activation='tanh', use_bias=True), # Width 256 (nodes) and tanh activation function
            layers.Dense(256, activation='tanh', use_bias=True), # Second layer as per depth of 2
            layers.Dense(1) # only gives back predicted value
        ])
        #logging.debug(f"Critic Level 1 Model Summary: \n{model.summary()}")
        return model

    
    def sampling_level1(self, env, valid_actions_level1):
        """
        Collects a trajectory of specified length by interacting with the environment using the level 1 policy.

        Args:
            env: The environment object compliant with OpenAI Gym interface, which is used to simulate the interaction dynamics.
            num_steps (int): The number of steps to simulate in the environment. This defines the trajectory length for level 1.
            valid_actions_level1 (list): List of valid actions for level 1.

        Returns:
            tuple of np.array: Returns arrays containing the states, actions, next states, done flags, profits, and expected future profits collected during the trajectory for level 1.
        """
        states1, dones1, profits1, future_profits1, action_masked_list1  = [], [], [], [], []

        state = env.get_state_level_I()  # start from the initial state provided

        # Create a mask for valid actions
        max_actions = self.actor_level1.output_shape[1]
        action_mask = np.zeros(max_actions)
        action_mask[valid_actions_level1] = 1

        # taking an action based on the actor network
        action_probs1 = self.actor_level1.predict([np.expand_dims(state, axis=0), np.expand_dims(action_mask, axis=0)])[0]

        action1 = np.random.choice(np.arange(action_probs1.shape[0]), p=action_probs1)  # Select a random action

        # Interact with environment at level 1 using the selected action
        next_state1, profit1, done1, _ = env.step(action1)
        next_state1 = env.get_state_level_I()  # Get the next state for Level I

        # Calculate expected future profit
        future_profit1 = self.critic_level1.predict(np.expand_dims(next_state1, axis=0))[0][0]

        # Store the results of level 1 interaction
        action_masked_list1.append(action_mask)
        states1.append(state)
        dones1.append(done1)
        profits1.append(profit1)
        future_profits1.append(future_profit1)

        state = next_state1  # Update state for the next iteration

        return states1, dones1, profits1, future_profits1, action_masked_list1


    def sampling_level2(self, env, valid_actions_level2):
        """
        Collects a trajectory of specified length by interacting with the environment using the level 2 policy.

        Args:
            env: The environment object compliant with OpenAI Gym interface, which is used to simulate the interaction dynamics.
            num_inner_steps (int): The number of inner steps to simulate in the environment. This defines the trajectory length for level 2.
            valid_actions_level2 (list): List of valid actions for level 2.
            initial_state (np.array): The initial state from which level 2 sampling starts.

        Returns:
            tuple of np.array: Returns arrays containing the states, actions, next states, done flags, profits, and expected future profits collected during the trajectory for level 2.
        """
        states2, dones2, profits2, future_profits2, action_masked_list2  =  [], [], [], [], []

        state = env.get_state_level_II()  # Get initial state for Level II
        
        # Create a mask for valid actions
        max_actions = self.actor_level2.output_shape[1]
        action_mask = np.zeros(max_actions)
        action_mask[valid_actions_level2] = 1

        # Predict action probabilities using actor_level2
        action_probs2 = self.actor_level2.predict([np.expand_dims(state, axis=0), np.expand_dims(action_mask, axis=0)])[0]

        # Select a random action based on the predicted probabilities
        action2 = np.random.choice(np.arange(action_probs2.shape[0]), p=action_probs2)

        # Interact with environment at level 2 using the selected action
        next_state2, profit2, done2, _ = env.step(action2)
        next_state2 = env.get_state_level_II()  # Get the next state for Level II

        # Calculate expected future profit
        future_profit2 = self.critic_level2.predict(np.expand_dims(next_state2, axis=0))[0][0]

        # Store the results of level 2 interaction
        action_masked_list2.append(action_mask)
        states2.append(state)
        dones2.append(done2)
        profits2.append(profit2)  # Store immediate reward as profit
        future_profits2.append(future_profit2)

        state = next_state2  # Update state for next inner step


        return states2, dones2, profits2, future_profits2, action_masked_list2


    def update_networks_level1(self, states, dones, profits, future_profits, action_masked_list1):
        """
        Updates the level 1 actor and critic networks using the Proximal Policy Optimization (PPO) algorithm.

        Args:
            states (np.array): Array of observed states from the environment.
            actions (np.array): Array of actions taken based on the actor's policy.
            next_states (np.array): Array of next states observed from the environment.
            dones (np.array): Array indicating whether each state was terminal (end of an episode).
            profits (np.array): Array of profits received for actions taken.
            future_profits (np.array): Array of expected future profits.
        """
        # Predict the current values of the states with the critic network.
        predicted_profits = self.critic_level1.predict(states)

        discounted_profits = []
        discounted_sum = 0  # Initialize the discounted sum.
        # Calculate discounted profits based on the formula from the paper
        # Iterating over each tuple (first reversing the lists)
        for profit, done, future_profit in zip(reversed(profits), reversed(dones), reversed(future_profits)):
            if done:
                discounted_sum = 0  # Reset the sum if the episode has ended.
            # Left side of equation see appendix
            # Apply the discount factor and add the profit. + gamma discounts profits to the present
            discounted_sum = profit + self.GAMMA_hyppar * discounted_sum 
            # Righ side of equation see appendix
            discounted_profits.insert(0, discounted_sum + self.GAMMA_hyppar * future_profit)  # Insert the result at the beginning to maintain the correct order.

        discounted_profits = np.array(discounted_profits, dtype=np.float32)  # Ensure discounted_profits is float32
        predicted_profits = predicted_profits.astype(np.float32)  # Ensure predicted_profits is float32

        # Calculate advantages as the difference between discounted profits and predicted profits.
        advantages = discounted_profits - predicted_profits.flatten()

        # Convert all arrays to TensorFlow tensors to enable processing with TensorFlow operations:
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        discounted_profits = tf.convert_to_tensor(discounted_profits, dtype=tf.float32)
        delta_float = tf.cast(self.DELTA_hyppar, dtype=tf.float32)  # Ensure DELTA_hyppar is float32
        action_masked_list1 = tf.convert_to_tensor(action_masked_list1, dtype=tf.float32)

        # starting an epoch
        for epoch in range(self.MU_hyppar):
            # Shuffle and split the data at the start of each epoch
            dataset = self.shuffle_and_split_data(self.ETA_hyppar, states, advantages, discounted_profits, action_masked_list1)
            # Iterate over the mini-batches
            for mini_batch in dataset:
                mb_states, mb_advantages, mb_discounted_profits, mb_action_masked_list = mini_batch
                with tf.GradientTape(persistent=True) as tape:
                    pi = self.actor_level1([mb_states, mb_action_masked_list], training=True)
                    pi_old = tf.stop_gradient(self.actor_level1([mb_states, mb_action_masked_list], training=False))
                    predicted_values = self.critic_level1(mb_states, training=True)
                    total_average_loss = self.custom_loss_fn(mb_advantages, pi, pi_old, delta_float, mb_discounted_profits, predicted_values)

                actor_grads = tape.gradient(total_average_loss, self.actor_level1.trainable_variables)
                critic_grads = tape.gradient(total_average_loss, self.critic_level1.trainable_variables)

                actor_grads_vars = [(grads, var) for grads, var in zip(actor_grads, self.actor_level1.trainable_variables)]
                critic_grads_vars = [(grads, var) for grads, var in zip(critic_grads, self.critic_level1.trainable_variables)]
            
                # Apply gradients
                self.actor_optimizer_level1.apply_gradients(actor_grads_vars)
                self.critic_optimizer_level1.apply_gradients(critic_grads_vars)

        return total_average_loss  # Optionally return the total average loss for monitoring purposes


    def custom_loss_fn(self, advantages, pi, pi_old, delta_float, discounted_profits, predicted_values, periods=1):
        ratio = pi / (pi_old + 1e-8) # This is π(q_k | S_k, θ'1) / π(q_k | S_k, θ1) (plus small value to avoid div. by zero)
        clipped_ratio = tf.clip_by_value(ratio, 1 - self.EPSILON_hyppar, 1 + self.EPSILON_hyppar)
        advantages_expanded = tf.expand_dims(advantages, axis=1)
        policy_loss = -tf.reduce_sum(tf.minimum(ratio * advantages_expanded, clipped_ratio * advantages_expanded))  # Calculate the clipped policy loss.
        entropy_loss = -self.BETA_E_hyppar * tf.reduce_sum(pi * tf.math.log(pi + 1e-10))  # Entropy loss to encourage exploration.
        actor_loss = policy_loss + entropy_loss
        
        # Forward pass through the critic network to ensure it's used in the computation graph
        residuals = discounted_profits - predicted_values
        huber_loss = tf.where(tf.abs(residuals) <= delta_float,
                              0.5 * tf.square(residuals),
                              delta_float * tf.abs(residuals) - 0.5 * tf.square(delta_float))
        value_loss = tf.reduce_sum(huber_loss) # Sum Huber loss over the mini-batch
        total_average_loss = (value_loss + actor_loss) / (self.B_hyppar * periods / self.ETA_hyppar)
        return total_average_loss


    def shuffle_and_split_data(self, num_batches, *arrays):
        # Ensure all arrays have the same length
        assert all(len(array) == len(arrays[0]) for array in arrays), "All input arrays must have the same length"

        # Get the indices and shuffle them
        indices = tf.range(start=0, limit=tf.shape(arrays[0])[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        # Shuffle each array according to the shuffled indices
        shuffled_arrays = [tf.gather(array, shuffled_indices) for array in arrays]

        # Split each array into the specified number of batches
        batch_size = len(shuffled_arrays[0]) // num_batches
        split_arrays = [
            [array[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
            for array in shuffled_arrays
        ]

        # Combine the batches
        return list(zip(*split_arrays))


    def update_networks_level2(self, states, dones, profits, future_profits, action_masked_list2, periods):
        """
        Updates the level 2 actor and critic networks using the Proximal Policy Optimization (PPO) algorithm.

        Args:
            states (np.array): Array of observed states from the environment.
            actions (np.array): Array of actions taken based on the actor's policy.
            next_states (np.array): Array of next states observed from the environment.
            dones (np.array): Array indicating whether each state was terminal (end of an episode).
            profits (np.array): Array of profits received for actions taken.
            future_profits (np.array): Array of expected future profits.
        """

        # Predict the current values of the states with the critic network.
        predicted_profits = self.critic_level2.predict(states)

        discounted_profits = []
        discounted_sum = 0  # Initialize the discounted sum.
        # Calculate discounted profits based on the formula from the paper
        for profit, done, future_profit in zip(reversed(profits), reversed(dones), reversed(future_profits)):
            if done:
                discounted_sum = 0  # Reset the sum if the episode has ended.
            discounted_sum = profit + self.GAMMA_hyppar * discounted_sum  # Apply the discount factor and add the profit.
            discounted_profits.insert(0, discounted_sum + self.GAMMA_hyppar * future_profit)  # Insert the result at the beginning to maintain the correct order.

        discounted_profits = np.array(discounted_profits, dtype=np.float32)  # Ensure discounted_profits is float32
        predicted_profits = predicted_profits.astype(np.float32)  # Ensure predicted_profits is float32

        # Calculate advantages as the difference between discounted profits and predicted profits.
        advantages = discounted_profits - predicted_profits.flatten()

        # Convert all arrays to TensorFlow tensors to enable processing with TensorFlow operations:
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        discounted_profits = tf.convert_to_tensor(discounted_profits, dtype=tf.float32)
        delta_float = tf.cast(self.DELTA_hyppar, dtype=tf.float32)  # Ensure DELTA_hyppar is float32
        action_masked_list2 = tf.convert_to_tensor(action_masked_list2, dtype=tf.float32)

        # Shuffle data and create mini-batches
        for epoch in range(self.MU_hyppar):
            # Shuffle and split the data at the start of each epoch
            dataset = self.shuffle_and_split_data(self.ETA_hyppar, states, advantages, discounted_profits, action_masked_list2)
            # Iterate over the mini-batches
            for mini_batch in dataset:
                mb_states, mb_advantages, mb_discounted_profits, mb_action_masked_list = mini_batch
                with tf.GradientTape(persistent=True) as tape:
                    pi = self.actor_level2([mb_states, mb_action_masked_list], training=True)
                    pi_old = tf.stop_gradient(self.actor_level2([mb_states, mb_action_masked_list], training=False))
                    predicted_values = self.critic_level2(mb_states, training=True)
                    total_average_loss = self.custom_loss_fn(mb_advantages, pi, pi_old, delta_float, mb_discounted_profits, predicted_values, periods=periods)

                actor_grads = tape.gradient(total_average_loss, self.actor_level2.trainable_variables)
                critic_grads = tape.gradient(total_average_loss, self.critic_level2.trainable_variables)

                actor_grads_vars = [(grads, var) for grads, var in zip(actor_grads, self.actor_level2.trainable_variables)]
                critic_grads_vars = [(grads, var) for grads, var in zip(critic_grads, self.critic_level2.trainable_variables)]
            
                # Apply gradients
                self.actor_optimizer_level2.apply_gradients(actor_grads_vars)
                self.critic_optimizer_level2.apply_gradients(critic_grads_vars)

        return total_average_loss  # Optionally return the total average loss for monitoring purposes


    def evaluate_policy(self, env, actor_level1, actor_level2, T, num_periods=1000000, warm_up_period=1000):
        """
        Evaluates the current policy by executing it in the environment for a specified number of periods
        and computes the average profit per period. Includes a convergence check.

        Args:
            env: The environment to evaluate on. Must be compatible with the actor's action space.
            actor_level1 (tf.keras.Model): The actor model for level 1 used to predict actions.
            actor_level2 (tf.keras.Model): The actor model for level 2 used to predict actions.
            num_periods (int): The number of periods to run the policy for evaluation.
            warm_up_period (int): The number of periods to warm up before evaluation.

        Returns:
            float: The average profit obtained over the evaluation periods.
        """
        recent_profits = []  # List to keep track of the profits for the last 30 periods.
        cycle_service_list_offline = [] # List to keep track of the service level for the last 30 periods.
        cycle_service_list_online = [] # List to keep track of the service level for the last 30 periods.
        best_average_profit = -np.inf
        stable_count = 0  # Count of consecutive stable evaluations

        # Warm-up phase
        env.reset()
        K = 1
        k = 1
        while K <= warm_up_period:
            # Level I action
            state_level_1 = env.get_state_level_I()
            max_actions = actor_level1.output_shape[1]
            valid_actions_level1 = list(range(env.action_space_level_I.n))
            action_mask = np.zeros(max_actions)
            action_mask[valid_actions_level1] = 1
            action_probs = actor_level1.predict([np.expand_dims(state_level_1, axis=0), np.expand_dims(action_mask, axis=0)])[0]
            action = np.argmax(action_probs)
            env.step(action)
            k = 1

            while k <= T:
                # Level II action
                #logging.debug(f"Level 2 sampling for K = {K}")
                state_level_2 = env.get_state_level_II()
                max_actions = actor_level2.output_shape[1]
                valid_actions_level2 = list(range(env.action_space_level_II.n))
                action_mask = np.zeros(max_actions)
                action_mask[valid_actions_level2] = 1
                action_probs = actor_level2.predict([np.expand_dims(state_level_2, axis=0), np.expand_dims(action_mask, axis=0)])[0]
                action = np.argmax(action_probs)
                env.step(action)
                k += 1
            K += 1

        # Evaluation phase
        while K <= num_periods:
            period_profit = 0  # Initialize the period profit

            # Level I action
            state_level_1 = env.get_state_level_I()
            max_actions = actor_level1.output_shape[1]
            valid_actions_level1 = list(range(env.action_space_level_I.n))
            action_mask = np.zeros(max_actions)
            action_mask[valid_actions_level1] = 1
            action_probs = actor_level1.predict([np.expand_dims(state_level_1, axis=0), np.expand_dims(action_mask, axis=0)])[0]
            action = np.argmax(action_probs)
            _, reward, done, _ = env.step(action)
            period_profit += reward  # Add the reward from the level I action

            # Level II loop
            k = 1
            offline_demand_fulfilled_flags = []  # Initialize list to track if offline demand was fulfilled
            online_demand_fulfilled_flags = []  # Initialize list to track if online demand was fulfilled

            while k <= T:
                # Level II action
                state_level_2 = env.get_state_level_II()
                max_actions = actor_level2.output_shape[1]
                valid_actions_level2 = list(range(env.action_space_level_II.n))
                action_mask = np.zeros(max_actions)
                action_mask[valid_actions_level2] = 1
                action_probs = actor_level2.predict([np.expand_dims(state_level_2, axis=0), np.expand_dims(action_mask, axis=0)])[0]
                action = np.argmax(action_probs)
                _, reward, done, info = env.step(action)

                period_profit += reward  # Accumulate the reward from the level II action

                # Tracking demand fulfillment
                offline_demand_fulfilled_flags.append(info['offline_demand_fulfilled'])
                online_demand_fulfilled_flags.append(info['online_demand_fulfilled'])

                k += 1

            # Determine if all sub-periods had all demand fulfilled
            if sum(offline_demand_fulfilled_flags) == T:
                cycle_service_list_offline.append(1)
            else:
                cycle_service_list_offline.append(0)

            if sum(online_demand_fulfilled_flags) == T:
                cycle_service_list_online.append(1)
            else:
                cycle_service_list_online.append(0)

            # Maintain the cycle_service_list length to 30
            if len(cycle_service_list_offline) > 30:
                cycle_service_list_offline.pop(0)
            if len(cycle_service_list_online) > 30:
                cycle_service_list_online.pop(0)

            # Store the profit for this period
            recent_profits.append(period_profit)
            if len(recent_profits) > 30:
                recent_profits.pop(0)

            K += 1

            # Calculate the average profit over the last 30 periods
            if len(recent_profits) == 30:
                average_profit = np.mean(recent_profits)

                # Log the evaluation result
                logging.debug(f"Evaluation over {K} periods: Average Profit = {average_profit}")

                # Check for convergence
                if abs(average_profit - best_average_profit) / best_average_profit <= 0.005:
                    stable_count += 1
                else:
                    stable_count = 0

                best_average_profit = max(best_average_profit, average_profit)
                # Calculate the cycle service levels
                cycle_service_level_offline = np.mean(cycle_service_list_offline) if cycle_service_list_offline else 0
                cycle_service_level_online = np.mean(cycle_service_list_online) if cycle_service_list_online else 0
                if stable_count >= 3:
                    logging.debug("Policy has converged.")
                    return average_profit, cycle_service_level_offline, cycle_service_level_online

        return np.mean(recent_profits) if recent_profits else 0, cycle_service_level_offline, cycle_service_level_online


    def train(self, env, T):
        """
        Trains the PPO agent in the given environment.

        Args:
            env: The environment to train on. Must be compatible with the actor's action space.
            K_MAX: The number of episodes to train the policy for.
            T (int): Number of sub-periods in a period.
        """
        K_MAX = self.K_MAX_hyppar
        B = self.B_hyppar   # Adjust if necessary for the specific value


        episode = 0
        for episode in range(K_MAX):
            env.reset()  # Reset environment
            K = 1
            k = 1
            episode += 1
            logging.debug(f"Episode: {episode}")
            # Initialize empty lists to accumulate data
            cumulative_states1, cumulative_dones1, cumulative_profits1, cumulative_future_profits1, cumulative_action_masked_list1 = [], [], [], [], []
            cumulative_states2, cumulative_dones2, cumulative_profits2, cumulative_future_profits2, cumulative_action_masked_list2  =  [], [], [], [], []
            
            # Start episode
            while K <= B:
                logging.debug(f"Level 1 sampling for K = {K}, Episode: {episode}")
                # Level 1 sampling
                states1, dones1, profits1, future_profits1, action_masked_list1  = self.sampling_level1(
                    env=env,  
                    valid_actions_level1=list(range(env.action_space_level_I.n)),
                )

                # Accumulate data
                cumulative_action_masked_list1.extend(action_masked_list1)
                cumulative_states1.extend(states1)
                cumulative_dones1.extend(dones1)
                cumulative_profits1.extend(profits1)
                cumulative_future_profits1.extend(future_profits1)

                if K == B:
                    logging.debug(f"Updating level 1 networks at K = {K}")
                    # Convert accumulated data to arrays
                    states_array1 = np.array(cumulative_states1)
                    dones_array1 = np.array(cumulative_dones1)
                    profits_array1 = np.array(cumulative_profits1)
                    future_profits_array1 = np.array(cumulative_future_profits1)
                    cumulative_action_masked_list1 = np.array(cumulative_action_masked_list1)
                    # Update Level 1 networks
                    self.update_networks_level1(
                        states=states_array1,
                        dones=dones_array1,
                        profits=profits_array1,
                        future_profits=future_profits_array1,
                        action_masked_list1=cumulative_action_masked_list1
                    )

                    # Clear cumulative data for next Episiode
                    cumulative_states1, cumulative_dones1, cumulative_profits1, cumulative_future_profits1 = [], [], [], []

                logging.debug(f"Level 2 sampling for K = {K}, k = {k}, Episode: {episode}")
                # Level 2 sampling
                while k <= T:
                    # Level 2 sampling
                    states2, dones2, profits2, future_profits2, action_masked_list2 = self.sampling_level2(
                        env=env,
                        valid_actions_level2=list(range(env.action_space_level_II.n)),
                    )

                    # Accumulate data
                    cumulative_states2.extend(states2)
                    cumulative_dones2.extend(dones2)
                    cumulative_profits2.extend(profits2)
                    cumulative_future_profits2.extend(future_profits2)
                    cumulative_action_masked_list2.extend(action_masked_list2)



                    if (k * K) == (T * B):
                        logging.debug(f"Updating level 2 networks at k = {k}")
                        # Convert accumulated data to arrays
                        states_array2 = np.array(cumulative_states2)
                        dones_array2 = np.array(cumulative_dones2)
                        profits_array2 = np.array(cumulative_profits2)
                        future_profits_array2 = np.array(cumulative_future_profits2)
                        cumulative_action_masked_array2 = np.array(cumulative_action_masked_list2)

                        # Update Level 2 networks
                        self.update_networks_level2(
                            states=states_array2,
                            dones=dones_array2,
                            profits=profits_array2,
                            future_profits=future_profits_array2,
                            action_masked_list2=cumulative_action_masked_array2,
                            periods = T
                        )

                        # Clear cumulative data for next batch
                        cumulative_states2, cumulative_dones2, cumulative_profits2, cumulative_future_profits2 = [], [], [], []


                    k += 1

                # Move to next state
                K += 1
                k = 1


        # Evaluate policy after Kmax episodes
        average_profit, cycle_service_level_offline, cycle_service_level_online = self.evaluate_policy(env, self.actor_level1, self.actor_level2, T)
        logging.debug(f"Average Profit: {average_profit}, Cycle Service Level Offline: {cycle_service_level_offline}, Cycle Service Level Online: {cycle_service_level_online}")

        logging.debug("Training complete.")
