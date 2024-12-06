import numpy as np
import gym
from scipy.stats import poisson
from utils.poisson_sampling import adjusted_poisson_pmf, sample_from_adjusted_poisson
from utils.logging_config import logger


class OmniChannelRetailerEnv(gym.Env):
    def __init__(self, parameters):
        """
        Initialize the environment with given parameters.

        Parameters:
        - parameters: A dictionary containing environment settings and hyperparameters.
        """
        super(OmniChannelRetailerEnv, self).__init__()
        self.parameters = parameters
        self.inventory_level = parameters['initial_inventory']
        self.orders = []  # List to track ordered quantities that haven not arrived yet
        # actually means unreturned products from previous periods
        self.returns = [0] * parameters['return_window']  # Initialize with zero sales for return window periods
        self.sold_online_R0 = [0] * parameters['T']
        self.current_period = 0
        self.current_sub_period = 0
        self.return_probability = parameters['return_probability']
        self.return_window = parameters['return_window']
        self.max_periods = parameters.get('max_periods', 512)  # Default to 512 periods if not specified
        
        # To truncate the poisson distributions
        self.D1_poisson_point_offline = poisson.ppf(0.99, parameters['demand_rate_offline'])
        self.D2_poisson_point_online = poisson.ppf(0.99, parameters['demand_rate_online'])
        self.pmf1 = adjusted_poisson_pmf(parameters['demand_rate_offline'], self.D1_poisson_point_offline)
        self.pmf2 = adjusted_poisson_pmf(parameters['demand_rate_online'], self.D2_poisson_point_online)
        self.pmf_offline = adjusted_poisson_pmf(self.parameters['demand_rate_offline'], self.D1_poisson_point_offline)
        self.pmf_online = adjusted_poisson_pmf(self.parameters['demand_rate_online'], self.D2_poisson_point_online)

        # Calculate the maximum expected inventory
        self.max_inventory = (parameters['T'] + parameters['lead_time']) * self.D1_poisson_point_offline + \
                             parameters['return_window'] * parameters['T'] * self.D2_poisson_point_online

        # Calculate maximum expected demand = max expected returns
        self.max_returns = (parameters['return_window'] + parameters['T']) * self.D2_poisson_point_online
        self.max_online_sales_per_subperiod = self.D2_poisson_point_online

        self.max_demand_period = (self.parameters['T'] + self.parameters['lead_time']) * self.D1_poisson_point_offline + (self.parameters['T'] + self.parameters['lead_time']) * self.D2_poisson_point_online


        # Calculate maximum expected demand
        self.expected_demand = (self.parameters['T'] + self.parameters['lead_time']) * self.D1_poisson_point_offline + (self.parameters['T'] + self.parameters['lead_time']) * self.D2_poisson_point_online


        # Define action spaces using gym.spaces.Box
        # Min: 0
        # Max: Expected demand minus inventory position
        self.action_space_level_I = gym.spaces.Box(
            low=0,
            high= max(self.expected_demand - self.inventory_level + 1, 1), # to include all possible values add 1 (for zero)
            dtype=np.float32
        )

        # Min: 0
        # Max: Inventory position
        self.action_space_level_II = gym.spaces.Box(
            low=0,
            high=self.inventory_level + 1,
            dtype=np.float32
        )  
        
        # Initialize action spaces based on the initial state
        self.update_action_spaces()


    def update_action_spaces(self):
        """
        Dynamically update action spaces based on the current state.
        for Level I (ordering) and Level II (rationing)
        """            
        self.expected_demand = (self.parameters['T'] + self.parameters['lead_time']) * self.D1_poisson_point_offline + (self.parameters['T'] + self.parameters['lead_time']) * self.D2_poisson_point_online
        self.action_space_level_I = gym.spaces.Discrete(max(int(self.expected_demand - self.inventory_level + 1), 1))
        self.action_space_level_II = gym.spaces.Discrete(self.inventory_level + 1)  


    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
        - Initial state as a numpy array.
        """
        self.inventory_level = self.parameters['initial_inventory']
        self.orders = []
        self.returns = [0] * self.return_window # Make sure the unreturned products are reset correctly
        self.sold_online_R0 = [0] * self.parameters['T']
        self.current_period = 0
        self.current_sub_period = 0

        self.update_action_spaces()  # Update action spaces based on reset state

        return self.get_state_level_I()  # depending on the level you want to reset to

    
    def step(self, action):
        """
        Apply an action to the environment and update its state.

        Parameters:
        - action: The action to be taken. For Level I, it's the order quantity. For Level II, it's the rationing decision.

        Returns:
        - next_state: The next state of the environment as a numpy array.
        - reward: The reward obtained from the action.
        - done: A boolean flag indicating if the episode has ended.
        - info: Additional information.
        """
        if self.current_sub_period == 0:
            next_state, reward, done, info = self.step_level_I(action)
        else:
            next_state, reward, done, info = self.step_level_II(action)
        
        # Update sub-period and period counters
        self.current_sub_period += 1
        if self.current_sub_period > self.parameters['T']:
            self.current_period += 1
            self.current_sub_period = 0
            self.sold_online_R0 = [0] * self.parameters['T'] # Reset the sold_online_R0 list every period

        
        self.update_action_spaces()  # Update action spaces based on new state

        return next_state, reward, done, info


    def step_level_I(self, action):
        """
        Apply a Level I action to the environment and update its state.

        Parameters:
        - action: The action to be taken. For Level I, it's the order quantity.

        Returns:
        - next_state: The next state of the environment as a numpy array.
        - reward: The reward obtained from the action.
        - done: A boolean flag indicating if the episode has ended.
        - info: Additional information (empty dictionary here).
        """
        q = action  # Level I action (ordering)
        self.orders.append((q, self.current_sub_period + self.parameters['lead_time']))  # Append order with delivery sub-period

        # Calculate reward
        # based on ordering quantity
        reward = self.calculate_reward(0, 0, 0, q, [0])

        # Construct next state for Level I
        next_state = self.get_state_level_I()

        # Check if the episode is done
        done = self.inventory_level <= 0 or self.current_period >= self.max_periods

        # Render the current state of the environment
        #self.render()

        return next_state, reward, done, {}


    def step_level_II(self, action):
        """
        Apply a Level II action to the environment and update its state.

        Parameters:
        - action: The action to be taken. For Level II, it's the rationing decision.

        Returns:
        - next_state: The next state of the environment as a numpy array.
        - reward: The reward obtained from the action.
        - done: A boolean flag indicating if the episode has ended.
        - info: Additional information (empty dictionary here).
        """
        # make sure action space is a updated before steping
        a = action  # Level II action (rationing)

        # Process deliveries of outstanding orders
        replenishment = sum(order[0] for order in self.orders if order[1] == self.current_sub_period)
        self.orders = [order for order in self.orders if order[1] != self.current_sub_period]  # Remove delivered orders

        # Sample demand for offline and online channels
        demand_offline, demand_online = self.sample_demand()
        returns = self.calculate_returns()  # Calculate returns for this sub-period
        total_returns = sum(returns)
        # Calculate sales based on demand and rationing decision
        sold_offline = min(a, demand_offline)

        sold_online = min(self.inventory_level - a, demand_online)
        # Update inventory level
        self.inventory_level += replenishment - sold_offline - sold_online + total_returns
        self.update_returns_list(sold_online)  # Update list tracking online sales that might return

        # Calculate reward
        reward = self.calculate_reward(sold_offline, sold_online, a, replenishment, returns)
        
        # Construct next state for Level II
        next_state = self.get_state_level_II()

        # Check if the episode is done
        done = self.inventory_level <= 0 or self.current_period >= self.max_periods

        # Determine if offline and online demand were fulfilled
        offline_demand_fulfilled = 1 if sold_offline == demand_offline else 0
        online_demand_fulfilled = 1 if sold_online == demand_online else 0

        # Render the current state of the environment
        #self.render()

        return next_state, reward, done, {'offline_demand_fulfilled': offline_demand_fulfilled, 'online_demand_fulfilled': online_demand_fulfilled}


    def get_state_level_I(self):
        max_inventory = self.max_inventory
        max_returns = self.max_returns 

        # Inventory position (I) Scaled to [-5, 5]
        #inventory_position = np.array([self.inventory_level])
        inventory_position = (self.inventory_level / max_inventory) * 10 - 5

        # Total unreturned products (sum of returns list) scaled to [-5, 5]
        #total_unreturned_products = np.array([sum(self.returns)])
        total_unreturned_products = (sum(self.returns) / max_returns) * 10 - 5

        # Combine into a single state vector
        state = np.array([inventory_position, total_unreturned_products])
        return state


    def get_state_level_II(self):
        max_inventory = self.max_inventory
        max_returns = self.max_returns
        max_online_sales_per_period = max(self.max_online_sales_per_subperiod * self.current_sub_period, self.max_online_sales_per_subperiod)
        max_order_quantity = self.max_demand_period - self.inventory_level

        # Inventory position (I), scaled to [-5, 5]
        inventory_position = (self.inventory_level / max_inventory) * 10 - 5
        
        # Number of products sold online in the previous sub-periods (R0), scaled to [-5, 5]
        R0 = (sum(self.sold_online_R0) / max_online_sales_per_period) * 10 - 5 
        
        # Unreturned products from previous periods (R), scaled to [-5, 5]
        unreturned_products = sum((np.array(self.returns[:self.return_window])) / max_returns) * 10 - 5

        # Outstanding replenishment order (Q)
        Q = sum(order[0] for order in self.orders) / max(max_order_quantity,0) * 10 - 5
        
        # Current sub-period (t)
        t = (self.current_sub_period / self.parameters['T']) * 10 - 5
        
        # Combine into a single state vector
        state = np.array([inventory_position, unreturned_products, R0, Q, t])
        #logging.debug(f"GETSTATE State Level II: {state}")
                
        return state
    

    def update_returns_list(self, sold_online):
        """
        # adds new sold online products to the returns list
        Parameters:
        - sold_online: The number of products sold online in the current sub-period.
        """
        self.returns.append(sold_online)
        self.sold_online_R0.append(sold_online)
        if len(self.returns) > self.parameters['return_window']:
            self.returns.pop(0)  # Remove sales older than the return window


    def calculate_returns(self):
        """
        Calculate the number of returned products for each period in the return window.

        Returns:
        - A list of returned products for each period in the return window.
        # updates returns list (subtracts returns)
        """
        period_returns = []
        for i in range(len(self.returns)):
            sales = self.returns[i]
            period_return = np.random.binomial(n=sales, p=self.return_probability)
            period_returns.append(period_return)
            self.returns[i] -= period_return  # Subtract actual returns from potential returns

        return period_returns


    def calculate_reward(self, sold_offline, sold_online, rationing_decision, replenishment, returns):
        """
        Calculate the reward for the current state and action.

        Parameters:
        - sold_offline: Number of products sold offline.
        - sold_online: Number of products sold online.
        - rationing_decision: Quantity allocated to offline sales.
        - replenishment: Quantity of products replenished.
        - returns: List of products returned in the current sub-period.

        Returns:
        - Total reward as a float.
        """
        p = self.parameters['selling_price']
        cu = self.parameters['fulfillment_cost']
        ch1 = self.parameters['holding_cost_offline']
        ch2 = self.parameters['holding_cost_online']
        cr = self.parameters['return_handling_cost']
        cp = self.parameters['procurement_cost']

        handling_returns_costs = cr * sum(returns) + p * sum(returns) # return handling costs + refund
        revenue_offline = p * sold_offline # sold offline
        revenue_online = (p - cu) * sold_online # sold online - fulfillment costs	
        holding_costs = ch1 * rationing_decision + ch2 * (self.inventory_level - rationing_decision)
        procurement_costs = cp * replenishment
        total_reward = revenue_offline + revenue_online - holding_costs - handling_returns_costs - procurement_costs

        return total_reward


    def sample_demand(self):
        """
        Sample demand for offline and online channels.

        Returns:
        - demand_offline: Sampled demand for the offline channel.
        - demand_online: Sampled demand for the online channel.
        """

        # Generate samples using the custom sampling function
        demand_offline = sample_from_adjusted_poisson(self.pmf_offline)
        demand_online = sample_from_adjusted_poisson(self.pmf_online)

        return demand_offline[0], demand_online[0] #besser direkt als int wiedergeben

        #demand_offline = np.random.poisson(self.parameters['demand_rate_offline'])
        #demand_online = np.random.poisson(self.parameters['demand_rate_online'])
        #return demand_offline, demand_online


    def render(self, mode='human'):
        """
        Render the current state of the environment to the console.

        Parameters:
        - mode: The mode to render with. (default: 'human')
        """
        logging.debug(f"Period: {self.current_period}, Sub-period: {self.current_sub_period}")
        logging.debug(f"Inventory Level: {self.inventory_level}")
        logging.debug(f"Outstanding Orders: {self.orders}")
        logging.debug(f"Returns: {self.returns}")
        logging.debug("-" * 40)