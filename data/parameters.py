parameters = {
    'initial_inventory': 10, # randomly chosen
    'return_window': 14, # M = 2 -> two periods = 14 sub-periods
    'return_probability': 0.036, # Probability per sub-period, when 0,4 for 14 sub-periods
    'T': 7, # sub_periods_per_period
    'lead_time': 2+1, # +1 so logic works out, because sub-period 0 exists
    'demand_rate_offline': 6, # Mü of Poisson distribution
    'demand_rate_online': 2, # Mü of Poisson distribution
    'max_periods': 512, # same as B_hyppar, has to be at least as larg # not important
    'selling_price': 100,
    'fulfillment_cost': 5, # online fullfillment cost / handling costs
    'holding_cost_offline': 1,
    'holding_cost_online': 0.5, # cheaper to hold
    'return_handling_cost': 5, 
    'procurement_cost': 30
}