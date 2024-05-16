import numpy as np

# Total amount to be distributed
total_amount = 100000

# Number of instances
num_instances = 5

# Concentration parameters for the Dirichlet distribution
# Equal values mean each instance is equally likely to receive any proportion of the total
alpha = np.ones(num_instances)  # This can be adjusted to emphasize different expected proportions


# Generate a sample from the Dirichlet distribution
proportions = np.random.dirichlet(alpha)

# Calculate the amounts for each instance
amounts = proportions * total_amount

# Print the results
print("Proportions:", proportions)
print("Amounts:", amounts)
