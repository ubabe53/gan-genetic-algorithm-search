"""
Python File only for encoding function
"""

batch_size = {1: 125, 2: 250, 3: 325, 4: 500, 5: 550, 6: 700}
lr = {1: 0.001, 2: 0.005, 3: 0.0001, 4: 0.0005, 5: 0.00001, 6: 0.00005}
beta1 = {1: 0.98, 2: 0.95, 3: 0.90, 4: 0.85, 5: 0.80, 6: 0.75}
beta2 = {1: 0.999, 2: 0.995, 3: 0.990, 4: 0.9995, 5: 0.9999, 6: 0.985}
n_critic = {1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}
weight_gp = {1: 7, 2: 8, 3: 9, 4: 10, 5: 11, 6: 12}

dict_list = [batch_size, lr, beta1, beta2, n_critic, weight_gp]

def encoder(list):
    """
    :param list: The list representing the encoded solution
    :return: list representing the decoded solution
    """
    decoded_solution = []
    for i in range(len(list)):
        decoded_solution.append(dict_list[i][list[i]])
    return decoded_solution

