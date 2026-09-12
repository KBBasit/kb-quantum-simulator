'''
Deutsch-Jozsa Algorithm
'''



def make_f(n, function_type):
    """
    Creates a function f: {0,1}^n -> {0,1} that is either constant or balanced.
    Parameters:
        n (int): The number of input bits.
        function_type (str): The type of function to create ('constant' or 'balanced').
    Returns:
        function: A function that takes a binary string of length n and returns 0 or 1.
    """
    if function_type == 'constant':
        return lambda x: 0
    elif function_type == 'balanced':
        def balanced_function(x):
            index = int(x, 2)
            return 0 if index < (2 ** n) / 2 else 1
        return balanced_function


def deutsch_jozsa_algorithm(n, function):

    return