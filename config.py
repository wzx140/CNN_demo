# numbers of neurons in each convolution layer, 0 is the max pool, -1 is the average pool
conv_layers = [8, 0, 16, 0]

# (filter size, step, pad) in filters in each layers. For pad, fill in 'SAME' or 'VALID'.
filters = [(4, 1, 'SAME'), (8, 8, 'SAME'), (2, 1, 'SAME'), (4, 4, 'SAME')]

# the dims of full connected layers
fc_layers = [6]

learning_rate = 0.009

# adam, none is default value
beta1 = None
beta2 = None

num_epochs = 100

mini_batch_size = 64

img_path = ''

# print cost every 5-echo
print_cost = True
