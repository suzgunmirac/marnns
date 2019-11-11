# Import libraries and relevant dependencies
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import string
import argparse
from torch.autograd import Variable

# Dyck library
from tasks.dyck_generator import DyckLanguage

# Stack-augmented architectures (Stack-RNN + Stack-LSTM)
from models.rnn_models import VanillaRNN, SRNN_Softmax, SRNN_Softmax_Temperature, SRNN_GumbelSoftmax 
from models.lstm_models import VanillaLSTM, SLSTM_Softmax, SLSTM_Softmax_Temperature, SLSTM_GumbelSoftmax 
# Baby Neural Turing Machine (Baby-NTM)
from models.ntm_models import BNTM_Softmax, BNTM_SoftmaxTemperature, BNTM_GumbelSoftmax

# Set default tensor type "double"
torch.set_default_tensor_type('torch.DoubleTensor')

# GPU/CPU Check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## GPU stuff
print (device)

# Output threshold
epsilon = 0.5

randomseed_num = 1729
print ('RANDOM SEED: {}'.format(randomseed_num))
random.seed (randomseed_num)
np.random.seed (randomseed_num)
torch.manual_seed(randomseed_num)

def get_args ():
	parser = argparse.ArgumentParser(description='parser')
	# Dyck language parameters
	parser.add_argument('--num_par', default=2, type=int, help='number of parentheses pairs')
	parser.add_argument('--p_val', default = 0.5, type=float, help='p value for the PCFG for Dyck-n')
	parser.add_argument('--q_val', default = 0.25, type=float, help='q value for the PCFG for Dyck-n')
	# Training and test corpura parameters
	parser.add_argument('--training_window', default=[2,50], type=int, help='training corpus window')
	parser.add_argument('--training_size', default=5000, type=int, help='number of training samples')
	parser.add_argument('--test_size', default=1000, type=int, help='number of test samples')
	# Model choice
	parser.add_argument ('--model_name', type=str, default='baby_ntm_softmax', 
		choices=['vanilla_rnn', 'vanilla_lstm', 'joulin_mikolov_stack_rnn',
		'stack_rnn_softmax', 'stack_rnn_softmax_temp', 'stack_rnn_gumbel_softmax', 
		'stack_lstm_softmax', 'stack_lstm_softmax_temp', 'stack_lstm_gumbel_softmax', 
		'baby_ntm_softmax', 'baby_ntm_softmax_temp', 'baby_ntm_gumbel_softmax'],
		help='model choice')
	parser.add_argument('--n_layers', default=1, type=int, help='number of layers')
	parser.add_argument('--n_hidden', default=8, type=int, help='hidden dimension')
	# Stack/Memory size
	parser.add_argument('--memory_size', default=104, type=int, help='memory/stack size')
	parser.add_argument('--memory_dim', default=5, type=int, help='memory/stack dimension')
	# Temperature + Annealing rate
	parser.add_argument('--temp_min', default=0.5, type=float, help='minimum temperature value')
	parser.add_argument('--anneal_rate', default=0.0001, type=float, help='anneal rate')
	# Learning rate and number of epochs
	parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
	parser.add_argument('--n_epoch', default=3, type=int, help='number of epochs')
	# Save/load model 
	parser.add_argument('--save_path', default='', type=str, help='where you want to save the model')
	parser.add_argument('--load_path', default='', type=str, help='path of the saved model')
	# Parse known arguments
	args, _ = parser.parse_known_args ()
	return args

def choose_model (model_name, n_hidden, vocab_size, n_layers, memory_size, memory_dim):
	if model_name == 'vanilla_rnn':
		model = VanillaRNN (n_hidden, vocab_size, vocab_size, n_layers)
	elif model_name == 'vanilla_lstm':
		model = VanillaLSTM (n_hidden, vocab_size, vocab_size, n_layers)
	elif model_name == 'stack_rnn_softmax':
		model = SRNN_Softmax (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'stack_lstm_softmax':
		model = SLSTM_Softmax (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'baby_ntm_softmax':
		model = BNTM_Softmax (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'stack_rnn_softmax_temp':
		model = SRNN_Softmax_Temperature (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'stack_lstm_softmax_temp':
		model = SLSTM_Softmax_Temperature (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'baby_ntm_softmax_temp':
		model = BNTM_SoftmaxTemperature (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'stack_rnn_gumbel_softmax':
		model = SRNN_GumbelSoftmax (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'stack_lstm_gumbel_softmax':
		model = SLSTM_GumbelSoftmax (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'baby_ntm_gumbel_softmax':
		model = BNTM_GumbelSoftmax (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	elif model_name == 'joulin_mikolov_stack_rnn':
		model = SRNN_Softmax (n_hidden, vocab_size, vocab_size, n_layers, memory_size, memory_dim)
	else:
		print ('Oh dear... something went wrong...')
	return model.to(device)

def main ():
	args = get_args()

	# Parameters of the Probabilistic Dyck Language 
	NUM_PAR = args.num_par
	MIN_SIZE = args.training_window[0]
	MAX_SIZE = args.training_window[1]
	P_VAL = args.p_val
	Q_VAL = args.q_val

	# Number of samples in the training corpus
	TRAINING_SIZE = args.training_size
	# Number of samples in the test corpus
	TEST_SIZE = args.test_size

	# Create a Dyck language generator
	Dyck = DyckLanguage (NUM_PAR, P_VAL, Q_VAL)
	all_letters = word_set = Dyck.return_vocab ()
	n_letters = vocab_size = len (word_set)

	print('Loading data...')

	training_input, training_output, st = Dyck.training_set_generator (TRAINING_SIZE, MIN_SIZE, MAX_SIZE)
	test_input, test_output, st2 = Dyck.training_set_generator (TEST_SIZE, MAX_SIZE + 2, 2 * MAX_SIZE)

	for i in range (1):
	    print (training_input[i])
	    print (training_output[i])
	    print (Dyck.lineToTensor(training_input[i]))
	    print (Dyck.lineToTensorSigmoid(training_output[i]))
	    print (test_input[i])
	    print (test_output[i])

	# Number of hidden units
	n_hidden = args.n_hidden
	# Number of hidden layers
	n_layers = args.n_layers
	# Memory size
	memory_size = args.memory_size
	memory_dim = args.memory_dim

	# Parameters for the temperature-based methods
	temp = 1.0
	temp_min = args.temp_min
	ANNEAL_RATE = args.anneal_rate

	## Stack-RNN with Softmax
	model = choose_model (args.model_name, n_hidden, vocab_size, n_layers, memory_size, memory_dim)
	# Learning rate
	learning_rate = args.lr
	# Minimum Squared Error (MSE) loss
	criterion = nn.MSELoss() 
	# Adam optimizer (https://arxiv.org/abs/1412.6980)
	optim = torch.optim.Adam(model.parameters(), lr = learning_rate)

	print ('Model details:')
	print (model)

	if args.load_path == '':
		# Number of epochs to train our model
		epoch_num = args.n_epoch

		# Arrays for loss and "moving" accuracy per epoch
		loss_arr = []
		correct_arr = []
		for epoch in range(1, epoch_num + 1):
			print ('Epoch: {}'.format(epoch))
			# Total loss per epoch
			total_loss = 0
			# Total number of "correctly" predicted samples so far in the epoch
			counter = 0

			for i in range (TRAINING_SIZE):
				len_input = len (training_input[i])
				# Good-old zero grad
				model.zero_grad ()
				# Initialize the hidden state
				hidden = model.init_hidden()
				# Initialize the memory 
				memory = torch.zeros (memory_size, memory_dim).to(device)
				# Target values
				target = Dyck.lineToTensorSigmoid(training_output[i]).to(device) 
				# Output values
				output_vals = torch.zeros (target.shape).to(device)
				# Temperature update -- can uncomment the following if no temp used
				temp = np.maximum(temp * np.exp(-ANNEAL_RATE), temp_min)

				for j in range (len_input):
					output, hidden, memory = model (Dyck.lineToTensor(training_input[i][j]).to(device), hidden, memory, temp)
					output_vals [j] = output
	            
				# MSE (y, y_bar)
				loss = criterion (output_vals, target)
				# Add the current loss to the total loss
				total_loss += loss.item()
				# Backprop! 
				loss.backward ()
				optim.step ()
	            
				# Print the performance of the model every 500 steps
				if i % 500 == 0:
					print ('Sample Number {}: '.format(i))
					print ('Input : {}'.format(training_input[i]))
					print ('Output: {}'.format(training_output[i]))
					print ('* Counter: {}'.format(counter))
					print ('* Avg Loss: {}'.format(total_loss/(i+1))) 

				# Binarize the entries based on the output threshold
				out_np = np.int_(output_vals.detach().numpy() >= epsilon)
				target_np = np.int_(target.detach().numpy())

				# "Moving" training accuracy
				if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
					counter += 1
	                
				# At the end of the epoch, append our total loss and "moving" accuracy
				if i == TRAINING_SIZE - 1:
					print ('* Moving Accuracy: {}%'.format(float(counter)/TRAINING_SIZE))
					loss_arr.append (total_loss)
					correct_arr.append(counter) 

			print ('Moving Accuracy %: ', correct_arr)
			print ('Loss: ', loss_arr)
			
			save_path = args.save_path

			if save_path == '':
				save_path = 'models/{}_model_weights.pth'.format(args.model_name)

			# Save the model weights
			torch.save(model.state_dict(), save_path)


	else:
		# Load the model weights
		model.load_state_dict(torch.load(args.load_path))
	
	# Training set accuracy 
	correct_num = test_model (model, Dyck, training_input, training_output, temp)
	print ('Training accuracy: {}.'.format(correct_num))

	# Test set accuracy 
	correct_num = test_model (model, Dyck, test_input, test_output, temp)
	print ('Test accuracy: {}.'.format(correct_num))



def test_model (model, Dyck, data_input, data_output, temp):
    # Turn on the eval mode
    model.eval()
    # Total number of "correctly" predicted samples
    correct_num = 0
    with torch.no_grad():
        for i in range (len(data_output)):
            len_input = len (data_input[i])
            model.zero_grad ()
            # Initialize the hidden state
            hidden = model.init_hidden()
            # Initialize the memory
            memory = torch.zeros (model.memory_size, model.memory_dim).to(device)
            # Target values
            target = Dyck.lineToTensorSigmoid(data_output[i]).to(device) 
            # Output values
            output_vals = torch.zeros (target.shape).to(device)
            
            for j in range (len_input):
                output, hidden, memory = model (Dyck.lineToTensor(data_input[i][j]).to(device), hidden, memory, temp)
                output_vals [j] = output

            # Binarize the entries based on the output threshold
            out_np = np.int_(output_vals.detach().cpu().numpy() >= epsilon)
            target_np = np.int_(target.detach().cpu().numpy())
            
            # (Double-)check whether the output values and the target values are the same
            if np.all(np.equal(out_np, target_np)) and (out_np.flatten() == target_np.flatten()).all():
                # If so, increase `correct_num` by one
                correct_num += 1
                
    return float(correct_num)/len(data_output) * 100, correct_num

if __name__ == "__main__":
    main()