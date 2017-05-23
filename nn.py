from file_utils import create_folder, write_json, read_json

import time
import numpy
import random
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle

from theano import tensor, function, config
from theano.tensor import basic, clip
from lasagne.updates import adam, nesterov_momentum
from lasagne.layers import InputLayer, DenseLayer, Conv1DLayer, BiasLayer, DropoutLayer,\
                           get_output, get_all_params, set_all_param_values,\
                           get_output_shape, ConcatLayer,get_all_param_values
from lasagne.nonlinearities import sigmoid, rectify
from lasagne.init import GlorotUniform
from theano.printing import Print
from lasagne.regularization import regularize_network_params, l2, l1

from file_utils import load_steps
from evaluate import eval_model

def my_loss(model, predictions, targets, regularization, params):
    predictions = predictions[0][0][params['left_border']:-params['right_border']]
    targets = targets[0][params['left_border']:-params['right_border']]
    loss = tensor.abs_(tensor.log((targets * predictions).sum() / targets.sum())) +\
           tensor.abs_(tensor.log(((1-targets) * (1-predictions)).sum() / (1-targets).sum()))
    reg_loss_l1 = regularize_network_params(model, l1) * 1e-4
    reg_loss_l2 = regularize_network_params(model, l2)
    if regularization:
        return loss + reg_loss_l1# + reg_loss_l2
    else:
        return loss

    
nonlinearity_mapping = {
    'rectify': rectify
}
def build_lasagne_model(architecture, dim, input_tensor, shape):
    last_layer = InputLayer(shape, input_var=input_tensor)
    print('Model shape:')
    for i, e in enumerate(architecture):
        dropout, layers = e
        print('  Layer {}:'.format(i))
        ll = []
        for l in layers:
            filter_size, num_filters, nonlinearity, pad = l
            cl = Conv1DLayer(last_layer, num_filters=num_filters, filter_size=(filter_size), nonlinearity= nonlinearity_mapping[nonlinearity], pad=pad)
            ll.append(cl)
            print('    - size: {}\tnum: {}'.format(filter_size, num_filters, get_output_shape(cl)))
        c = ConcatLayer(ll, axis=1)
        last_layer = DropoutLayer(c, p=dropout)
        print('    - dropout: {}'.format(dropout))
    return Conv1DLayer(last_layer, num_filters=1, filter_size=3, nonlinearity=sigmoid, pad='same')


class LasagneNN:
    def __init__(self, architecture, dim, params):
        t1 = time.time()
        self.t_in = tensor.ftensor3('inputs')  #  =X     float64
        self.t_out = tensor.imatrix('targets') # =Y_true int32
        self.input_shape = (None, dim, params['segment_size'],)
        self.output_shape  = (None, dim, params['segment_size'],)
        self.architecture = architecture
        self.model = build_lasagne_model(architecture, dim, self.t_in, self.input_shape)
        self.params = params
        self.trained = False
        self.dim = dim
        
        test_pred = get_output(self.model, deterministic=False)
        test_loss = my_loss(self.model, test_pred, self.t_out, False, params)
        test_loss_with_reg = my_loss(self.model, test_pred, self.t_out, True, params)
        test_acc  = tensor.mean(tensor.eq(tensor.argmax(test_pred, axis=1), self.t_out), dtype=config.floatX)
        
        self.eval_fn  = function([self.t_in, self.t_out], [test_loss, test_loss_with_reg, test_acc], allow_input_downcast=True)
        self.evaluate = function([self.t_in], get_output(self.model, self.t_in), allow_input_downcast=True)
        
        pred = get_output(self.model)
        loss_with_reg = my_loss(self.model, pred, self.t_out, True, params)
        params = get_all_params(self.model, trainable=True)
        updates = adam(loss_with_reg, params=params, learning_rate=0.0001)
        self.train_fn = function([self.t_in, self.t_out], loss_with_reg, updates=updates, allow_input_downcast=True)
        
        print('Neural network initialized in {:.2f}s'.format(time.time()-t1))
    
    def load_weights(self, weights):
        set_all_param_values(self.model, weights)
        self.trained = True
    
    def train(self, train_exs, test_exs, num_epochs, examples_by_epoch, data_dir, save_name, eval_during_training=False):
        trainN, testN = len(train_exs), len(test_exs)
        print('Training on {} examples, testing on {} examples.'.format(trainN, testN))
        print("Starting training...")
        
        ex_count = 0
        train_losses = []
        rps = []
        for epoch in range(1, num_epochs+1):
            if epoch < 3:
                train_losses = []
            start_time = time.time()
            print('Epoch {}/{} running...'.format(epoch, num_epochs), end='')
            train_loss = 0
            train_loss_with_reg = 0
            invalid_example = 0
            for z in range(examples_by_epoch):
                db, i, j = train_exs[ex_count%trainN]
                XY = load_steps(db, i, self.params)[j]
                x, y = numpy.reshape(XY[0], (1, 1, 5000)).astype('float16'), numpy.reshape(XY[1], (1, 5000)).astype('float16')
                ex_count += 1
                if numpy.sum(y) == 0.:
                    invalid_example += 1
                    continue
                tmp_loss = self.train_fn(x, y)
                train_loss += tmp_loss
                train_losses.append(train_loss/(z+1-invalid_example))
            print('Done in {:.3f}s!'.format(time.time() - start_time))
            nnn = examples_by_epoch - invalid_example
            print("  - training loss:\t\t{:.6f}".format(train_loss / nnn))
            
            if eval_during_training:
                # Eval on examples:
                test_loss = 0
                test_reg_loss = 0
                test_acc = 0
                invalid_example = 0
                for k in range(testN):
                    db, i, j = train_exs[k]
                    XY = load_steps(db, i, self.params)[j]
                    x, y = numpy.reshape(XY[0], (1, 1, 5000)).astype('float16'), numpy.reshape(XY[1], (1, 5000)).astype('float16')
                    if numpy.sum(y) == 0.:
                        invalid_example += 1
                        continue
                    tmp_loss, tmp_reg_loss, tmp_acc = self.eval_fn(x, y)
                    test_loss += tmp_loss
                    test_reg_loss += (tmp_reg_loss-tmp_loss)
                    test_acc  += tmp_acc
                N = (testN-invalid_example)
                print("  - test loss:\t\t\t{:.6f} | {:.6f} | {:.6f}".format(test_loss / N, test_reg_loss / N, (test_loss+test_reg_loss) / N))
                acc = test_acc / N * 100
                print("  - test accuracy:\t\t{:.4f} %".format(acc))
                plt.plot(train_losses, color='r')
                plt.plot(rps, color='g')
                plt.show()
                eval_model(test_exs, self.evaluate, self.params, plot_examples=True, nb=3, nearest_fpr=0.01, threshold=0.98, eval_margin=10)
            save_lasagne_nn_epoch(save_name, self, epoch, train_loss / examples_by_epoch)
        self.trained = True
    
    def evaluate(self, x, fs_target):
        if not self.trained:
            print('Neural Network not trained!')
            return
        fs_target, y_delay, segment_size, segment_step, normalized_steps = self.params

def save_lasagne_nn_epoch(save_name, nn, epoch, loss):
    name = '{}/epoch{}.loss{:.6f}'.format(save_name, int(epoch), loss)
    write_json({'arch': nn.architecture, 'dim': nn.dim, 'params': nn.params}, save_name + '.json')
    pickle.dump({'weights': get_all_param_values(nn.model)}, open(name + '.weights', 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)
        
        
def save_lasagne_nn(json_name, weights_name nn):
    write_json({'arch': nn.architecture, 'dim': nn.dim, 'params': nn.params}, json_name)
    pickle.dump({'weights': get_all_param_values(nn.model)}, open(weights_name, 'wb+'), protocol=pickle.HIGHEST_PROTOCOL)

    
def load_lasagne_nn(save_name):
    info = read_json(save_name)
    nn = LasagneNN(architecture=info['arch'], dim=info['dim'], params=info['params'])
    nn.load_weights(pickle.load(open(filepath,'rb')['weights']))
    return nn