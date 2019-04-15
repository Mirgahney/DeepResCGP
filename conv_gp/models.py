import numpy as np
import tensorflow as tf
import gpflow
from gpflow import settings, kernels, features

from doubly_stochastic_dgp.dgp import DGP_Base
from doubly_stochastic_dgp.layers import SVGP_Layer
from kernels import ConvKernel, PatchInducingFeatures, AdditivePatchKernel
from layers import ConvLayer
from views import FullView, RandomPartialView
from mean_functions import Conv2dMean, IdentityConv2dMean
from sklearn import cluster

import utils_res  

def parse_ints(int_string):
    if int_string == '':
        return []
    else:
        return [int(i) for i in int_string.split(',')]

def image_HW(patch_count):
    if type(patch_count) == tf.Dimension :
        patch_count = tf.dimension_value(patch_count)
    image_height = int(np.sqrt(patch_count))
    return [image_height, image_height]

def select_initial_inducing_points(X, M):
    kmeans = cluster.KMeans(n_clusters=M, init='k-means++', n_jobs=-1)
    kmeans.fit(X)
    return kmeans.cluster_centers_

def identity_conv(NHWC_X, filter_size, feature_maps_in, feature_maps_out, stride, padding = 'VALID'):
    conv = IdentityConv2dMean(filter_size, feature_maps_in, feature_maps_out, stride, padding)
    sess = conv.enquire_session()
    if type(NHWC_X.shape[0]) == tf.Dimension:
        batch = tf.dimension_value(NHWC_X.shape[0])
        # print(batch)
        with sess.as_default():
            NHWC_X = NHWC_X.eval()
    else:
        batch = NHWC_X.shape[0]
        # print(batch)
    random_images = np.random.choice(np.arange(batch), size=1000)
    return sess.run(conv(NHWC_X[random_images]))

class ModelBuilder(object):
    def __init__(self, flags, NHWC_X_train, Y_train, model_path=None):
        self.flags = flags
        self.X_train = NHWC_X_train
        self.Y_train = Y_train
        self.model_path = model_path
        self.global_step = None
        self.is_train = True

    def build(self):
        Ms = parse_ints(self.flags.M)
        feature_maps = parse_ints(self.flags.feature_maps)
        strides = parse_ints(self.flags.strides)
        filter_sizes = parse_ints(self.flags.filter_sizes)

        loaded_parameters = {}
        if self.flags.load_model is not None:
            global_step, loaded_parameters = self._load_layer_parameters(Ms)
            self.global_step = global_step

        assert len(strides) == len(filter_sizes)
        assert len(feature_maps) == (len(Ms) - 1)

        # conv_layers, H_X = self._conv_layers(Ms[0:-1], feature_maps, strides, filter_sizes,
        #         loaded_parameters)
        conv_layers, H_X = self._res_conv_layers(Ms = Ms[0:-1], feature_maps = feature_maps, strides = strides, filter_sizes = filter_sizes,
                loaded_parameters = loaded_parameters)
        print('layers ', conv_layers)
        print('befor last layer ', H_X.shape)
        last_layer_parameters = self._last_layer_parameters(loaded_parameters)
        last_layer = self._last_layer(H_X, Ms[-1], filter_sizes[-1], strides[-1],
                last_layer_parameters)
        # print(H_X.shape)

        layers = conv_layers + [last_layer]
        X = self.X_train.reshape(-1, np.prod(self.X_train.shape[1:]))
        return DGP_Base(X, self.Y_train,
                likelihood=gpflow.likelihoods.MultiClass(10),
                num_samples=self.flags.num_samples,
                layers=layers,
                minibatch_size=self.flags.batch_size, name='DGP')

    # def _conv_layers(self, Ms, feature_maps, strides, filter_sizes, loaded_parameters={}):
    #     H_X = self.X_train
    #     layers = []
    #     for i in range(len(feature_maps)):
    #         M = Ms[i]
    #         feature_map = feature_maps[i]
    #         filter_size = filter_sizes[i]
    #         stride = strides[i]
    #         layer_params = loaded_parameters.get(i)

    #         conv_layer, H_X = self._conv_layer(H_X, M, feature_map, filter_size, stride, layer_params)
    #         layers.append(conv_layer)
    #     return layers, H_X

    def _conv_layer(self, NHWC_X, M, feature_map, filter_size, stride, padding = 'VALID' ,layer_params=None):
        if layer_params is None:
            layer_params = {}
        # if padding == 'same':
        #     npad = ((0,0),(1,1),(1,1),(0,0))
        #     NHWC_X = np.pad(NHWC_X, pad_width=npad, mode='constant', constant_values=0)

        NHWC = NHWC_X.shape
        view = FullView(input_size=NHWC[1:3],
                filter_size=filter_size,
                feature_maps=NHWC[3],
                stride=stride)

        if self.flags.identity_mean:
            conv_mean = Conv2dMean(filter_size, NHWC[3], feature_map,
                    stride=stride)
        else:
            conv_mean = gpflow.mean_functions.Zero()
        conv_mean.set_trainable(False)

        output_shape = image_HW(view.patch_count) + [feature_map]

        H_X = identity_conv(NHWC_X, filter_size, NHWC[3], feature_map, stride, padding)
        if len(layer_params) == 0:
            conv_features = PatchInducingFeatures.from_images(
                    NHWC_X,
                    M,
                    filter_size)
        else:
            conv_features = PatchInducingFeatures(layer_params.get('Z'))

        patch_length = filter_size ** 2 * NHWC[3]
        if self.flags.base_kernel == 'rbf':
            lengthscales = layer_params.get('base_kernel/lengthscales', 5.0)
            variance = layer_params.get('base_kernel/variance', 5.0)
            base_kernel = kernels.RBF(patch_length, variance=variance, lengthscales=lengthscales)
        elif self.flags.base_kernel == 'acos':
            base_kernel = kernels.ArcCosine(patch_length, order=0)
        else:
            raise ValueError("Not a valid base-kernel value")

        q_mu = layer_params.get('q_mu')
        q_sqrt = layer_params.get('q_sqrt')

        conv_layer = ConvLayer(
            base_kernel=base_kernel,
            mean_function=conv_mean,
            feature=conv_features,
            view=view,
            white=self.flags.white,
            gp_count=feature_map,
            q_mu=q_mu,
            q_sqrt=q_sqrt)

        if q_sqrt is None:
            # Start with low variance.
            conv_layer.q_sqrt = conv_layer.q_sqrt.value * 1e-5

        return conv_layer, H_X

    def _resconv_layer(self, NHWC_X, M, feature_map, filter_size, stride, padding = 'SAME' ,layer_params=None):
        if layer_params is None:
            layer_params = {}
        # if padding == 'same':
        #     npad = ((0,0),(1,1),(1,1),(0,0))
        #     NHWC_X = np.pad(NHWC_X, pad_width=npad, mode='constant', constant_values=0)

        NHWC = NHWC_X.shape
        # shortcut = NHWC_X
        view = FullView(input_size=NHWC[1:3],
                filter_size=filter_size,
                feature_maps=NHWC[3],
                stride=stride)

        if self.flags.identity_mean:
            conv_mean = Conv2dMean(filter_size, NHWC[3], feature_map,
                    stride=stride)
        else:
            conv_mean = gpflow.mean_functions.Zero()
        conv_mean.set_trainable(False)

        output_shape = image_HW(view.patch_count) + [feature_map]

        H_X = identity_conv(NHWC_X, filter_size, NHWC[3], feature_map, stride, padding)
        if len(layer_params) == 0:
            conv_features = PatchInducingFeatures.from_images(
                    NHWC_X,
                    M,
                    filter_size)
        else:
            conv_features = PatchInducingFeatures(layer_params.get('Z'))

        patch_length = filter_size ** 2 * NHWC[3]
        if self.flags.base_kernel == 'rbf':
            lengthscales = layer_params.get('base_kernel/lengthscales', 5.0)
            variance = layer_params.get('base_kernel/variance', 5.0)
            base_kernel = kernels.RBF(patch_length, variance=variance, lengthscales=lengthscales)
        elif self.flags.base_kernel == 'acos':
            base_kernel = kernels.ArcCosine(patch_length, order=0)
        else:
            raise ValueError("Not a valid base-kernel value")

        q_mu = layer_params.get('q_mu')
        q_sqrt = layer_params.get('q_sqrt')

        conv_layer = ConvLayer(
            base_kernel=base_kernel,
            mean_function=conv_mean,
            feature=conv_features,
            view=view,
            white=self.flags.white,
            gp_count=feature_map,
            q_mu=q_mu,
            q_sqrt=q_sqrt)

        if q_sqrt is None:
            # Start with low variance.
            conv_layer.q_sqrt = conv_layer.q_sqrt.value * 1e-5
        H_X = H_X + NHWC_X
        return conv_layer, H_X

    def _last_layer(self, H_X, M, filter_size, stride, layer_params=None):
        if layer_params is None:
            layer_params = {}

        NHWC = H_X.shape
        conv_output_count = np.prod(NHWC[1:])
        Z = layer_params.get('Z')
        q_mu = layer_params.get('q_mu')
        q_sqrt = layer_params.get('q_sqrt')

        if Z is not None:
            saved_filter_size = int(np.sqrt(Z.shape[1] / NHWC[3]))
            if filter_size != saved_filter_size:
                print("filter_size {} != {} for last layer. Resetting parameters.".format(filter_size, saved_filter_size))
                Z = None
                q_mu = None
                q_sqrt = None

        if self.flags.last_kernel == 'rbf':
            H_X = H_X.reshape(H_X.shape[0], -1)
            lengthscales = layer_params.get('lengthscales', 5.0)
            variance = layer_params.get('variance', 5.0)
            kernel = gpflow.kernels.RBF(conv_output_count, lengthscales=lengthscales, variance=variance,
                    ARD=True)
            if Z is None:
                Z = select_initial_inducing_points(H_X, M)
            inducing = features.InducingPoints(Z)
        else:
            lengthscales = layer_params.get('base_kernel/lengthscales', 5.0)
            variance = layer_params.get('base_kernel/variance', 5.0)
            input_dim = filter_size**2 * NHWC[3]
            view = FullView(input_size=NHWC[1:],
                    filter_size=filter_size,
                    feature_maps=NHWC[3],
                    stride=stride)
            if Z is None:
                inducing = PatchInducingFeatures.from_images(H_X, M, filter_size)
            else:
                inducing = PatchInducingFeatures(Z)
            patch_weights = layer_params.get('patch_weights')
            if self.flags.last_kernel == 'conv':
                kernel = ConvKernel(
                        base_kernel=gpflow.kernels.RBF(input_dim, variance=variance, lengthscales=lengthscales),
                        view=view, patch_weights=patch_weights)
            elif self.flags.last_kernel == 'add':
                kernel = AdditivePatchKernel(
                        base_kernel=gpflow.kernels.RBF(input_dim, variance=variance, lengthscales=lengthscales),
                        view=view, patch_weights=patch_weights)
            else:
                raise ValueError("Invalid last layer kernel")
        return SVGP_Layer(kern=kernel,
                    num_outputs=10,
                    feature=inducing,
                    mean_function=gpflow.mean_functions.Zero(output_dim=10),
                    white=self.flags.white,
                    q_mu=q_mu,
                    q_sqrt=q_sqrt)

    def _load_layer_parameters(self, Ms):
        parameters = np.load(self.model_path).item()
        global_step = parameters['global_step']
        del parameters['global_step']
        layer_params = {}

        def parse_layer_path(key):
            if 'layers' not in key:
                return None, None
            parts = key.split('/')
            return int(parts[2]), "/".join(parts[3:])

        for key, value in parameters.items():
            layer, path = parse_layer_path(key)
            if layer is None:
                continue
            layer_values = layer_params.get(layer, {})
            if 'q_mu' in path:
                layer_values['q_mu'] = value
            elif 'q_sqrt' in path:
                layer_values['q_sqrt'] = value
            elif 'Z' in path:
                layer_values['Z'] = value
            elif 'base_kernel/variance' in path:
                layer_values['base_kernel/variance'] = value
            elif 'base_kernel/lengthscales' in path:
                layer_values['base_kernel/lengthscales'] = value
            elif 'patch_weights' in path:
                layer_values['patch_weights'] = value
            layer_params[layer] = layer_values

        stored_layers = max(layer_params.keys()) + 1
        model_layers = len(Ms)
        assert stored_layers <= model_layers, "Can't load model if "
        if stored_layers != model_layers:
            last_layer = stored_layers - 1
            last_layer_params = layer_params[last_layer]
            del layer_params[last_layer]
            layer_params[model_layers-1] = last_layer_params

        return global_step, layer_params

    def _last_layer_parameters(self, layer_params):
        keys = list(layer_params.keys())
        if len(keys) > 0:
            return layer_params[max(keys)]
        else:
            return None

    # resnet block self, NHWC_X, M, feature_map, filter_size, stride, layer_params=None
    def _residual_block(self, H_X, M, feature_map, filter_size , stride , layer_params, name = 'unit'):
        # num_channel = H_X.get_shape().as_list()[-1]
        # with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % name)
            # Shortcut connection
        shortcut = H_X
        # print(H_X.shape)
            # pading to get the same input dimensionality 
   #      paddings = tf.constant([[0, 0],[1, 1,], [1, 1],[0, 0]])
			# # 'constant_values' is 0.
			# # rank of 't' is 2.
   #      H_X = tf.pad(H_X, paddings, "CONSTANT")
   #      # print('res after pad ', type(H_X))
   #      with tf.Session() as sess:
   #          H_X = sess.run(H_X)
   #      print('res after pad-eval ', type(H_X))  
            # Residual
        res_layers = []
        # print(H_X.shape)
        conv_layer, H_X = self._conv_layer(H_X, M, feature_map, filter_size, stride, 'SAME', layer_params) # 'conv_1'
        res_layers.append(conv_layer)
        # print('after conv layer ' ,H_X.shape)
        # H_X = self._bn(H_X, name='bn_1')

            #H_X = self._relu(H_X, name='relu_1')

            # pading to get the same input dimensionality 
        # H_X = tf.pad(H_X, paddings, "CONSTANT")
        # with tf.Session() as sess:
        #     H_X = sess.run(H_X)

        conv_layer, H_X = self._conv_layer(H_X, M, feature_map, filter_size, stride, 'SAME', layer_params) # 'conv_2'
        res_layers.append(conv_layer)

        # H_X = self._bn(H_X, name='bn_2')

        H_X = H_X + shortcut
            #H_X = self._relu(H_X, name='relu_2')
        return res_layers, H_X

    def _conv_layers(self, Ms, feature_maps, strides, filter_sizes, loaded_parameters={}):
        H_X = self.X_train
        layers = []
        shapes = []
        for i in range(len(feature_maps)):
            M = Ms[i]
            feature_map = feature_maps[i]
            filter_size = filter_sizes[i]
            stride = strides[i]
            layer_params = loaded_parameters.get(i)
            conv_layer, H_X = self._conv_layer(H_X, M, feature_map, filter_size, stride, 'VALID', layer_params)
            layers.append(conv_layer)
            shapes.append(H_X.shape)
        print(shapes)
        print(len(feature_maps))
        return layers, H_X
    # need to be geralizable 
    def _res_conv_layers(self, Ms, feature_maps, strides, filter_sizes, res_blocks = 1, loaded_parameters={}):
        H_X = self.X_train
        layers = []
        shapes = []
        for i in range(len(feature_maps)):
            M = Ms[i]
            feature_map = feature_maps[i]
            filter_size = filter_sizes[i]
            stride = strides[i]
            layer_params = loaded_parameters.get(i)
            if i % 2 == 0:
                conv_layer, H_X = self._conv_layer(H_X, M, feature_map, filter_size, stride, 'VALID', layer_params)
                shapes.append(H_X.shape)
                # print(conv_layer)
                layers.append(conv_layer)
            else:
                npad = ((0,0),(1,1),(1,1),(0,0))
                # H_X_pad = np.pad(H_X, pad_width=npad, mode='constant', constant_values=0)
                # pad_layer = lambda x: np.pad(x, pad_width=npad, mode='constant', constant_values=0)
                # layers.append(pad_layer)
                conv_layer, H_X = self._conv_layer(H_X, M, feature_map, 3, 1, 'SAME', layer_params)
                print('sam padding ', H_X.shape)
                shapes.append(H_X.shape)
                # print(conv_layer)
                layers.append(conv_layer)
                # H_X_pad = np.pad(H_X, pad_width=npad, mode='constant', constant_values=0)
                # pad_layer = lambda x: np.pad(x, pad_width=npad, mode='constant', constant_values=0)
                # layers.append(pad_layer)
                conv_layer, H_X = self._conv_layer(H_X, M, feature_map, 3, 1, 'SAME', layer_params)
                shapes.append(H_X.shape)
                # print(conv_layer)
                layers.append(conv_layer)
            # print(layers)
            # print('conv_1 ', type(H_X))
            # if i == 0:
            #     for j in range(res_blocks): #H_X, M, feature_map, filter_size , stride , layer_params, name = 'unit'
            #         print('Build residual block ', str(j+1))
            #         # print('shape befor residual ', H_X.shape)
            #         conv_layer, H_X = self._residual_block(H_X = H_X, M = M, feature_map = feature_map, filter_size = 3, stride = 1, layer_params = layer_params,  name = ('unit ' + str(j+1)))
            #         shapes.append(H_X.shape)
            #         # print(conv_layer)
            #         for x in conv_layer:
            #             layers.append(x)
            # print('shape after residual ',H_X.shape)
            # print(layers)
        print(shapes)
        print(len(feature_maps))
        return layers, H_X

    def _bn(self, x, name="bn"):
        x = utils_res._bn(x, self.is_train, name = name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils_res._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x