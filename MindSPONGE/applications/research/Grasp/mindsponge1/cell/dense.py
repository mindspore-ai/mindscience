import mindspore as ms
import numpy as np
from mindspore import ops, Tensor, Parameter, nn
from mindspore.ops import operations as P

class NetBatch(nn.Cell):
    
    def __init__(self, batch_size=None):
        super().__init__()
        self.batch_size = batch_size
    
    def _new_shape(self, shape):
        if self.batch_size is not None:
            shape = [self.batch_size,]+list(shape)
        return shape
    
    def _get_params(self, index):
        if index is not None:
            ls = []
            for p in self._params.values():
                ls.append(p[index])
            return ls
        else:
            return self._params.values()


class Dense(NetBatch):
    # no activation, zero bias init, lecun weight init
    def __init__(self, input_dim, output_dim, batch_size=None, is_gate=False):
        super().__init__(batch_size)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_gate = is_gate
        self.matmul = P.MatMul()
        self._init_parameter()

    def construct(self, x, index=None):
        w, b = self._get_params(index)
        # y = ops.matmul(x, w) + b

        y = P.Reshape()(x, (-1, x.shape[-1]))
        y = self.matmul(y, w) + b
        y = P.Reshape()(y, x.shape[:-1]+(-1,))
        return y
    
    def _lecun_normal(self, dim_in, shape):
        stddev = 1./np.sqrt(dim_in)
        return np.random.normal(loc=0, scale=stddev, size=shape)
 
    def _init_parameter(self):
        w_shape = self._new_shape((self.input_dim, self.output_dim))
        b_shape = self._new_shape((self.output_dim,))
        if self.is_gate:
            self.weight = Parameter(Tensor(np.zeros(w_shape), ms.float32))
            self.bias = Parameter(Tensor(np.ones(b_shape), ms.float32))
        else:
            # self.weight = Parameter(Tensor(self._lecun_normal(self.input_dim, w_shape), ms.float32))
            self.weight = Parameter(Tensor(np.zeros(w_shape), ms.float32))
            self.bias = Parameter(Tensor(np.zeros(b_shape), ms.float32))

        
class ProcessSBR(nn.Cell):
    
    def __init__(self, sbr_act_dim, output_dim, batch_size=None, gate=False, pair_input_dim=0):
        super().__init__()
        self.sbr_act_dim = sbr_act_dim
        self.atte_dim = output_dim
        self.linear1 = Dense(sbr_act_dim, output_dim, batch_size)
        if gate:
            self.linear2 = Dense(sbr_act_dim+pair_input_dim, output_dim, batch_size, is_gate=True)
            self.sigmoid = nn.Sigmoid()
        
    def construct(self, sbr_act, sbr_mask, pair=None, index=None):
        y = self.linear1(sbr_act, index)
        if pair is not None:
            sbr_act = ops.Tile()(sbr_act, pair.shape[:-3]+(1, 1, 1))
            gate = ops.Concat(-1)((sbr_act, pair))
            gate = self.sigmoid(self.linear2(gate, index))
            y *= gate
        y *= sbr_mask[..., None]
        return y

class AddInterface(nn.Cell):
    
    def __init__(self, input_dim, batch_size=None):
        super().__init__()
        self.linear = Dense(input_dim+1, input_dim, batch_size)
        
    def construct(self, interface_mask, act, index=None):
        mask = interface_mask[..., None]
        mask = ops.Tile()(mask, act.shape[:-2]+(1, 1))
        x = ops.Concat(-1)((act, mask))
        y = self.linear(x, index)
        y *= mask
        return y
        
 
        
# ds = Dense(3, 5, 2)
# x = Tensor(np.arange(24).reshape((2,4,3)), ms.float32)
# y = ds(x, 1)
# y.shape

# sbr_act = Tensor(np.random.normal(size=(4,4,3)), ms.float32)
# atte = Tensor(np.random.normal(size=(4,4,2)), ms.float32)
# sbr_mask = Tensor(np.random.rand(4,4)<0.5, ms.float32)
# print(sbr_act, atte, sbr_mask)
# psbr = ProcessSBR(3, 2, batch_size=5)
# y = psbr(sbr_act, atte, sbr_mask, index=3)
# print(y, y.shape)

# single_act = Tensor(np.random.normal(size=(4, 2)), ms.float32)
# msa_act = Tensor(np.random.normal(size=(3, 4, 2)), ms.float32)
# interface_mask = Tensor(np.random.rand(4)<0.5, ms.float32)
# print(single_act, msa_act, interface_mask, sep='\n')
# aif = AddInterface(2, batch_size=5)
# y_single = aif(interface_mask, single_act, index=3)
# y_msa = aif(interface_mask, msa_act, index=3)
# print('single', y_single.shape, y_single)
# print('msa', y_msa.shape, y_msa)