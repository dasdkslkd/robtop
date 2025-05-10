import torch
import torch.nn as nn
from torch.utils import dlpack

# class Net(nn.Module):
#     def __init__(self, n_input,n_output,*args):
#         super(Net, self).__init__()
#         self.model = nn.Sequential()
#         self.model.add_module('input', nn.Linear(n_input, args[0]))
#         self.model.add_module('relu1', nn.ReLU())
#         for i in range(len(args)-1):
#             self.model.add_module('h'+str(i+1),nn.Linear(args[i],args[i+1]))
#             self.model.add_module('relu'+str(i+2), nn.ReLU())
#             # self.model.add_module('dropout'+str(i+2),nn.Dropout(0.3))
#         self.model.add_module('h'+str(len(args)), nn.Linear(args[-1], n_output))

#     def forward(self, x):
#         return self.model(x)
    
class Net(nn.Module):
    def __init__(self, n_input, n_output, *args, dropout_rate=0.3):
        super(Net, self).__init__()
        self.model = nn.Sequential()

        # 输入层
        self.model.add_module('input', nn.Linear(n_input, args[0]))
        # self.model.add_module('bn_input', nn.BatchNorm1d(args[0]))
        self.model.add_module('relu_input', nn.ReLU())
        # self.model.add_module('dropout_input', nn.Dropout(dropout_rate))

        # 隐藏层
        for i in range(len(args) - 1):
            self.model.add_module(f'hidden_{i}', nn.Linear(args[i], args[i + 1]))
            # self.model.add_module(f'bn_{i}', nn.BatchNorm1d(args[i + 1]))
            self.model.add_module(f'relu_{i}', nn.ReLU())
            # self.model.add_module(f'dropout_{i}', nn.Dropout(dropout_rate))

        # 输出层
        self.model.add_module('output', nn.Linear(args[-1], n_output))

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)
    
def jacobian(y: torch.Tensor, x: torch.Tensor, need_higher_grad=True) -> torch.Tensor:
    """基于 torch.autograd.grad 函数的更清晰明了的 API，功能是计算一个雅可比矩阵。

    Args:
        y (torch.Tensor): 函数输出向量
        x (torch.Tensor): 函数输入向量
        need_higher_grad (bool, optional): 是否需要计算高阶导数，如果确定不需要可以设置为 False 以节约资源. 默认为 True.

    Returns:
        torch.Tensor: 计算好的“雅可比矩阵”。注意！输出的“雅可比矩阵”形状为 y.shape + x.shape。例如：y 是 n 个元素的张量，y.shape = [n]；x 是 m 个元素的张量，x.shape = [m]，则输出的雅可比矩阵形状为 n x m，符合常见的数学定义。
        但是若 y 是 1 x n 的张量，y.shape = [1,n]；x 是 1 x m 的张量，x.shape = [1,m]，则输出的雅可比矩阵形状为1 x n x 1 x m，如果嫌弃多余的维度可以自行使用 torch.squeeze(Jac) 一步到位。
        这样设计是因为考虑到 y 是 n1 x n2 的张量； 是 m1 x m2 的张量（或者形状更复杂的张量）时，输出 n1 x n2 x m1 x m2 （或对应更复杂形状）更有直观含义，方便用户知道哪一个元素对应的是哪一个偏导。
    """
    (Jac,) = torch.autograd.grad(
        outputs=(y.flatten(),),
        inputs=(x,),
        grad_outputs=(torch.eye(torch.numel(y)).cuda(),),
        create_graph=need_higher_grad,
        allow_unused=True,
        is_grads_batched=True
    )
    if Jac is None:
        Jac = torch.zeros(size=(y.shape + x.shape))
    else:
        Jac.reshape(shape=(y.shape + x.shape))
    return Jac

def batched_jacobian(batched_y:torch.Tensor,batched_x:torch.Tensor,need_higher_grad = True) -> torch.Tensor:
    """计算一个批次的雅可比矩阵。
        注意输入的 batched_y 与 batched_x 应该满足一一对应的关系，否则即便正常输出，其数学意义也不明。

    Args:
        batched_y (torch.Tensor): N x y_shape
        batched_x (torch.Tensor): N x x_shape
        need_higher_grad (bool, optional):是否需要计算高阶导数. 默认为 True.

    Returns:
        torch.Tensor: 计算好的一个批次的雅可比矩阵张量，形状为  N x y_shape x x_shape
    """
    sumed_y = batched_y.sum(dim = 0) # y_shape
    J = jacobian(sumed_y,batched_x,need_higher_grad) # y_shape x N x x_shape

    # dims = list(range(J.dim()))
    # dims[0],dims[sumed_y.dim()] = dims[sumed_y.dim()],dims[0]
    # J = J.permute(dims = dims) # N x y_shape x x_shape
    # J=J.permute(1,2,0)
    # J=J.permute(2,1,0)
    
    # print(J.is_contiguous(),J.untyped_storage().data_ptr())
    # J=J.permute(0,2,1)
    # print(J.is_contiguous(),J.untyped_storage().data_ptr())
    return J

def predict(input,row:int,col:int) -> torch.Tensor:
    print("calling predict")
    fNN=torch.jit.load("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\fNN_cpu_64.pt")
    # fNN = Net(4,9,128,128,64,64,32,32)
    # fNN.load_state_dict(torch.load("D:\\Workspace\\tpo\\ai\\spinodal\\c++\\multitop\\fNN_aug_cpu_64.pth"))
    # fNN.double()
    fNN.eval()
    print("fNN loaded")
    x=torch.tensor(input,dtype=torch.float64)
    print(x.shape)
    print(row,col)
    x=x.reshape((row,col))
    # print(x)
    x[:,0]=(x[:,0]-0.3)/0.4
    x[:,1:]=x[:,1:]*2/torch.pi
    x.requires_grad=True
    print("x prepared")
    # print(x)
    y=fNN(x)
    J=batched_jacobian(y,x,False)
    # J[:,0,:]=J[:,0,:]/0.4
    # J[:,1:,:]=J[:,1:,:]*2/torch.pi
    J[0,:,:]=J[0,:,:]/0.4
    J[1:,:,:]=J[1:,:,:]*2/torch.pi
    # x.grad.zero_()
    # print(J)
    return y.detach().numpy().flatten(),J.detach().numpy().flatten()

Y_std=torch.tensor([0.15126401627803848227, 0.04597657597868053114, 0.04601151660781559183,0.15085248192644457044, 0.04598791325143272712, 0.15110439323726787553,0.04634756563962497827, 0.04636841337065750190, 0.04632001512579383973],dtype=torch.float32).to('cuda')
Y_mean=torch.tensor([0.17662967641632543181, 0.04973081306664266232, 0.04978806754443593913,0.17595569609413669321, 0.04969858252531161869, 0.17623666657966971516,0.05476377145749898590, 0.05488143547765828023, 0.05480586900733474404],dtype=torch.float32).to('cuda')
X_max=torch.tensor([0.7,1,1,1],dtype=torch.float32).to('cuda')
X_min=torch.tensor([0.1,0,0,0],dtype=torch.float32).to('cuda')

def inv_process(input:torch.Tensor) -> torch.Tensor:
    return input*Y_std+Y_mean

def inv_process_jacobian(input:torch.Tensor) -> torch.Tensor:
    return input*(Y_std.view(9,1,1)/(X_max-X_min).view(1,1,4)) # 9x1x4

def predict_dlpack(in_dlpack,out_pred,out_jacobian):
    print("calling predict_dlpack")
    fNN = Net(4,9,128,128,64,64,32,32)
    # file_path="./fNN_paper2_cpu_64_2.pth"
    file_path="best_model.pth"
    fNN.load_state_dict(torch.load(file_path))
    # fNN=torch.jit.load("./fNN_cpu_64.pt")
    fNN.float()
    fNN.cuda()
    fNN.eval()
    
    print("fNN loaded")
    x = dlpack.from_dlpack(in_dlpack)
    out_y = dlpack.from_dlpack(out_pred)
    out_J = dlpack.from_dlpack(out_jacobian)
    # out_y=out_y.reshape((9,2))
    # out_J=out_J.reshape((9,4,2))
    # print(out_y)
    # print(x.is_contiguous(),x.untyped_storage().data_ptr())
    # print(x.device)
    # print(x.dtype)
    # print(x.data_ptr())
    x=x.T
    # print(x)
    # print(x.is_contiguous(),x.untyped_storage().data_ptr())
    # x = x.to(torch.float64)
    # print(x)
    # print(x.is_contiguous(),x.untyped_storage().data_ptr())
    print(x[0,:])
    
    x[:,0]=(x[:,0]-0.1)/0.6
    # x[:,1:]=x[:,1:]*2/torch.pi
    x.requires_grad=True
    # print(x.shape,x.is_contiguous(),x.untyped_storage().data_ptr())
    # y:torch.Tensor=fNN(x)
    y=inv_process(fNN(x))
    print(y[0,:])
    # print(y)
    # print(y.shape)
    J=batched_jacobian(y,x,False)
    # J[:,:,0]=J[:,:,0]/0.4
    # J[:,:,1:]=J[:,:,1:]*2/torch.pi
    J=inv_process_jacobian(J)
    J=J.permute(0,2,1)
    y*=1e5
    J*=1e5
    out_y.copy_(y.T)
    # print(out_y)
    # print(out_y.is_contiguous(),out_y.untyped_storage().data_ptr())
    out_J.copy_(J)
    print('fNN finished')
    
    # print(J.is_contiguous(),J.untyped_storage().data_ptr())
    # print(J.shape)
    # print(J.detach().flatten())



if __name__ == '__main__':
    # x1=torch.tensor([[0.4,torch.pi/12,0,torch.pi/12]])
    x1=torch.tensor([[0.4137, 0.0, 0.2972, 0.0]])
    # x1[:,0]=(x1[:,0]-0.3)/0.4
    x1[:,1:]=x1[:,1:]*2/torch.pi
    x2=x1.clone()
    x2[0,0]+=1e-3
    x1.requires_grad=True
    fNN = Net(4,9,128,128,64,64,32,32)
    file_path="/home/xyz/robtop/bench/fNN_paper2_cpu_64_2.pth"
    fNN.load_state_dict(torch.load(file_path,weights_only=True))
    # fNN=torch.jit.load("./fNN_cpu_64.pt")
    fNN.float()
    fNN.cuda()
    fNN.eval()
    x1=x1.to('cuda')
    x2=x2.to('cuda')
    y1:torch.Tensor=fNN(x1)
    y2:torch.Tensor=fNN(x2)
    # y2:torch.Tensor=fNN(x2)
    # print(y1.untyped_storage().data_ptr())
    J1=batched_jacobian(y1,x1,False)
    # J1[0,:,:]=J1[0,:,:]/0.4
    J1[:,:,1:]=J1[:,:,1:]*2/torch.pi
    print(y1)
    print(J1)
    print((y2-y1)/1e-3)
    print(J1.permute(0,2,1).flatten())