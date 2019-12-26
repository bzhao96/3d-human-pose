import torch
import torch.nn as nn

def reprojection(pose3d, camera):
    camera_params = camera.unsqueeze(1)
    f = camera_params[..., :2]
    c = camera_params[..., 2:]
    X = pose3d[..., :2]
    Z = pose3d[..., 2:]
    # X = X / Z
    # X = torch.clamp(X / Z, min=-1, max=1)
    return f*X + c*Z

class Linear(nn.Module):
    def __init__(self, linear_size):
        super(Linear, self).__init__()
        self.linear_size = linear_size
        self.leaky_relu = nn.LeakyReLU()
        self.w1 = nn.Linear(self.linear_size, self.linear_size)
        self.w2 = nn.Linear(self.linear_size, self.linear_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.leaky_relu(y)
        y = self.w2(y)
        out = x + y
        out = self.leaky_relu(out)
        return out

### frames 27
class Regression(nn.Module):
    def __init__(self, pad, input_size, output_size, linear_size=1000):
        super(Regression, self).__init__()
        self.pad = pad
        self.linear_size = linear_size
        self.input_size = input_size
        self.output_size = output_size

        self.leaky_relu = nn.LeakyReLU()
        self.w1 = nn.Conv1d(self.input_size, self.linear_size, 3, stride=3)
        self.w21 = nn.Conv1d(self.linear_size, self.linear_size, 3, stride=3)
        self.w22 = nn.Conv1d(self.linear_size, self.linear_size, 1)
        self.w31 = nn.Conv1d(self.linear_size, self.linear_size, 3)
        self.w32 = nn.Conv1d(self.linear_size, self.linear_size, 1)        
        self.w4 = nn.Conv1d(self.linear_size, self.output_size, 1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        
        y1 = self.w1(x)
        y1 = self.leaky_relu(y1)

        y1_slice = y1[:, :, range(1, y1.shape[2], 3)]
        y2 = self.w21(y1)
        y2 = self.leaky_relu(y2)
        y2 = self.w22(y2)
        y2 = y2 + y1_slice
        y2 = self.leaky_relu(y2)

        y2_slice = y2[:, :, 1:2]
        y3 = self.w31(y2)
        y3 = self.leaky_relu(y3)
        y3 = self.w32(y3)
        y3 = y3 + y2_slice
        y3 = self.leaky_relu(y3)

        y4 = self.w4(y3)
        
        return y4.squeeze()


class Critic(nn.Module):
    def __init__(self, linear_size=100):
        super(Critic, self).__init__()
        self.linear_size = linear_size

        self.input_size = 16 * 3
        self.leaky_relu = nn.LeakyReLU()
        self.w11 = nn.Linear(self.input_size, self.linear_size)
        self.linear_stage1 = Linear(self.linear_size)
        self.w12 = nn.Linear(self.linear_size, self.linear_size)

        self.w2 = nn.Linear(15*15, 1000)
        self.linear_stage2 = Linear(1000)
        
        self.w3 = nn.Linear(self.linear_size + 1000, self.linear_size)
        self.w4 = nn.Linear(self.linear_size, 1)

    def forward(self, x):
        y1 = self.w11(x)
        y1 = self.leaky_relu(y1)
        y1 = self.linear_stage1(y1)
        y1 = self.w12(y1)

        psi = self.kcs(x)
        psi_vector = torch.flatten(psi, start_dim=1)
        y2 = self.w2(psi_vector)
        y2 = self.leaky_relu(y2)
        y2 = self.linear_stage2(y2)

        output = torch.cat((y1, y2), 1)
        output = self.w3(output)
        output = self.leaky_relu(output)
        output = self.w4(output)
        return output

    def kcs(self, x):
        # KCS matrix
        Ct = torch.tensor([
            [1., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0 , 0, 0, 0, 1, 0],
            [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0 , 0, 0, 0, 0,-1],
            [0, 0, 0, 0, -1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,-1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,-1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,-1, 0, 0]])
        Ct = Ct.transpose(0, 1)
        C = Ct.repeat(x.shape[0], 1, 1)
        x = x.view(-1, 16, 3)
        B = torch.matmul(C.to(x.device), x)
        psi = torch.matmul(B, B.transpose(1, 2))
        return psi


class Temporal_Critic(nn.Module):
    def __init__(self, pad, linear_size=1000):
        super(Temporal_Critic, self).__init__()
        self.pad = pad
        self.input_size = 16 * 3 * (self.pad * 2)

        self.linear_size = linear_size
        self.leaky_relu = nn.LeakyReLU()
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.linear_stage1 = Linear(self.linear_size)
        self.linear_stage2 = Linear(self.linear_size)
        self.linear_stage3 = Linear(self.linear_size)
        self.w2 = nn.Linear(self.linear_size, 100)
        self.w3 = nn.Linear(100, 1)

    def forward(self, x):
        if len(x.shape) != 3:
            x = x.view(-1, 2*self.pad+1, 48)
        
        x1 = x[:,:self.pad,:] - x[:,self.pad:self.pad+1,:]
        # x2 = x[:, self.pad:self.pad+1,:]
        x3 = x[:,self.pad+1:,:] - x[:,self.pad:self.pad+1,:]
        x_in = torch.flatten(torch.cat((x1, x3), 1), start_dim=1)
        y1 = self.w1(x_in)
        y1 = self.leaky_relu(y1)
        y1 = self.linear_stage1(y1)
        y1 = self.linear_stage2(y1)
        y1 = self.linear_stage3(y1)
        y1 = self.w2(y1)
        y1 = self.leaky_relu(y1)
        y1 = self.w3(y1)
        return y1


if __name__ == '__main__':
    a = torch.randn(8, 52)
    b = reprojection(a)
    print(b)