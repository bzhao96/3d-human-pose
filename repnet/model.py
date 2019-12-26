import torch
import torch.nn as nn


def reprojection(x):
    pose3d = x[:, :48].view(-1, 16, 3)
    camera = x[:, 48:].view(-1, 3, 2)
    pose2d_rep = torch.matmul(pose3d, camera).view(-1, 32)
    return pose2d_rep


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

class Regression(nn.Module):
    def __init__(self, linear_size=1000):
        super(Regression, self).__init__()
        self.linear_size = linear_size
        self.input_size = 16 * 2
        self.output_pose3d = 16 * 3
        self.output_camera = 6
        self.leaky_relu = nn.LeakyReLU()

        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.linear_stage = Linear(linear_size)

        self.pose3d_stages1 = Linear(self.linear_size)
        self.pose3d_stages2 = Linear(self.linear_size)
        self.w21 = nn.Linear(self.linear_size, self.linear_size)
        self.w22 = nn.Linear(self.linear_size, self.output_pose3d)

        self.camera_stages1 = Linear(self.linear_size)
        self.camera_stages2 = Linear(self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.output_camera)

    def forward(self, x):
        y1 = self.w1(x)
        y1 = self.leaky_relu(y1)
        y1 = self.linear_stage(y1)

        y2 = self.pose3d_stages1(y1)
        y2 = self.pose3d_stages2(y2)
        y2 = self.w21(y2)
        y2 = self.leaky_relu(y2)
        y2 = self.w22(y2)

        y3 = self.camera_stages1(y1)
        y3 = self.camera_stages2(y3)
        y3 = self.w3(y3)

        out = torch.cat((y2, y3), 1)
        return out

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

if __name__ == '__main__':
    a = torch.randn(8, 52)
    b = reprojection(a)
    print(b)