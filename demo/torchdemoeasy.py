#! /home/cky/anaconda3/bin/python
import torch as torch
from torch import nn
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
#x = torch.randn(N, D_in).cuda()  .to("cuda:0")     yongyu 
y = torch.randn(N, D_out)
# w1 = torch.randn(D_in, H)
# w2 = torch.randn(H,D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H, bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out, bias=False),
)
#model init: normal model weight
torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)

loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-6  
# tidu xiajiang buchang setting

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for t in range(1000):
    # # forward pass
    # h = x.mm(w1)#N*H
    # h_relu = h.clamp(min=0)
    # y_pred = h_relu.mm(w2)#N*D_out
    y_pred = model(x)
    # #computloss
    # loss = (y_pred-y).pow(2).sum().item()
    # print(t, loss)
    # loss = loss_fn(y_pred, y)
    loss = loss_fn(y_pred, y)
    print (t, loss.item())


    # #backword loss
    loss.backward()

    optimizer.step()
    
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    
    model.zero_grad()
    # #compute the gradient
    # grad_y_pred = 2.0 * (y_pred - y)
    # grad_w2 = h_relu.t().mm(grad_y_pred)
    # grad_h_relu = grad_y_pred.mm(w2.t())
    # grad_h = grad_h_relu.clone()
    # grad_h[h<0] = 0
    # grad_w1 = x.t().mm(grad_h)
    
    # #update weights of w1 and w2
    # w1 -=learning_rate * grad_w1
    # w2 -=learning_rate * grad_w2
# print(y_pred-y)

    
