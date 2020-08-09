#! /home/cky/anaconda3/bin/python
import torch as torch
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
#x = torch.randn(N, D_in).cuda()  .to("cuda:0")     yong gpu yunsuan 
y = torch.randn(N, D_out)
w1 = torch.randn(D_in, H)
w2 = torch.randn(H,D_out)
learning_rate = 1e-6  
# tidu xiajiang buchang setting
for t in range(1000):
    # forward pass
    h = x.mm(w1)#N*H
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)#N*D_out
    #computloss
    loss = (y_pred-y).pow(2).sum().item()
    print(t, loss)
    
    #backword loss
    #compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h<0] = 0
    grad_w1 = x.t().mm(grad_h)

    #update weights of w1 and w2
    w1 -=learning_rate * grad_w1
    w2 -=learning_rate * grad_w2
print(y_pred-y)

    
