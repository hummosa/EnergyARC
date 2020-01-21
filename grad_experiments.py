#%%
import torch
import numpy as np
from torch.nn import functional as F
#%%
yh = torch.tensor(1.)
x = torch.randn(1)
par =  torch.randn(1)
par.requires_grad_()

x = torch.tensor(.6622)
par = torch.tensor(0.28747)
par.requires_grad_()

print('x: ', x.item())
for i in range(30):
    y = x *par
    loss = torch.abs(y - yh)
    loss.sum().backward()
    print( 'loss: {:.4f} \t grad: {:.4f} \t y: {:.4f} \t par: {:.4f}'.format (loss.sum().detach().numpy(), par.grad.item(), y.item(), par.item()) )
    par.data.add_(-.3 *loss.sum().item(), par.grad.item())
    par.grad.zero_()
#%%
def mp(a):
    with np.printoptions(precision=3, suppress=True): #suppress suppresses use of sicentific notation for small numbers. 
        if isinstance(a, torch.FloatTensor):
            l = a.detach().cpu().numpy()
        elif isinstance(a, np.ndarray):
            pass
        print(l)
#%%
# Now. Let's try some thing recursive. Find the gradient of a function, run opt, then take the graident of that. 
# 
# 1) learn a pattern in the world. That x=.6 and y=1. occur together. by setting parameter to 1.6667
# OR Do it EBM style. x and y are input, and out comes energy. Then NLL
# 2) but let's keep this up. Now x = 0.2, can you retieve it?  
yh = torch.tensor(1.)

x = torch.tensor(.6)
par = torch.tensor(0.2)
par.requires_grad_()

print('x: ', x.item())
for i in range(30):
    y = x *par
    loss = torch.abs(y - yh)
    loss.sum().backward()
    print( 'loss: {:.4f} \t grad: {:.4f} \t y: {:.4f} \t par: {:.4f}'.format (loss.sum().detach().numpy(), par.grad.item(), y.item(), par.item()) )
    par.data.add_(-.3 *loss.sum().item(), par.grad.item())
    # par.grad.zero_() # Turning this on improves convergence. It preserves momentum. Despite some oscillations later.

x = torch.tensor(.20)
x.requires_grad_()
par.requires_grad_(False)

print('x: ', x.item())
for i in range(30):
    y = x *par
    loss = torch.abs(y - yh)
    use_grad = True
    if use_grad:
        gx = torch.autograd.grad(loss.sum(), x)
        x.data.add_(-.3 *loss.sum().item(), gx[0].item()) # oh man the abs operator is terrrible for gradients. It is either +1 or -1
        # multiplying byt the loss value is critical for super fast convergence. Otherwise it just oscillates forever around 0 loss.
    else:
        loss.sum().backward()
        x.data.add_(-.3 , x.grad.item())
    print( 'loss: {:.4e} \t x grad: {:.4f} \t y: {:.4f} \t x: {:.4f}'.format (loss.sum().detach().numpy(), x.grad.item() if not use_grad else gx[0].item(), y.item(), x.item()) )
    if not use_grad: x.grad.zero_() 
        
# %% This computes the second gradient.
x = torch.tensor(2.)
x.requires_grad_()
y = x**2
g = torch.autograd.grad(y, x, create_graph=True, retain_graph=False)
g[0].requires_grad #true if you specify create_graph=True. Creating graph refers to treating grad as any ops in the graph tree that 
# that requires gradients to flow through it. Which is the behavior I want. But it blows up the memory very quickly :(
print('first grad: ', g)

g2 = torch.autograd.grad(g[0], x)

print('second grad: ', g2)

#%% now let's take several gradient steps w.r.t inputs, and then do one step of back grad w.r.t to parameters.
yh = torch.tensor(1.)

x = torch.tensor(.6)
par = torch.tensor(0.2)
par.requires_grad_()

print('x: ', x.item())
for i in range(30):
    y = x *par
    loss = torch.abs(y - yh)
    loss.sum().backward()
    # print( 'loss: {:.4f} \t grad: {:.4f} \t y: {:.4f} \t par: {:.4f}'.format (loss.sum().detach().numpy(), par.grad.item(), y.item(), par.item()) )
    par.data.add_(-.3 *loss.sum().item(), par.grad.item())
    # par.grad.zero_() # Turning this on improves convergence. It preserves momentum. Despite some oscillations later.

#so now we learned the par at 1.667. But at this value it does not allow x to converge fast enough to its value. 
# But since it is one parameter value, both goals will have to share and it will be a conflict. Will be interesting to see two parameters
x = torch.tensor(.20)
x.requires_grad_()
par.requires_grad_(True) 

print('x: ', x.item())
for i in range(20):
    y = x *par
    loss = torch.abs(y - yh)
    use_grad = True
    if use_grad:
        gx = torch.autograd.grad(loss.sum(), x, create_graph=True, retain_graph=True)
        # x.data.add_(-.01, gx[0])   #This works but breaks the graph
        # x.add_(-.01, gx[0])     #This complains in-place ops 
        x = x + (-.01 * gx[0])  #This works! 
        
        # oh man the abs operator is terrrible for gradients. It is either +1 or -1
        # multiplying by the loss value is critical for super fast convergence. Otherwise it just oscillates forever around 0 loss.
    else:
        loss.sum().backward()
        x.data.add_(-.01 , x.grad)
    print( 'loss: {:.4f} \t x grad: {:.4f} \t y: {:.4f} \t x: {:.4f}'.format (loss.sum().detach().numpy(), x.grad.item() if not use_grad else gx[0].item(), y.item(), x.item()) )
    if not use_grad: x.grad.zero_() 

ggx = torch.autograd.grad(gx, par, retain_graph=True ) #This works.
print('gradient of grad_x w.r.t par  :', ggx[0])
gxg = torch.autograd.grad(x, gx , retain_graph=True)  # this says they are not connected.
print('gradient of    x   w.r.t grad_x:', gxg[0])
gpar = torch.autograd.grad(x, par, retain_graph=True )  # this says they are not connected.
print('gradient of    x   w.r.t par  :', gpar[0])


# %%
par = par + gpar[0]

x = torch.tensor(.20)
x.requires_grad_()
par.requires_grad_(True) 

print('x: ', x.item())
for i in range(20):
    y = x *par
    loss = torch.abs(y - yh)

    if use_grad:
        gx = torch.autograd.grad(loss.sum(), x, create_graph=True, retain_graph=True)
        x = x + (-.01 * gx[0]) 

    print( 'loss: {:.4f} \t x grad: {:.4f} \t y: {:.4f} \t x: {:.4f}'.format (loss.sum().detach().numpy(), gx[0].item(), y.item(), x.item()) )
