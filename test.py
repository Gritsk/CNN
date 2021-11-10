from _typeshed import Self
import torch
from torch._C import dtype
# #empty tensor with 1D with 3 elements
# x=torch.empty(3)

# #empty tensor with 2D with 2 elements
# y=torch.empty(2,2)
# z = torch.ones(2,2,dtype=int)
# print(x.dtype)
# print(y)
# print(z)
# w = z+y
# print(w)
# ## w=torch.add(y,z)
# #y.add_(z) # does inplace operation
# tens = torch.tensor([2,4])
# print(tens)


# ## Gradients we want to calculate gradient with respect to randTens, so we adnn requires grad
# randTens = torch.randn(3, requires_grad=True)
# print(randTens)

# graph =randTens+2
# print(graph)
# uu= graph*graph*2
# uu = uu.mean()
# print(uu)
# # when we want to calculate gradient determinant, .backward
# uu.backward()
# print(randTens.grad)


### Backward Propagation:

xx = torch.tensor(1.0)
yy = torch.tensor(2.0)
ww = torch.tensor(1.0, requires_grad=True)

# forward pass:
y1= xx*ww
loss= (y1 -yy)**2
print (loss)

## backward pass
loss.backward()
print(ww.grad)

x2=Self.co