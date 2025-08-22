import torch

x = torch.randn(2, 5).cuda()
z = torch.randn(10, 5).cuda()
exponent = 1.0
bandwidth = 1.5
alphas = torch.randn(10, 1).cuda()
kernel_fn = lambda x, z: torch.exp(-torch.cdist(x,z,p=exponent)**exponent / bandwidth**exponent)
fxn = lambda x: kernel_fn(x, z) @ alphas

print("fxn(x)", fxn(x))

# jacrev with vmap: compute per-sample gradients (2,5)
# f_single returns a scalar for a single sample; jacrev over it gives (5,)
def f_single(x_i, z, alphas):
    return (kernel_fn(x_i.unsqueeze(0), z) @ alphas).squeeze()

grads_vmapped = torch.func.vmap(
    torch.func.jacrev(f_single, argnums=0), in_dims=(0, None, None)
)(x, z, alphas).T
print("grads_vmapped shape", grads_vmapped.shape)
print("grads_vmapped", grads_vmapped)

import kermac

# if exponent == 2.0 and bandwidth == 1.0:
#     a_mat = -kernel_fn(z, x) / torch.cdist(z,x,p=exponent)
# elif exponent == 1.0 and bandwidth == 1.0:
#     a_mat = -kernel_fn(z, x)
# elif bandwidth == 1.0:
#     a_mat = -kernel_fn(z, x) / torch.cdist(z,x,p=exponent)**(exponent-1)
# else:
#     a_mat = -kernel_fn(z, x) / torch.cdist(z,x,p=exponent)**(exponent-1)
a_mat = -kernel_fn(z, x) * exponent / bandwidth**exponent

x = x.T.contiguous()
z = z.T.contiguous()
alphas = alphas.T.contiguous()

print("a_mat.shape", a_mat.shape)
print("z.shape", z.shape)
print("alphas.shape", alphas.shape)
print("x.shape", x.shape)

grads = kermac.cdist_grad(a_mat, z, alphas, x, p=exponent)

print("kermac grad", grads)
print("kermac grad shape", grads.shape)

# Compare the shapes:
# auto_grads_diag: (2, 5) - gradients for each sample with respect to its own input
# kermac grad: should also be (2, 5) for comparison
# print("auto_grads_diag shape:", auto_grads_diag.shape)
print("kermac grad shape:", grads.shape)
print("grads_vmapped shape:", grads_vmapped.shape)