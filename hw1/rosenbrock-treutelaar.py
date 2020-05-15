'''
Max Treutelaar

The rosenbrock test function is a common "banana-shaped" function to test how well optimization routines work.
See: https://en.wikipedia.org/wiki/Rosenbrock_function

'''
import torch

def rosenbrock(x,y):
    a = 1 
    b = 10
    return (a-x)**2 + b*(y-x**2)**2

def rosenbrock_mod(x):
a = 1 
b = 10 
return (a-x[0])**2 + b*(x[1]-x[0]**2)**2

# add your code here
x = torch.tensor(
    0.0,
    requires_grad=True
    )
y = torch.tensor(
    1.0,
    requires_grad=True
    )

# Calculate the derivative
z = rosenbrock(x,y)
z.backward()
print("z = ", z)


print('x.grad=',x.grad)
print('y.grad=',y.grad)

alpha = 0.01 #learning rate

x = torch.tensor(7.0,requires_grad=True)
y = torch.tensor(7.0,requires_grad=True)

for i in range(50):
    print('i=',i,'x=',x, "/n y=" , y)
    z = rosenbrock(x,y)
    z.backward()
    x = x - alpha * x.grad
    x = torch.tensor(x,requires_grad=True)
    for j in range(50):
        print('i=',i,'x=',x, "/n y=" , y)
        y = y - alpha * y.grad
        y = torch.tensor(y,requires_grad=True)
