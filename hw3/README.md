# CS181 HW3 - Max Treutelaar
## Structure
There are three files here to complete the three tasks in the assignment.
## Task 1
The first task was to complete part 1 of the given tutorial. It can be found [here in pytorch1.py](pytorch1.py)

The second task was to complete part 2 of the given tutorial. It can be found [here in pytorch2.py](pytorch2.py)

## Task 2

Modyfing the code to work with Fashion-MNIST. This can be found [here in fashionMNIST_working.py](FashionMNIST_working.py)


To complete this task, I used the nn from task 1- part 1, as it was set up to handle 1 dimesnional images. Then, I tried a lot of ways to tranform the image or the nn before I came across changing the tranform statement to:
```python
transform = transforms.Compose(
    [
    transforms.Resize((32,32)),
    transforms.ToTensor(),
     ])
```

The key thing here is both transforming the PIL image, and also doing it *before* converting to tensor is quite key.

## Task 3

This task was to include torchsample in the augmentation in the network. My code can be found [here](included_torchsample.py)

**Note:** at the time that I did this HW, the repo linked from the assignment was dead. I instead used (https://github.com/ncullen93/torchsample)


Basically all I did was add the torchsample library, and then when creating the trainer, I called a torchsample instance instead of the existing library.