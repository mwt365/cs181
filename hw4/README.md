# HW4
There is one task here, to extend the code from hw3 into a 20-layer resnet

To do this, I updated code [here](hw4.py). It came from the file `pytorch1.py` in the hw3 repository.

## Method
What I ended up doing was initializing a variable to hold the outputs from the NN, and then holding it until called. When it was time, I summed the held outputs (in the resnet), and then updated the resnet to be the new outputs.

```python
inside of the training loop

#first run, store the resnet
if i == 0:
    resnetStore = outputs
#if it's divisible by 20, add in the stored resnet
elif (i%20) == 0:
    outputs = (outputs + resnetStore)
    resnetStore = outputs
```

This achieved a better training loss than previously. I experimented with storing the resnet as the outputs before and after the previous resent had been added in, and did not find significant difference. 