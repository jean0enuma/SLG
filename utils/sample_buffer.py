import torch
import random
import numpy as np
class SampleBuffer:
    def __init__(self, max_samples=10000):
        self.max_samples = max_samples
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples,input_length,targets,targets_length):
        samples = samples.detach().to('cpu')
        input_length=input_length.detach().to('cpu')
        class_ids = targets.detach().to('cpu')
        class_length=targets_length.detach().to('cpu')
        self.buffer.append((samples,input_length,class_ids,class_length))
        if len(self.buffer) > self.max_samples:
            self.buffer.pop(0)



    def get(self, device='cuda'):
        items = random.choices(self.buffer, k=1)
        sample, input_length,class_id,class_length = zip(*items)
        class_ids =class_id[0].to(device)
        input_length=input_length[0].to(device)
        samples = sample[0].to(device)
        class_length=class_length[0].to(device)

        return samples, input_length,class_ids,class_length



def sample_buffer(buffer,input_length,targets,targets_length, image_size=(2,100,1, 224, 224),n_class=1296, p=0.95, device='cuda'):
    batch_size=image_size[0]
    if len(buffer) < 1:
        return (
            torch.rand(batch_size, image_size[1], image_size[2], image_size[3],image_size[4], device=device),
            input_length.clone().detach(),targets,targets_length
        )
    is_sample=np.random.rand(1)<p
    if is_sample==True:
        replay_sample,replay_input_length, replay_id,replay_target_length = buffer.get()
        return (
            replay_sample,replay_input_length,replay_id,replay_target_length
        )
    else:
        random_sample = torch.rand(batch_size , image_size[1], image_size[2], image_size[3],image_size[4], device=device)
        return (
            random_sample,input_length,targets,targets_length
        )
