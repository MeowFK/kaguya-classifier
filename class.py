# if there's problems here, update fastai and torch to latest versions
# pip install 'name' == 'version'
from fastai.vision.all import *

# remember to set dls.device = device and learn.model.to(device)
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# use 
