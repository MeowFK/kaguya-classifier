# if there's problems here, update fastai and torch to latest versions
# pip install 'name' == 'version'
import dotenv
from fastai.vision.all import *

# remember to set dls.device = device and learn.model.to(device)
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# load environment variables
dotenv.load_dotenv()

# use Azure search key
azure_search_key = os.getenv('AZURE_IMAGE_SEARCH_KEY')

if not azure_search_key:
    raise Exception("Set azure key in environment")
print(azure_search_key)
key = os.environ.get('AZURE_SEARCH_KEY', azure_search_key)


