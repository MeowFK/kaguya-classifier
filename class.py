# if there's problems here, update fastai and torch to latest versions
# pip install 'name' == 'version'
# fastbook needed for bing_image_search
import dotenv
from fastai.vision.all import *
from fastbook import *

# remember to set dls.device = device and learn.model.to(device)
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# load environment variables
dotenv.load_dotenv()

# use Azure search key
image_key = os.getenv('AZURE_IMAGE_SEARCH_KEY')

if not image_key:
    raise Exception("Set azure key in environment")

# search up images, probably needs a more specific search...
# kaguya_images = search_images_bing(image_key, 'Shinomiya Kaguya')
# ims = kaguya_images.attrgot('contentUrl')

# looking at an image
# dest = 'images/kaguya.jpg'
# download_url(ims[0], dest)
# im = Image.open(dest)
# im.to_thumb(128,128)

characters = ['shinomiya kaguya', 
              'miyuki shirogane', 
              'fujiwara chika', 
              'yu ishigami', 
              'miko iino', 
              'ai hayasaka', 
              'shirogane kei']
path = Path('kaguyasama images')

# make folders for characters
if not path.exists():
    path.mkdir()
    for o in characters:
        dest = path/o 
        dest.mkdir(exist_ok = True) # ok if exists, leave unaltered
        results = search_images_bing(image_key, f"{o}") # search images
        urls = results.attrgot('contentUrl') # get list of urls for images
        download_images(dest, urls=urls) # download images into characters folder

# get image files and unlink failed ones
images = get_image_files(path)
failed = verify_images(fns = images)
failed.map(Path.unlink)

# just checking the number of images which remain after the carnage
# for o in characters:
#     img_list = os.listdir(path/o)
#     print(f'{o}: {len(img_list)}')

# make DataBlock