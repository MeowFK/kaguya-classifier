# if there's problems here, update fastai and torch to latest versions
# pip install 'name' == 'version'
# fastbook needed for bing_image_search
# widgets needed for cleaner
import dotenv
from fastai.vision.all import *
from fastbook import *
from fastai.vision.widgets import *

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
# input: images, output: categories (name)
# retrieve input images by calling get_image_files
# do random split with 20% as validation data, make seed random later
# get answer by the name of the folder the image is in (parent_label)
chars = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items = get_image_files,
    splitter = RandomSplitter(valid_pct = 0.2, seed = 42),
    get_y = parent_label,
    item_tfms = RandomResizedCrop(224, min_scale = 0.5),
    batch_tfms = aug_transforms()
)

# make DataLoader from DataBlock, pass path data into the dataloader
dls = chars.dataloaders(path)

# learn with resnet18
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(0)

# clean data
cleaner = ImageClassifierCleaner(learn)