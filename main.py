from src.utils import args
from src.image_generator import ImageGenerator
from src.image_processing import ImageStacker, ImageAugmentations
from src.preprocessing import split_cc_dataset, get_split_sol_dataset
import tqdm
import os
import glob

def empty_file(path):
    files = glob.glob(path + '/*')
    for file in files:
        os.remove(file)

if __name__ == '__main__':
    if args.input == 'image':
        img_gen = ImageGenerator(args.dataset)

        if args.dataset == 'cocrystal':
            img_stacker = ImageStacker()
            model_df = img_stacker.fastai_data_table()
            model_df = split_cc_dataset(model_df, 0.2)


        if args.dataset == 'solubility':
            df = get_split_sol_dataset(val_pct=0.2, test_pct=0.25)
            print(df.head(100))
            print('done')

        #### Come back to apply augmentations
        ## you have 2 x fastai dataframes.
        # apple augs to the train only
        # still need to implement all descriptor baselines

