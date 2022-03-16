from src.utils import args
from src.image_generator import ImageGenerator
from src.image_processing import ImageStacker, ImageAugmentations
from src.preprocessing import split_cc_dataset, get_split_sol_dataset
from src.models import *
from tqdm import tqdm
import os
import glob
import re
import pandas as pd

def empty_file(path):
    files = glob.glob(path + '/*')
    for file in files:
        os.remove(file)
    return None

def get_aug_df(dataset):
    print('GETTING AUG DF')
    image_dir = f'./data/{dataset}/aug_images'
    paths = [f'{image_dir}/{x}' for x in tqdm(os.listdir(image_dir))]
    labels = [re.findall(r'.*__(.*).png', y)[0] for y in tqdm(paths)]
    model_df = pd.DataFrame({'fname': paths,
                             'label': labels})
    model_df['is_valid'] = 0
    return model_df

if __name__ == '__main__':
    if args.input == 'image':
        # Generate the training images - uses all cpu cores
        img_gen = ImageGenerator(args.dataset)

        # Generate dataframes of [paths, label, is_valid]
        if args.dataset == 'cocrystal':
            img_stacker = ImageStacker()
            raw_df = img_stacker.fastai_data_table()
            raw_df = split_cc_dataset(raw_df, 0.2)
        elif args.dataset == 'solubility':
            raw_df = get_split_sol_dataset(val_pct=0.2, test_pct=0.25)

        # Apply Image Augmentations to the training sets only!
        # Generate the dataframe for the model with new aug_paths and labels
        if not args.no_augs:
            augmentor = ImageAugmentations(args.dataset)
            augmentor.do_image_augmentations(raw_df)
            aug_df = get_aug_df(args.dataset)
            val_df = raw_df[raw_df['is_valid'] == 1]
            model_df = pd.concat([aug_df, val_df], axis=0)
        else:
            model_df = raw_df[raw_df['is_valid'] != 'test']

        # Apply FastAI model
        if args.dataset == 'cocrystal':
            trainer = train_fastai_model_classification(model_df)
        elif args.dataset == 'solubility':
            trainer = train_fastai_model_regression(model_df)


            ### test sets!!!!!

            model = load_learner(f'./checkpoints/{args.dataset}/trained_model.pkl', cpu=True)
            # get best val acc
            print(f'best_metrics = {model.final_record}')

        #     true = []
        #     preds = []
        #     for idx, path in enumerate(test_df['fname']):
        #         img = torch.tensor(cv2.imread(path)).cpu()
        #         true.append(test_df.label.values[idx])
        #         preds.append(model.predict(img)[0].lower())
        #     acc = accuracy_score(true, preds)
        #     final_acc.append(acc)
        #     split_idx = split_idx + 1
        #     torch.cuda.empty_cache()
        # print(f'Val Acc Values: {val_acc}')
        # print(f'Test Acc Values: {final_acc}')
        # print(f'Mean Val Acc = {np.mean(val_acc)}')
        # print(f'Mean Test Acc = {np.mean(final_acc)}')
        print('done')






        empty_file(f'./data/{args.dataset}/aug_images')
    print('done')

        #### Come back to apply augmentations
        ## you have 2 x fastai dataframes.
        # apple augs to the train only
        # still need to implement all descriptor baselines

