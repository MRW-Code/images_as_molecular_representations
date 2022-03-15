import numpy as np
import cv2
import os
import tqdm
import pandas as pd
from joblib import Parallel, delayed
import os
import tqdm

class ImageAugmentations:

    def __init__(self, dataset):
        self.dataset = dataset
        self.img = None
        self.rotated_images = None
        self.flipped_images = None
        self.cropped_images = None
        self.augmented_images = None

    def get_image(self, img_path):
        self.img = cv2.imread(img_path)
        return self.img

    def rotating(self, num_rotations):
        (h, w) = self.img.shape[:2]
        center = (w / 2, h / 2)
        self.rotated_images = []
        for i in range(num_rotations):
            rotations = list(range(0, 180, num_rotations))
            M = cv2.getRotationMatrix2D(center, rotations[i], 1.0)
            rotated = cv2.warpAffine(self.img, M, (w, h))
            self.rotated_images.append(rotated)
        return self.rotated_images

    def flipping(self, img):
        self.flipped_images = []
        originalImage = img
        flipVertical = cv2.flip(originalImage, 0)
        flipHorizontal = cv2.flip(originalImage, 1)
        flipBoth = cv2.flip(originalImage, -1)
        self.flipped_images.append(flipVertical)
        self.flipped_images.append(flipHorizontal)
        self.flipped_images.append(flipBoth)
        return self.flipped_images

    def do_image_augmentations(self):
        save_path = f'./data/{self.dataset}/aug_images/'
        if self.dataset == 'solubility':
            parent_path = f'./data/{self.dataset}/images/'
            image_paths = os.listdir(parent_path)
        elif self.dataset == 'cocrystal':
            parent_path = f'./data/{self.dataset}/concat_images/'
            image_paths = os.listdir(parent_path)
        else:
            AttributeError('No such Dataset Exists')
        full_paths = []
        def worker(image):
            test_image_path = parent_path + image
            count = 0
            img = self.get_image(test_image_path)
            rot = self.rotating(6)
            for im in rot:
                flip = self.flipping(im)
                for flipped in flip:
                    file_name = save_path + f'aug{count}_{image}.png'
                    cv2.imwrite(file_name, flipped)
                    count += 1
        Parallel(n_jobs=os.cpu_count())(delayed(worker)(image) for image in tqdm.tqdm(image_paths, ncols=80))

class ImageStacker:

    def __init__(self):
        self.compA = pd.read_csv('./data/cocrystal/jan_raw_data.csv').Component1
        self.compB = pd.read_csv('./data/cocrystal/jan_raw_data.csv').Component2
        self.label = pd.read_csv('./data/cocrystal/jan_raw_data.csv').Outcome

    def get_image(self, filename):
        img = cv2.imread(filename)
        return img

    def join_image(self, img1, img2):
        img = np.concatenate((img1, img2), axis=0)
        return img

    def do_joining(self):
        read_path = './data/cocrystal/images/'
        pathA = [read_path + x + '.png' for x in self.compA]
        pathB = [read_path + x + '.png' for x in self.compB]
        save_path = './data/cocrystal/concat_images/'
        counter = 0
        paths = []
        labels = []
        for i in tqdm.tqdm(range(len(pathA))):
        # for i in tqdm.tqdm(range(5)):
            img1 = self.get_image(pathA[i])
            img2 = self.get_image(pathB[i])
            img_join = self.join_image(img1, img2)
            cv2.imwrite((save_path + str(counter) + '.png'), img_join)
            paths.append((save_path + str(counter) + '.png'))
            labels.append(self.label[i])
            counter = counter + 1

            img_join = self.join_image(img2, img1)
            cv2.imwrite((save_path + str(counter) + '.png'), img_join)
            paths.append((save_path + str(counter) + '.png'))
            labels.append(self.label[i])
            counter = counter + 1
        return paths, labels

    def fastai_data_table(self):
        paths, labels = self.do_joining()
        df = pd.DataFrame(paths, columns=['paths'])
        df['labels'] = labels
        return df