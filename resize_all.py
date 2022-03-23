import torch.utils.data as data
from glob import glob
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import imageio
from tqdm import tqdm


class DataSet(data.Dataset):
    def __init__(self, img_dir, resize):
        super(DataSet, self).__init__()
        self.img_paths = glob('{:s}/*'.format(img_dir))
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize))])

    def __getitem__(self, item):
        img = Image.open(self.img_paths[item]).convert('L')  # To convert grayscale images
        # img = Image.open(self.img_paths[item]) # To convert color images
        img = self.transform(img)

        return img, self.img_paths[item]

    def __len__(self):
        return len(self.img_paths)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./Data/')
    parser.add_argument('--resize', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='./Resize_Data/')
    args = parser.parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    for file_path in tqdm(os.listdir(args.img_dir)):
        new_img_dir = args.img_dir + file_path + '/'
        dataset = DataSet(new_img_dir, args.resize)
        print('dataset:', len(dataset))
        for i in tqdm(range(len(dataset))):
            img, path = dataset[i]
            path = os.path.basename(path)
            print('Processing:', path)

            if not os.path.exists(args.save_dir + file_path + '/'):
                os.mkdir(args.save_dir + file_path + '/')

            imageio.imwrite(args.save_dir + file_path + '/' + path[0:path.find('.')] + '.png', img)  # Convert to png
