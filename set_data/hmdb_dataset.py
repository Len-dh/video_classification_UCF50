import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from torch.nn.utils.rnn import pack_sequence
import matplotlib.pyplot as plt 




class hmdb_dataset(Dataset):
    
    def __init__(self, image_seq_paths, labels, transform=None):
        """
        Args:
            image_seq_paths (list of list containing paths of image sequence)
            labels (list of labels for each sequence)
            transform (callable, optional): Optional transform to be applied
                on one sequence.
        """
        self.image_seq_paths = image_seq_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        img_paths = self.image_seq_paths[idx]
        print("img path", img_paths[0], " / and index : ", idx)

        image_seq = self.read_images(img_paths)
        # print("type image sequences : ", type(image_seq), " image_seq : ", image_seq)
        
        # LongTensor are for int64 instead of FloatTensor
        target = self.labels[idx]
        print("target : ", target, " / and index : ", idx)
        
        
        return image_seq, target

    def read_images(self, img_paths):
        image_seq = []

        if self.transform:
            # define the same seed for each sequence
            seed = np.random.randint(1e9)    
       
        # print('img paths', len(img_paths))#, img_paths)
        for img_path in img_paths:        
            image = Image.open(img_path).convert('RGB')
            # print("image in transform : ", type(image), image.size)
            # fig_img = plt.figure
            # plt.imshow(image)
            # plt.title('original image')
            # plt.show()

            # print("\nType of image : ",type(image))
            
            # print("\nType of toTansor_image_seq : ",type(toTansor_image_seq))    
            # print('bef', img_path, image.size)

            if self.transform:
                # set seed
                seed = np.random.randint(1e9) 
                random.seed(seed)
                np.random.seed(seed)
                image = self.transform(image)
                # print("image in transform : ", type(image))
                # plt.figure
                # plt.imshow(image)
                # plt.title('image with transfrom')
                # plt.show()



            # print('aft', image.size)
            loader = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
            image = loader(image) #.permute((1,2,0))#unsqueeze(0)
            # print("type of image loader : ", type(image), image.size) 
            # plt.figure
            # plt.imshow(image)
            # plt.title('Image with loader toTensor and resize 224/224')
            # plt.show()

            # print('aft', image.size())
            image_seq.append(image)
        # print(len(image_seq))
        image_seq = torch.stack(image_seq, dim=0)
        return image_seq