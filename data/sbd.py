     
# class SBDClassSeg(VOCClassSeg):
#     def __init__(self, root, split='train.txt', transform=False):
#         self.root = root
#         self.split = split
#         self._transform = transform

#         dataset_dir = root + 'benchmark_RELEASE/dataset/'
#         imgsets_file = dataset_dir + split

#         with open(imgsets_file) as f:
#             imglist = f.readlines()

#         data, label = [], []
#         for img in imglist:
#             img = img.strip()
#             data.append(dataset_dir + 'img/%s.jpg' % img)
#             label.append(dataset_dir + 'cls/%s.mat' % img)

#         self.data = data
#         self.label = label

#     def __getitem__(self, idx):
#         data_file = self.data[idx]
#         label_file = self.label[idx]

#         # load image
#         img = cv2.imread(data_file)

#         # load label
#         mat = scipy.io.loadmat(label_file)
#         label = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)

#         if self._transform:
#             return self.transform(img, label)
#         else:
#             return img, label
