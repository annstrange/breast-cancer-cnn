 # Methods for reading files/data
import numpy as np
import sys
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from skimage import color, transform, restoration, io, feature


class ImagePipeline(object):
    
    def __init__(self, parent_dir):
        """
        Manages reading, transforming and saving images

        :param parent_dir: Name of the parent directory containing all the sub directories
        """
        # Define the parent directory
        self.parent_dir = parent_dir

        # Sub directory variables that are filled in when read()
        self.raw_sub_dir_names = None
        self.sub_dirs = None  # used for sub dirs of magnification level
        self.label_map = None

        # Image variables that are filled in when read() and vectorize()
        self.img_lst2 = []
        self.img_names2 = []
        self.features = None
        self.labels = None
        self.tumor_class_vector = None
        self.img_attribs = {}  # dictionary with known attributes by filename

        #mine
        root_dir = '../BreaKHis_v1/histology_slides/breast'
        srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}
        self.tumor_types = {'DC', 'LC', 'MC', 'PC', 'A', 'F', 'PT', 'TA'}        

    def _make_label_map(self):
        """
        Get the sub directory names and map them to numeric values (labels)

        :return: A dictionary of dir names to numeric values
        """
        return {label: i for i, label in enumerate(self.raw_sub_dir_names)}

    def _path_relative_to_parent(self, some_dir):
        """
        Get the full path of a sub directory relative to the parent

        :param some_dir: The name of a sub directory
        :return: Return the full path relative to the parent
        """
        cur_path = os.getcwd()
        return os.path.join(cur_path, self.parent_dir, some_dir)

    def _make_new_dir(self, new_dir):
        """
        Make a new sub directory with fully defined path relative to the parent directory

        :param new_dir: The name of a new sub dir
        """
        # Make a new directory for the new transformed images
        new_dir = self._path_relative_to_parent(new_dir)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        else:
            raise Exception('Directory already exist, please check...')

    def _empty_variables(self):
        """
        Reset all the image related instance variables
        """
        self.img_lst2 = []
        self.img_names2 = []
        self.features = None
        self.labels = None
        self.img_attribs = {}

    @staticmethod
    def _accepted_file_format(fname):
        """
        Return boolean of whether the file is of the accepted file format

        :param fname: Name of the file in question
        :return: True or False (if the file is accpeted or not)
        """
        formats = ['.png', '.jpg', '.jpeg', '.tiff']
        for fmt in formats:
            if fname.endswith(fmt):
                return True
        return False

    def _accepted_subdir(self, dir_name):
        """
        Return boolean of whether the dir is one we're working on

        :param fname: Name of the leaf dir in question e.g. 40X
        :return: True or False (if the subdir is accpeted or not)
        """
        subdirs = self.raw_sub_dir_names
        print ('raw sd names {} and data type {}'.format(self.raw_sub_dir_names, type(self.raw_sub_dir_names)))
        print('compare dir_name {} w subdirs list {}'.format(dir_name, subdirs))
        for sd in subdirs:
            print('compare dir_name {} w ea sd {}'.format(dir_name, sd))
            if dir_name == sd:
                return True
        return False    

    @staticmethod
    def _accepted_dir_name(dir_name):
        """
        Return boolean of whether the directory is of the accepted name (i.e. no hidden files)

        :param dir_name: Name of the directory in question
        :return: True or False (if the directory is hidden or not)
        """
        if dir_name.startswith('.'):
            return False
        else:
            return True

    def _assign_sub_dirs(self, sub_dirs=tuple('all')):
        """
        Assign the names (self.raw_sub_dir_names) and paths (self.sub_dirs) based on
        the sub dirs that are passed in, otherwise will just take everything in the
        parent dir

        :param sub_dirs: Tuple contain all the sub dir names, else default to all sub dirs
        """
        # Get the list of raw sub dir names
        if sub_dirs[0] == 'all':
            self.raw_sub_dir_names = tuple('40X', '100X', '200X', '400X')
        else:
            self.raw_sub_dir_names = sub_dirs
        # Make label to map raw sub dir names to numeric values
        self.label_map = self._make_label_map()

        # Get the full path of the raw sub dirs
        filtered_sub_dir = list(filter(self._accepted_dir_name, self.raw_sub_dir_names))
        self.sub_dirs = map(self._path_relative_to_parent, filtered_sub_dir)

    def read(self, sub_dirs=tuple('all'), brief_mode=False):
        """
        Read images from each sub directories into a list of matrix (self.img_lst2)

        :param sub_dirs: Tuple contain all the sub dir names, else default to all sub dirs
        e.g. specify ('40X') for only the 40X sub_dir aka magnification level
        """
        if (type(sub_dirs)) != 'list':
            print('Invalid sub_dirs data type: {}. List expected'.format(type(sub_dirs)))

        # Empty the variables containing the image arrays and image names, features and labels
        self._empty_variables()

        # Assign the sub dir names based on what is passed in
        self._assign_sub_dirs(sub_dirs=sub_dirs)  

        print('root dir {}'.format(self.parent_dir))
        # ToDo: test parent_dir is valid and give error message

        for root,d_names,f_names in os.walk(self.parent_dir):
            print ('traverse {}\t {}\t {}'.format(root, d_names, len(f_names)))  

            # Test last folder for 40X, 100X, 200X, 400X
            fldr = root.split("/")[-1:]
            print("fldr {}".format(fldr))

            if self._accepted_subdir(fldr[0]):
                print ('valid subdir {}'.format(fldr[0]))
                img_names = list(filter(self._accepted_file_format, f_names))
                if len(img_names) > 0:
                    print('image name 0 {}'.format(os.path.join(root, img_names[0])))
                if (brief_mode == True and len(img_names) > 0):   
                     img_names = [img_names[0]]
                self.img_names2.append(img_names)
                #print ('img_names: {} total shape {}'.format(img_names, self.img_names2 ))
                  
                img_lst = [io.imread(os.path.join(root, fname)) for fname in img_names]
                print('len img_lst {}'.format(len(img_lst)))

                self.img_lst2.append(img_lst)

        # img_lst2 is a list of lists...
        print('img_lst2 should have list of {} patients, 1st has {} images, filenames of shape {}'.format(len(self.img_lst2), len(self.img_lst2[0]), (self.img_lst2[0][0].shape)))
        for i in np.arange(0, len(self.img_lst2)):
            for j in np.arange(0, len(self.img_lst2[i])):
                if (i == 0 and j == 0 ):
                    print ('first image shape {}'.format(self.img_lst2[i][j].shape))

        # img_names2 is also list of lists...
        print('img_names2 should have list of {} patients, 1st has {} entries'.format(len(self.img_names2), len(self.img_names2[0])))
        for i in np.arange(0, len(self.img_names2)):
            for j in np.arange(0, len(self.img_names2[i])):
                if (i == 0 and j == 0 ):
                    print ('first names entry {}'.format(self.img_names2[i][j]))
            
        # collapse outer nesting to avoid later problems with different length inner lists
        self.img_lst2 = self.collapse_outer_list(self.img_lst2)    
        self.img_names2 = self.collapse_outer_list(self.img_names2) 

        # img_lst2 is a list ...
        print('img_lst2 should have list of {} patients x images, filenames of shape {} '.format(len(self.img_lst2), len(self.img_lst2[0]) ))
        for i in np.arange(0, len(self.img_lst2)):
            if (i == 0 ):
                print ('first image shape {}'.format(self.img_lst2[i].shape))

        # img_names2 is also lists...
        print('img_names2 should have list of {} patients * images'.format(len(self.img_names2)))
        for i in np.arange(0, len(self.img_names2)):
            if (i == 0 ):
                print ('first names entry {}'.format(self.img_names2[i]))

    '''
    def save(self, keyword):
        """
        Save the current images into new sub directories

        :param keyword: The string to append to the end of the original names for the
                        new sub directories that we are saving to
        """
        # Use the keyword to make the new names of the sub dirs
        new_sub_dirs = ['%s.%s' % (sub_dir, keyword) for sub_dir in self.sub_dirs]

        # Loop through the sub dirs and loop through images to save images to the respective subdir
        for new_sub_dir, img_names, img_lst in zip(new_sub_dirs, self.img_names2, self.img_lst2):
            new_sub_dir_path = self._path_relative_to_parent(new_sub_dir)
            self._make_new_dir(new_sub_dir_path)

            for fname, img_arr in zip(img_names, img_lst):
                io.imsave(os.path.join(new_sub_dir_path, fname), img_arr)

        self.sub_dirs = new_sub_dirs

    def show(self, sub_dir, img_ind):
        """
        View the nth image in the nth class

        :param sub_dir: The name of the category
        :param img_ind: The index of the category of images
        """
        sub_dir_ind = self.label_map[sub_dir]
        # prolly dont' want this
        print('in ip.show, sub_dir = {} and sub_dir_ind = {} and map {}'.format(sub_dir, sub_dir_ind, self.label_map))

        io.imshow(self.img_lst2[0][img_ind])  # prolly change sub_dir to sub_dir_ind (0-81)
        plt.show()
    '''
    def collapse_outer_list(self, nested_list):
        '''
        Input: list of a list of whatever
        Returns: longer list with the first level of nesting removed, like a partial flatten
        '''
        new_list = []
        for i, sub_list in enumerate(nested_list):
            new_list.extend(sub_list)
        print('collapsed {}x{}ish list to {}'.format(len(nested_list), len(nested_list[0]), len(new_list)))
        return new_list

    def get_image(self, img_ind):
        """
        View the nth image 

        :param sub_dir: The name of the category # deprecated
        :param img_ind: The index of the category of images
        """
        #sub_dir_ind = self.label_map[sub_dir]
        # prolly dont' want this
        #print('in ip.show, sub_dir = {} and sub_dir_ind = {} and map {}'.format(sub_dir, sub_dir_ind, self.label_map))

        io.imshow(self.img_lst2[img_ind])  
        plt.show()    
        return self.img_lst2[img_ind]

    '''
    def savefig(self, sub_dir, img_ind, file_attrib):
        """
        Save the nth image in the nth class

        :param sub_dir: The name of the category
        :param img_ind: The index of the category of images
        """
        print ('**** in savefig() *****')
        sub_dir_ind = self.label_map[sub_dir]
        io.imshow(self.img_lst2[sub_dir_ind][img_ind])
        plt.show()
        filename = sub_dir + str(img_ind) + '_' + file_attrib + '.png'
        print(filename)
        plt.savefig(filename)    
    '''

    def transform_dict (self, func, params, img_dict):
        '''
        Args: img_dict is a dictionary of filename and images
        Returns: img_dict after the function is applied
        This function gets the parameters needed to call transform for
        each image, builds a new image dictionary, shows and saves it,
        and returns it.  
        No permanent object updates are made.
        Output of this can be fed as input, to set up a series of transformations
        on our sample biopsies
        '''
        rtn_dict = {}
        # get sub_dir and img_ind for each dictionary item
        for k, v in img_dict.items():
            idx = self.get_img_lst2_indices(k)

            changed_img = self.get_transform_copy(func, params, img_ind=idx )
            rtn_dict[k] = changed_img

        return rtn_dict

    def get_transform_copy(self, func, params, img_ind=None):
        """
        Takes a function and apply to every img_arr in self.img_arr returning a copy 
        instead of updating the source images
        Have to option to transform one as  a test case

        :param sub_dir_ind: The index for the image in img_lst2
        :param img_ind: The index of the category of images

        returns transformed image(s) instead of updating self
        """
        # Apply to one test case
        if img_ind is not None:
            img_arr = self.img_lst2[img_ind]
            img_arr = func(img_arr, **params).astype(float)
            return(img_arr)
        else:  # This is a lot, fyi
            new_img_lst2 = self.img_lst2
            new_img_lst2 = ([func(img_arr, **params).astype(float) for img_arr in self.img_lst2])
            return(new_img_lst2)

    def transform(self, func, params, img_ind=None):
        """
        Takes a function and apply to every img_arr in self.img_arr.
        Have to option to transform one as  a test case

        :param sub_dir_ind: The index for the image in img_lst2  # deprecated
        :param img_ind: The index of the category of images
        """
        # Apply to one test case
        if img_ind is not None:
            img_arr = self.img_lst2[img_ind]
            img_arr = func(img_arr, **params).astype(float)
            io.imshow(img_arr)
            plt.show()
        # Apply the function and parameters to all the images
        else:
            new_img_lst2 = ([func(img_arr, **params).astype(float) for img_arr in self.img_lst2])
            self.img_lst2 = new_img_lst2

    def grayscale(self, img_ind=None):
        """
        Grayscale all the images in self.img_lst2

        :param img_ind: The index of the image within the chosen sub dir
        """
        self.transform(color.rgb2gray, {}, img_ind=img_ind)

    def canny(self, img_ind=None):
        """
        Apply the canny edge detection algorithm to all the images in self.img_lst2

        :param img_ind: The index of the image within the chosen sub dir
        """
        self.transform(feature.canny, {}, img_ind=img_ind)

    def dye_separation(self, img_ind=None):
        """
        Apply the canny edge detection algorithm to all the images in self.img_lst2

        :param img_ind: The index of the image within the chosen sub dir
        """
        self.transform(dye_separation, {}, img_ind=img_ind)

    def tv_denoise(self, weight=2, multichannel=True, img_ind=None):
        """
        Apply to total variation denoise to all the images in self.img_lst2

        :param img_ind: The index of the image within the chosen sub dir
        """
        self.transform(restoration.denoise_tv_chambolle,
                       dict(weight=weight, multichannel=multichannel),
                       img_ind=img_ind)

    def resize(self, shape, save=False):
        """
        Resize all images in self.img_lst2 to a uniform shape

        :param shape: A tuple of 2 or 3 dimensions depending on if your images are grayscaled or not
        :param save: Boolean to save the images in new directories or not
        """
        self.transform(transform.resize, dict(output_shape=shape))
        
        #if save:
        #    shape_str = '_'.join(map(str, shape))
        #    self.save(shape_str)

    def _vectorize_features(self):
        """
        Take a list of images and vectorize all the images. Returns a feature matrix where each
        row represents an image
        """

        print('img_lst2 len {} should be 1 '.format(len(self.img_lst2) ), len(self.img_lst2[0]), len(self.img_lst2[1]))

        for i in np.arange(0, len(self.img_lst2)):
            #j = len(self.img_lst2[i])
            print ('len of sub list {} {}'.format(i, len(self.img_lst2)))

        # tupli-tizes each row in each sub list, sometimes of different lengths (ok)
        row_tup = tuple(img_arr.ravel()[np.newaxis, :]
                        for img_arr in self.img_lst2 )
        # r_ will mess up, however, if the image dimensions are not consistent with error
        # ValueError: all the input array dimensions except for the concatenation axis must match exactly
        self.features = np.r_[row_tup]
        
    def _vectorize_features_b(self):
        """
        Take a list of images and vectorize all the images. Returns a 4D feature matrix 
        """

        print('img_lst2 len {} '.format(len(self.img_lst2) ))
        #dim1 = len(self.img_lst2)
        #dim2 = len(self.img_lst2[0])
        #dim3 = len(self.img_lst2[1])
        #dim4 = 227
        #for i in np.arange(0, len(self.img_lst2)):
        #for j in np.arange(0, len(self.img_lst2[i])):
        #print ('whole shebang patient 0{}'.format(self.img_lst2[0]))
        
        to_array = np.array(self.img_lst2, dtype=np.float32)
        '''
        outer_np_list = []
        for i in np.arange(len(self.img_lst2)):
            to_array = np.array(self.img_lst2[i], dtype=np.float64)
            outer_np_list.append(to_array)
        '''
        #full_np = np.stack(outer_np_list)    #ValueError('all input arrays must have the same shape')
        #print('shape of full_np {}'.format(full_np.shape))

        print('shape of np array converted img_lst2 going in {}'.format(to_array.shape))
        #new_dim = to_array.shape[0] * to_array.shape[1]
        #new_shape = tuple( (to_array.shape[0] to_array.shape[1], to_array.shape[3]))
        #print( 'lets reshape as {}'.format(new_shape))

        #img_arr = np.stack(to_array.reshape(new_shape))

        #print('shape of sub list array-itized inner {} main '.format(img_arr.shape))
                
        # tupli-tizes each row in each sub list, sometimes of different lengths (ok)
        
        #row_tup = tuple(img_arr.ravel()[np.newaxis, :]
        #                for img_lst in self.img_lst2 for img_arr in img_lst)
        # r_ will mess up, however, if the image dimensions are not consistent with error
        # ValueError: all the input array dimensions except for the concatenation axis must match exactly
        self.features = to_array

    def _vectorize_labels(self):
        """
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        """
        # Get the labels with the dimensions of the number of image files
        self.labels = np.concatenate([np.repeat(i, len(img_names))
                                     for i, img_names in enumerate(self.img_names2)])

    @staticmethod
    def _parse_filename(filename):
        '''
        grabs the bits and pieces from the filename, e.g. SOB_M_DC-14-11951-100-008
            BIOPSY_PROCEDURE, e.g. “SOB” is the Surgical Open Biopsy procedure name and CNB is for Core Needle Biopsy
            TUMOR_CLASS  “M” for malignant, “B” for benign
            TUMOR_TYPE: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenona (TA), carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC) and papillary carcinoma (PC)
            YEAR
            SLIDE_ID
            MAG: Magnification Level (40x, 100x, 200x, or 400x)
            SEQ: Sample number (e.g 001, 002, etc)
        returns a single entry dictionary line
        '''

        d1 = {}
        seq = filename.split('-')[-1].replace('.png', '')   # e.g. 010
        mag = filename.split('-')[-2]   # 40
        code = filename.split('-')[-3]  # 22549AB
        year = filename.split('-')[-4] # 14
        proc = filename.split('_')[0]  # SOB
        m_b = filename.split('_')[1].split('-')[0]  # B
        type1 = filename.split('-')[-5].split('_')[-1]

        d1[filename] = {'tumor_class': m_b, \
            'biopsy_procedure': proc, \
            'tumor_type': type1, \
            'year' : year,  \
            'slide_id' : code, \
            'mag' : mag, \
            'seq' : seq  }

        return d1     


    def vectorize_y (self):
        ''' 
        Assuming _vectorize_attribs has parsed the filenames for M (malignant) and B (benign), 
        create the y vector, represented in self.tumor_class_vector

        '''

        arr = np.zeros(len(self.img_attribs), dtype = np.int32)
        i = 0
        for k1, v1 in self.img_attribs.items():
            if i < 40:
                print (v1)
            if (v1['tumor_class'] == "M"):
                arr[i] = 1
                i += 1        

        print ('tumor_class vector num malig {} out of {} samples'.format(np.sum(arr), len(arr)))
        self.tumor_class_vector = arr



        #self.diagnosis = np.concatenate([np.repeat( v1['diagnosis'])
        #            for k1, v1 in self.img_attribs.items()])

        #print ('tumor_class vector : {}'.format(self.tumor_class_vector [:20]))




    def _vectorize_attribs(self):
        ''' 
        parse and collect the attributes of each file by filename

        '''
        d1 = {}
        for i in np.arange(0, len(self.img_names2)):
            #print ('image name to parse {}'.format(self.img_names2[i][j]))
            filename = self.img_names2[i]
            d_one = self._parse_filename(filename)
            d1[filename] = d_one[filename]
                
        # let's add original image sizes (assumes same dimensions)
        # only do this the first time
        for i in np.arange(0, len(self.img_lst2)):
            fn = self.img_names2[i]
            d1[fn].update({'image_size' : self.img_lst2[i].shape})
            if i == 0:
                print (d1[fn])

        self.img_attribs = d1                
        #print (self.img_attribs)



    def vectorize(self):
        """
        Return (feature matrix, the response) if output is True, otherwise set as instance variable.
        Run at the end of all transformations
        """
        # these are all arrays
        self._vectorize_features_b()
        print ('features shape {}'.format( self.features.shape))
        self._vectorize_labels()
        print ('labels shape {}'.format( self.labels.shape))
        self._vectorize_attribs()
        print ('attribs (dict) len {}'.format( len(self.img_attribs)))

    def get_img_lst2_indices(self, filename):
        '''
        Gets the sub_dir_ind (first level index) and 2nd level index to use
        as an address for the images in self.img_lst2
        (as well as self.img_names2 and self.img_attribs)
        Used before calling transform()
        returns tuple of outer_i and inner_i

        '''
 
        idx = self.img_names2.index(filename)    
        print ('found fn {} at nested index [{}]'.format(filename, idx))

        return idx
        
    def get_one_of_each(self):
        '''
        For image analysis, let's have set of 8 slides to look at, one of each tumor type

        returns list of images; one of each tumor_type (8 expected)
        '''
        img8_lst = []
        img_dict = {}
        # look through img_attribs for first instance of each tumor type
        for i, ttype in enumerate (self.tumor_types):
            print ('getting sample of type {}'.format(ttype))
            # list of a generator expression (which does same thing as a list comprehension, fancy!)
            found_keys = list(k for k, v in self.img_attribs.items() if ttype == v['tumor_type'])
            #print ('found_keys {}'.format(found_keys))
            fn = found_keys[0]
            # get first one's filename to test
            #fn = 'SOB_M_PC-14-9146-40-001.png'

            # find filename position in img_names()
            #j = self.img_names[fn].index
            #found_keys2 = list(j for j, lst in enumerate(self.img_names2) if fn in lst)

            outer_i, inner_i = self.get_img_lst2_indices(fn)
            '''
            outer_i = next(j for j, lst in enumerate(self.img_names2) if fn in lst)
            print ('outer_i {} {}'.format(type(outer_i), outer_i))
            if outer_i == None:
                print ('**** There is a bug *****')
            inner_i = self.img_names2[outer_i].index(fn)    
            print ('found fn {} at nested index [{}] [{}]'.format(fn, outer_i, inner_i))
            '''
            img = self.get_image(outer_i, inner_i)

            img8_lst.append(img)
            img_dict[fn] = img

        print('returning {} sample images in list shape {}'.format(len(img8_lst), img8_lst[0].shape))

        # more clever?    
        # img_dict2 =  dict(zip(self.tumor_types, img8_lst))
        # print('is same? {}'.format(img_dict2))    
        return img_dict

if __name__ == '__main__':

    # see if we can locate our .png files relative to here
    '''
    assumes folder structure like
    data
    - BreaKHis_v1 with unzipped contents
        - histology_slides
            - breast  (more subdirs and files)

    
    root_dir = './data/BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB'
    
    IP = ImagePipeline(root_dir)
    for root,d_names,f_names in os.walk(root_dir):
     
        if (len(f_names) > 0):    # files found ending in .png?
            print (root, d_names, f_names) 
            # loop through filenames, read image and parse/store attributes
    '''

# FROM Keras Dog/Cat ex
'''
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join("PetImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

"""
## Generate a `Dataset`
"""

image_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

"""
## Visualize the data

Here are the first 9 images in the training dataset. As you can see, label 1 is "dog"
 and label 0 is "cat".
"""


plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")
'''