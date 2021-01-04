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
from skimage import color, transform, restoration, io, feature, util
from itertools import compress
# flake8: noqa

verbose = False

class ImagePipeline(object):
    
    def __init__(self, parent_dir):
        """
        Manages reading, transforming and saving images
        Arguments:
            parent_dir: Name of the parent directory containing all the sub directories

        """
        # Define the parent directory
        self.parent_dir = parent_dir

        # Sub directory variables that are filled in when read()
        self.raw_sub_dir_names = None
        self.sub_dirs = None  # used for sub dirs of magnification level
        self.label_map = None

        # Image variables that are filled in when read() and vectorize()
        self.images_list = []
        self.images_filename_list = []
        self.features = None
        #self.labels = None
        self.tumor_class_vector = None
        self.images_attributes = {}  # dictionary with known attributes by filename
        self.group_list = []

        #dataset specifics
        #root_dir = '../BreaKHis_v1/histology_slides/breast'
        srcfiles = {'DC': '%s/malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': '%s/malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': '%s/malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': '%s/malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': '%s/benign/SOB/adenosis/%s/%sX/%s',
                'F': '%s/benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': '%s/benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': '%s/benign/SOB/tubular_adenoma/%s/%sX/%s'}
        self.tumor_types = {'DC', 'LC', 'MC', 'PC', 'A', 'F', 'PT', 'TA'}        
        self.malig_tumor_types = {'DC', 'MC', 'PC', 'LC'}
        self.benign_tumor_types = {'A', 'TA', 'F', 'PT'}

    #def _make_label_map(self):
        """
        Get the sub directory names and map them to numeric values (labels)

        :return: A dictionary of dir names to numeric values
        """
        #return {label: i for i, label in enumerate(self.raw_sub_dir_names)}

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
        self.images_list = []
        self.images_filename_list = []
        # self.labels = None
        self.images_attributes = {}
        self.features = None
        self.tumor_class_vector = None
        self.group_list = []

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
        for sd in subdirs:
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

    def _assign_sub_dirs(self, sub_dirs=['all']):
        """
        Arguments: 
        sub_dirs (optional) looks like list(['40X', '100X', '200X', '400X']) or some subset 
            to specify which magnification to read in. 

        """
        # Get the list of raw sub dir names
        if sub_dirs[0] == 'all':
            self.raw_sub_dir_names = (['40X', '100X', '200X', '400X'])
        else:
            self.raw_sub_dir_names = sub_dirs

        if verbose: print('raw_sub_dir_names {}'.format(self.raw_sub_dir_names)) 
        # Make label to map raw sub dir names to numeric values
        #self.label_map = self._make_label_map()

        # Get the full path of the raw sub dirs
        filtered_sub_dir = list(filter(self._accepted_dir_name, self.raw_sub_dir_names))
        self.sub_dirs = map(self._path_relative_to_parent, filtered_sub_dir)

    def read(self, sub_dirs=['all'], brief_mode=False):
        """
        Read images from each sub directories into a list of matrix (self.images_list)
        Arguments:
        sub_dirs: List or tuple contain all the sub dir names, else default to all sub dirs
            e.g. specify (['40X']) for only the 40X sub_dir aka magnification level
        """

        # Empty the variables containing the image arrays and image names, features and labels
        self._empty_variables()

        # Assign the sub dir names based on what is passed in
        self._assign_sub_dirs(sub_dirs=sub_dirs)  

        if verbose: print('root dir {}'.format(self.parent_dir)) 
        # ToDo: test parent_dir is valid and give error message

        for root,d_names,f_names in os.walk(self.parent_dir):
            if verbose: print ('traverse {}\t {}\t {}'.format(root, d_names, len(f_names)))  

            # Test last folder for 40X, 100X, 200X, 400X
            fldr = root.split("/")[-1:]
            #print("fldr {}".format(fldr))

            if self._accepted_subdir(fldr[0]):
                if verbose: print ('valid subdir {}'.format(fldr[0])) 
                img_names = list(filter(self._accepted_file_format, f_names))
                if len(img_names) > 0:
                    if verbose: print('image name 0 {}'.format(os.path.join(root, img_names[0]))) 
                if (brief_mode == True and len(img_names) > 0):   
                     img_names = [img_names[0]]
                self.images_filename_list.append(img_names)
                #print ('img_names: {} total shape {}'.format(img_names, self.images_filename_list ))
                  
                img_lst = [io.imread(os.path.join(root, fname)) for fname in img_names]
                if verbose: print('len img_lst {}'.format(len(img_lst)))

                self.images_list.append(img_lst)

        # images_list is a list of lists...
        #print('images_list should have list of {} patients, 1st has {} images, filenames of shape {}'.format(len(self.images_list), len(self.images_list[0]), (self.images_list[0][0].shape)))
        for i in np.arange(0, len(self.images_list)):
            for j in np.arange(0, len(self.images_list[i])):
                if (i == 0 and j == 0 ):
                    print ('first image shape {}'.format(self.images_list[i][j].shape))

        # images_filename_list is also list of lists...
        #print('images_filename_list should have list of {} patients, 1st has {} entries'.format(len(self.images_filename_list), len(self.images_filename_list[0])))
        for i in np.arange(0, len(self.images_filename_list)):
            for j in np.arange(0, len(self.images_filename_list[i])):
                if (i == 0 and j == 0 ):
                    print ('first names entry {}'.format(self.images_filename_list[i][j]))
            
        # collapse outer nesting to avoid later problems with different length inner lists
        self.images_list = self.collapse_outer_list(self.images_list)    
        self.images_filename_list = self.collapse_outer_list(self.images_filename_list) 

        # images_list is a list ...
        print('images_list should have list of {} patients x images, filenames of shape {} '.format(len(self.images_list), len(self.images_list[0]) ))
        for i in np.arange(0, len(self.images_list)):
            if (i == 0 ):
                print ('first image shape {}'.format(self.images_list[i].shape))

        # images_filename_list is also lists...
        print('images_filename_list should have list of {} patients * images'.format(len(self.images_filename_list)))
        for i in np.arange(0, len(self.images_filename_list)):
            if (i == 0 ):
                print ('first names entry {}'.format(self.images_filename_list[i]))

        self._parse_attributes()

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
        for new_sub_dir, img_names, img_lst in zip(new_sub_dirs, self.images_filename_list, self.images_list):
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

        io.imshow(self.images_list[0][img_ind])  # prolly change sub_dir to sub_dir_ind (0-81)
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
        if verbose: print('collapsed {}x{}ish list to {}'.format(len(nested_list), len(nested_list[0]), len(new_list))) 
        return new_list

    def get_image(self, img_ind):
        """
        View the nth image 
        Arguments:
            sub_dir: The name of the category # deprecated
            img_ind: The index of the category of images
        """

        io.imshow(self.images_list[img_ind])  
        plt.show()    
        return self.images_list[img_ind]
            

    def transform_dict (self, func, params, img_dict):
        '''
        Arguments: 
            img_dict is a dictionary of filename and images
        Returns: 
            img_dict after the function is applied
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
            idx = self.get_image_index(k)

            changed_img = self.get_transform_copy(func, params, img_ind=idx )
            rtn_dict[k] = changed_img

        return rtn_dict

    def get_transform_copy(self, func, params, img_ind=None):
        """
        Takes a function and apply to every img_arr in self.img_arr returning a copy 
        instead of updating the source images
        Have to option to transform one as  a test case
        Arguments:
            sub_dir_ind: The index for the image in images_list
            img_ind: The index of the category of images

        Returns: 
            transformed image(s) instead of updating self
        """
        # Apply to one test case
        if img_ind is not None:
            img_arr = self.images_list[img_ind]
            img_arr = func(img_arr, **params).astype(float)
            return(img_arr)
        else:  # This is a lot, fyi
            new_images_list = self.images_list
            new_images_list = ([func(img_arr, **params).astype(float) for img_arr in self.images_list])
            return(new_images_list)

    def transform(self, func, params, img_ind=None):
        """
        Takes a function and apply to every img_arr in self.img_arr.
        Have to option to transform one as  a test case
        Arguments:
            sub_dir_ind: The index for the image in images_list  # deprecated
            img_ind: The index of the category of images
        """
        # Apply to one test case
        if img_ind is not None:
            img_arr = self.images_list[img_ind]
            img_arr = func(img_arr, **params).astype(float)
            io.imshow(img_arr)
            plt.show()
        # Apply the function and parameters to all the images
        else:
            new_images_list = ([func(img_arr, **params).astype(float) for img_arr in self.images_list])
            self.images_list = new_images_list

    def grayscale(self, img_ind=None):
        """
        Grayscale all the images in self.images_list
        Arguments:
            img_ind: The index of the image within the chosen sub dir
        """
        self.transform(color.rgb2gray, {}, img_ind=img_ind)

    def canny(self, img_ind=None):
        """
        Apply the canny edge detection algorithm to all the images in self.images_list
        Arguments:
            img_ind: The index of the image within the chosen sub dir
        """
        self.transform(feature.canny, {}, img_ind=img_ind)

    def dye_separation(self, img_ind=None):
        """
        Apply the dye separation detection algorithm to all the images in self.images_list
        Arguments:
            img_ind: The index of the image within the chosen sub dir
        """
        self.transform(self.dye_separation, {}, img_ind=img_ind)

    def tv_denoise(self, weight=.3, multichannel=True, img_ind=None):
        """
        Apply to total variation denoise to all the images in self.images_list
        Arguments:
            img_ind: The index of the image within the chosen sub dir
            weight: the chambolle denoising weight to use 
        """
        self.transform(restoration.denoise_tv_chambolle,
                       dict(weight=weight, multichannel=multichannel),
                       img_ind=img_ind)

    def resize(self, shape, save=False):
        """
        Resize all images in self.images_list to a uniform shape
        Arguments:
            shape: A tuple of 2 or 3 dimensions depending on if your images are grayscaled or not
            save: Boolean to save the images in new directories or not
        """
        self.transform(transform.resize, dict(output_shape=shape))
        
        #if save:
        #    shape_str = '_'.join(map(str, shape))
        #    self.save(shape_str)

    '''
    def crop(self, shape, img_ind=None):
        """
        Crop all images in self.images_list to a uniform shape
        Arguments:
            shape: A tuple of 2 or 3 dimensions depending on if your images are grayscaled or not
        """
        self.transform(crop, dict(width=shape))    
    '''


    def _vectorize_X(self):
        """
        Take a list of images and vectorize all the images. Returns a 3D feature matrix 
        """
        print('images_list len {} '.format(len(self.images_list) ))
        to_array = np.array(self.images_list, dtype=np.float32)
        print('shape of np array converted images_list going in {}'.format(to_array.shape))

        self.features = to_array

    '''
    def _vectorize_labels(self):
        """
        Convert file names to a list of y labels (in the example it would be either cat or dog, 1 or 0)
        """
        # Get the labels with the dimensions of the number of image files
        self.labels = np.concatenate([np.repeat(i, len(img_names))
                                     for i, img_names in enumerate(self.images_filename_list)])
    '''

    @staticmethod
    def _parse_filename(filename):
        '''
        grabs the bits and pieces from the filename, 
            BIOPSY_PROCEDURE, e.g. “SOB” is the Surgical Open Biopsy procedure name and CNB is for Core Needle Biopsy
            TUMOR_CLASS  “M” for malignant, “B” for benign
            TUMOR_TYPE: adenosis (A), fibroadenoma (F), phyllodes tumor (PT), and tubular adenona (TA), carcinoma (DC), lobular carcinoma (LC), mucinous carcinoma (MC) and papillary carcinoma (PC)
            YEAR
            SLIDE_ID
            MAG: Magnification Level (40x, 100x, 200x, or 400x)
            SEQ: Sample number (e.g 001, 002, etc)

        Arguments:
            filename is the key, not including path e.g. SOB_M_DC-14-11951-100-008
        returns 
            single entry dictionary line
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


    def _vectorize_y (self):
        ''' 
        Assuming _parse_attributes has parsed the filenames for M (malignant) and B (benign), 
        create the y vector, represented in self.tumor_class_vector

        Return array of 1 or 0 for Malignant or Benign
        '''

        arr = np.zeros(len(self.images_filename_list), dtype = np.int32)
        i = 0

        for i, filename in enumerate(self.images_filename_list):

            v1 = self.images_attributes[filename]
            if i < 4:
                print (v1) 
            if (v1['tumor_class'] == "M"):
                arr[i] = 1    

        print ('tumor_class vector num malig {} out of {} samples'.format(np.sum(arr), len(arr)))
        print ('tumor_class vector looks like {}'.format(arr))
        self.tumor_class_vector = arr


    def _get_patient_group_list (self):
        ''' 
        Assuming _parse_attributes has parsed the filenames for M (malignant) and B (benign), 
        create a list to correspond to the y vector, with patient ids (slide_id)

        Return list of patient ids
        '''

        slide_id_list = []
        for filename in self.images_filename_list:

            v1 = self.images_attributes[filename]
            slide_id_list.append(v1['slide_id'])     

        distinct_ids = set(slide_id_list)
        print ('number of distinct patients: {}'.format(len(distinct_ids)))

        return slide_id_list

    def get_images_of_class (self, tumor_class, magnification=None):
        '''
        Useful for EDA, browsing images
        Arguments:
            class is string to indicate tumor class of 'M' or 'B'
            magnification is int 40, 100, 200, or 400. None should look for all available from initial read
        returns:
            list of all images pre-train/test of matching class and magnfication
            list of all image names that match the criteria
        '''
        lst_filtered_imgs = []
        lst_filtered_names = []
        for k1, v1 in self.images_attributes.items():
            if (v1['tumor_class'] == tumor_class):
                if v1['mag' == magnification] or magnification is None :
                    print ('getting filename {}'.format(k1))
                    lst_filtered_names.append(k1)
                    lst_filtered_imgs.append (self.get_image(k1))

        return lst_filtered_imgs, lst_filtered_names

    def _parse_attributes(self):
        ''' 
        parse and collect the attributes of each file by filename

        '''
        d1 = {}
        for i in np.arange(0, len(self.images_filename_list)):
            #print ('image name to parse {}'.format(self.images_filename_list[i][j]))
            filename = self.images_filename_list[i]
            d_one = self._parse_filename(filename)
            d1[filename] = d_one[filename]
                
        # add image sizes
        # only do this the first time
        for i in np.arange(0, len(self.images_list)):
            fn = self.images_filename_list[i]
            d1[fn].update({'image_size' : self.images_list[i].shape})
            if i == 0:
                print (d1[fn])

        self.images_attributes = d1                
        #print (self.images_attributes)


    def vectorize(self):
        """
        sets feature matrix (X), otherwise set as instance variable.
        Run at the end of all transformations
        """
        self._vectorize_X()
        print ('features shape {}'.format( self.features.shape))
        print ('attribs (dict) len {}'.format( len(self.images_attributes)))
        self._vectorize_y()
        self.group_list = self._get_patient_group_list()

    def double_the_benigns(self):
        '''
        Extracts the benign images in X and y and doubles them to even out the target y values.
        This should be used with data augmentation so we hopefully don't get the same image twice very often
        pseudo-bagging
        returns:
            the number of elements that were doubled
        '''    
        benign_filter_arr = self.tumor_class_vector == 0
        print ('shapes {} {} {}'.format(self.tumor_class_vector.shape, benign_filter_arr.shape, self.features.shape))
        y_to_append = self.tumor_class_vector[benign_filter_arr] 
        X_to_append = self.features[benign_filter_arr, :, :, :]
        print ('shapes X y to append {} {} '.format(X_to_append.shape, y_to_append.shape))
        self.tumor_class_vector = np.concatenate((self.tumor_class_vector, y_to_append), axis = 0)
        self.features = np.vstack((self.features, X_to_append))
        print ('post append shapes {} {} {}'.format(self.tumor_class_vector.shape, benign_filter_arr.shape, self.features.shape))

        # also fix self.group_list and self.filenames

        fn_to_append = compress(self.images_filename_list, benign_filter_arr) 
        self.images_filename_list.extend(fn_to_append)

        grp_to_append = compress(self.group_list, benign_filter_arr)
        self.group_list.extend(grp_to_append)

        return len(y_to_append)


    def get_image_index(self, filename):
        '''
        Arguments:
            filename without path
        returns 
            int index to be used as an address for the images in self.images_list
        Used before calling transform()
        '''
 
        idx = self.images_filename_list.index(filename)    
        print ('found fn {} at index [{}]'.format(filename, idx))

        return idx
        
    def get_one_of_each(self, tumor_class= None):
        '''
        For image analysis, let's have set of 8 slides to look at, one of each tumor type
        Arguments:
            tumor_class(optional) = "B" or "M" to get 4 (one of each tumor_type)
        returns 
            list of images; one of each tumor_type (8 expected)
        '''
        img8_lst = []
        img_dict = {}
        # look through images_attributes for first instance of each tumor type
        if tumor_class == 'M':
            req_tumor_types = self.malig_tumor_types
        elif tumor_class == 'B':
            req_tumor_types = self.benign_tumor_types
        else:    
            req_tumor_types = self.tumor_types
          
        for i, ttype in enumerate (req_tumor_types):
            print ('getting sample of type {}'.format(ttype))
            # list of a generator expression (which does same thing as a list comprehension, fancy!)
            found_keys = list(k for k, v in self.images_attributes.items() if ttype == v['tumor_type'])
            fn = found_keys[0]

            idx = self.get_image_index(fn)
            img = self.get_image(idx)

            img8_lst.append(img)
            img_dict[fn] = img

        #print('returning {} sample images in list shape {}'.format(len(img8_lst), img8_lst[0].shape))
   
        return img_dict

if __name__ == '__main__':

    # include this file in bc.py
    '''
    assumes folder structure like
    data
    - BreaKHis_v1 with unzipped contents
        - histology_slides
            - breast  (more subdirs and files)

    '''




