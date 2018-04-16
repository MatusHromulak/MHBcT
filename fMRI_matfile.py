#read in .mat files from the StarPlus study and convert them to .npy file containing numpy array formated data

from __future__ import print_function
import scipy.io as spio
import numpy as np

def process_mat_file(mat_file):    
    #load .mat data file
    mat_data = spio.loadmat(mat_file)
    #name = str(mat_data['meta']['subject'][0][0][0])
    
    #get sizing information
    num_trials = int(mat_data['meta']['ntrials'][0])
    dimx = int(mat_data['meta']['dimx'][0,0])
    dimy = int(mat_data['meta']['dimy'][0,0])
    dimz = int(mat_data['meta']['dimz'][0,0])
    
    #retrieve the first picture/sentence (4 images) and the first rest period (+ another 4 images) from each trial
    img_count = 8
    
    #voxel count in trial
    vox_count = int(mat_data['meta']['nvoxels'][0,0])
    
    #identify stimulus per trial and load non-erroneous data
    img_lbl = list() #prepare label list
    img_data = list() #prepare image list
    for i in range(0, num_trials): #get data from all trials
        cond = mat_data['info']['cond'][0,i].astype('int') #read in current condition
        stim = mat_data['info']['firstStimulus'][0,i].astype('str') #read in current stimulus
        
        if (cond == 0): #ignore segment
            continue #no data or label load
            
        elif (cond == 1): #rest interval
            img_lbl.extend([0,0,0,0,0,0,0,0]) #no stimulus in all images
            
            raw_data = mat_data['data'][i] #get raw data for said trial
            coords = mat_data['meta']['colToCoord'] #get coordinates for column to 3D image transformation
            for j in range(0, img_count): #get first 8 images
                image_column = raw_data[0][j].astype('float') #retrieve single image data
                image_3d = np.zeros((dimx,dimy,dimz)) #prepare 3D zero array
                for k in range(0, vox_count): #transform column shaped data to 3D data
                    x, y, z = coords[0][0][k] #get voxel coordinates
                    image_3d[x-1][y-1][z-1] = image_column[k] #put voxel value into 3D image
                img_data.append(image_3d) #push 3D data on stack
        
        elif (cond == 2 or cond == 3): #sentence or picture
            if (stim == 'P'): 
                img_lbl.extend([1,1,1,1,0,0,0,0]) #picture stimulus in first 4 images, no stimulus in last 4 images
            elif (stim == 'S'): 
                img_lbl.extend([2,2,2,2,0,0,0,0]) #sentence stimulus in first 4 images, no stimulus in last 4 images
            
            raw_data = mat_data['data'][i] #get raw data for said trial
            coords = mat_data['meta']['colToCoord'] #get coordinates for column to 3D image transformation
            for j in range(0, img_count): #get first 8 images
                image_column = raw_data[0][j].astype('float') #retrieve single image data
                image_3d = np.zeros((dimx,dimy,dimz)) #prepare 3D zero array
                for k in range(0, vox_count): #transform column shaped data to 3D data
                    x, y, z = coords[0][0][k] #get voxel coordinates
                    image_3d[x-1][y-1][z-1] = image_column[k] #put voxel value into 3D image
                img_data.append(image_3d) #push 3D data on stack
    
    #array of labels and images
    images = np.array(img_data)
    labels = np.array(img_lbl)
    
    print("Processed: ", mat_file)
    return images, labels

def main():
    
    #load training data and labels
    mat_files = ['fMRI_data/data-starplus-04799-v7.mat', 'fMRI_data/data-starplus-04847-v7.mat', 
                'fMRI_data/data-starplus-05675-v7.mat', 'fMRI_data/data-starplus-05680-v7.mat', 'fMRI_data/data-starplus-05710-v7.mat']
    
    temp_images = list()
    temp_labels = list()
    for file in mat_files:
        images, labels = process_mat_file(file)
        temp_images.extend(images)
        temp_labels.extend(labels)
        
    #convert to numpy array
    train_images = np.array(temp_images)
    train_labels = np.array(temp_labels)
    
    #load test data and labels
    mat_file = 'fMRI_data/data-starplus-04820-v7.mat'
    test_images, test_labels = process_mat_file(mat_file)

    #write processed arrays to output files
    out_file = open('fMRI_data/fMRI_test_images.npy', 'wb+')
    np.save(out_file, test_images)
    out_file.close()
    
    out_file = open('fMRI_data/fMRI_test_labels.npy', 'wb+')
    np.save(out_file, test_labels)
    out_file.close()
    
    out_file = open('fMRI_data/fMRI_train_images.npy', 'wb+')
    np.save(out_file, train_images)
    out_file.close()
    
    out_file = open('fMRI_data/fMRI_train_labels.npy', 'wb+')
    np.save(out_file, train_labels)
    out_file.close()

#run program
if __name__ == "__main__":
    main()
