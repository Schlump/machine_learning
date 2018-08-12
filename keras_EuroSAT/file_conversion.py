
def convert_2_PNG(dataDir):
    files = []
    counter = 0
    parDir = os.path.abspath(os.path.join(dataDir, os.pardir))
    for subdir in os.listdir(dataDir):
        createDir = "mkdir -p "+parDir+"/Train_Data/"+subdir
        subprocess.call(createDir, shell=True)
        for file in os.listdir(dataDir+subdir):
            if file.endswith(".tif"):
                outFile = parDir+'/Train_Data/'+subdir+'/'+os.path.splitext(os.path.basename(file))[0]+'.png'
                inFile = dataDir + subdir +'/' + file
                counter += 1
                subprocess.call(['gdal_translate -of PNG -b 2 -b 3 -b 4 ' + inFile +' '+ outFile], shell=True)
                if counter % 5000 == 0:
                    print ('Files processed: ', counter)
                if counter % 500 == 0:, end=""
                    print ('.', end="")
            
def split_Train_Test(dataDir,num):
    path = os.path.dirname(os.path.dirname(dataDir))
    shuffleFiles = "cd "+dataDir+" && for d in ./*/; do ( mkdir -p "+path+"/Test_Data/$d && cd $d && shuf -zen"+str(num)+" *.png | xargs -0 mv -t "+path+"/Test_Data/$d/ ); done"
    subprocess.call(shuffleFiles, shell=True)


def read_data_from_dir(dataDir,extension):
    """ Read a stack of images located in subdirectories into a dask array
        returning X (array of data) and y (array of labels)
    """
    X = np.concatenate([imread(dataDir+subdir+'/*.'+extension).compute() for subdir in os.listdir(dataDir)]) 

    filesdict = {}

    for subdir in sorted(os.listdir(dataDir)):
        files = next(os.walk(dataDir+subdir))[2]
        files = len([fi for fi in files if fi.endswith("."+extension)])
        filesdict.update({subdir:files})

    if sum(filesdict.values()) != X.shape[0]:
        
        raise ValueError('Images and Labels does not Match')

    else:
        y = np.zeros([X.shape[0],1], dtype=np.uint8)
        i = 0
        imagelist = []
        for category in list(filesdict.keys()):
            z = filesdict[category]
            y[sum(imagelist):sum(imagelist)+z] = i
            imagelist.append(z)
            i += 1   

    return X,y
