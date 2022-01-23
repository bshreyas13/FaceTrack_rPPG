from tensorflow.keras.models import load_model
import argparse as ap
import os
import numpy as np
import pathlib
from tqdm import tqdm
from modules.preprocessor import Preprocessor

def flatten(t):
    return [item for sublist in t for item in sublist]

if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-m","--model", required = True , help = "FaceTrack_rPPG or DeepPhys")
    parser.add_argument("-m_path","--model_path", required = True , help = "Path to saved model")
    parser.add_argument("-ap","--appearance", required = True , help = "Path to Apperance Stream Data")
    parser.add_argument("-mo","--motion", required = True , help = "Path to Motion Stream Data")
    parser.add_argument("-lp","--labels", required = True , help = "Path to  Label by video")   
    parser.add_argument("-ims", "--image_size", required = False , help = "Desired input img size.")

    args = vars(parser.parse_args())
     
    ## Get args
    m_path = args["model_path"]
    roi_path = args["motion"]
    nd_path = args["appearance"]
    model_name = args["model"]
    img_size = args["image_size"]
    if img_size == None:
            img_size = "36X36X3"
    else:
            img_size = args["image_size"]
    
    img_size = [int(dim) for dim in img_size.split('X')]
    img_size = (img_size[0],img_size[1],img_size[2])

    split = 'Prediction'
    
    p = Preprocessor()
    model = load_model(m_path)

    if model_name == "FaceTrack_rPPG":
        from modules.tfrecordhandler_FTR import TFRWriter
        from modules.tfrecordhandler_FTR import TFRReader
    else :
        from modules.tfrecordhandler_DP import TFRWriter
        from modules.tfrecordhandler_DP import TFRReader

    txt_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Dataset' , 'Txt', model_name, 'Test'))

    tfrecord_path= pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Prediction' , 'TFRecords',model_name))
    tfrecord_path.mkdir(parents=True,exist_ok=True)

    tfwrite = TFRWriter(img_size)
    
    file_list= tfwrite.makeShuffledDict(txt_path)

    ## Batching is done on Video level ==> use small batch size
    batch_size = 10    
    # if batch_size > len(in_data):
    #     batch_size = len(in_data)
    timesteps=5

    tfwrite.writePredTFRecords(roi_path,nd_path, txt_path, tfrecord_path, file_list, batch_size,split,timesteps )

    tfwrite = TFRWriter(img_size)
    videos = os.listdir(os.path.join(tfrecord_path,split))
    for vid in tqdm(videos):
    	tfrpath = os.path.join(tfrecord_path,split,vid)
    	save_path = pathlib.Path(os.path.join(os.path.dirname(os.getcwd()),'Prediction' , 'Signals',model_name,vid+'.dat'))
    	tfrecord_path.mkdir(parents=True,exist_ok=True)
    	# make a dataset iterator
    	data = TFRReader(batch_size, timesteps)
    	signal = []
    	batch = data.getBatch(tfrpath, 1, False, 0)
    	for (x_l,x_r),(y) in batch.take(3000/batch_size):
    		pred= model.predict([x_l,x_r],batch_size = 10)
    		signal.append(pred)
    	signal =flatten(signal)
    	signal = np.array(signal)
    	#print(signal.shape) 
    	p.saveData(save_path,signal)
    print('Saved Predicted Signals for all test videos')