from tensorflow.keras.models import load_model
import argparse as ap
import os
import numpy as np
from modules.preprocessor import Preprocessor

def flatten(t):
    return [item for sublist in t for item in sublist]

if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-mp","--model_path", required = True , help = "Path to metrics file")
    parser.add_argument("-tfr_path","--tfrecord_path", required = True , help = "Path to metrics file")
    parser.add_argument("-mn","--model_name", required = True , help = "DeepPhys/FaceTrack_rPPG")

    args = vars(parser.parse_args())
     
    ## Get args
    m_path = args["model_path"]
    tfrecord_path =args["tfrecord_path"]
    model_name = args["model_name"]
    split = 'example'
    save_path = '../test.dat'
    p = Preprocessor()
    model = load_model(m_path)
    if model_name == "FaceTrack_rPPG":
        from modules.tfrecordhandler_FTR import TFRWriter
        from modules.tfrecordhandler_FTR import TFRReader
    elif model_name == "DeepPhys" :
        from modules.tfrecordhandler_DP import TFRWriter
        from modules.tfrecordhandler_DP import TFRReader
    #tfwrite = TFRWriter()
    tfrpath = os.path.join(tfrecord_path,split)
    # make a dataset iterator
    data = TFRReader(10, 5)
    signal = []
    batch = data.getBatch(tfrpath, 1, False, 0)
    for (x_l,x_r),(y) in batch.take(300):
    	pred= model.predict([x_l,x_r],batch_size = 10)
    	signal.append(pred)
    signal =flatten(signal)
    signal = np.array(signal)
    print(signal.shape) 
    p.saveData(save_path,signal)