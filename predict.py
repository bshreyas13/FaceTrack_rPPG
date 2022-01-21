from tensorflow.keras.models import load_model
import argparse as ap
import os

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

    batch = data.getBatch(tfrpath, 1, False, 0)
    for (x_l,x_r),(y) in batch.take(1):
    	signal= model.predict([x_l,x_r],batch_size = 10)
    	print(signal)