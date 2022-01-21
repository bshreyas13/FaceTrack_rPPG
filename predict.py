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
    data = TFRReader(batch_size, timesteps)


    signal= model.predict(data,batch_size = 2)
    print(len(signal))