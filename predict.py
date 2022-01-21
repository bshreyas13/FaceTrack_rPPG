from tensorflow.keras.models import load_model
import argparse as ap
import os

if __name__ == '__main__':
    
    parser = ap.ArgumentParser()
    parser.add_argument("-mp","--model_path", required = True , help = "Path to metrics file")
    parser.add_argument("-mp","--model_name", required = True , help = "DeepPhys/FaceTrack_rPPG")
    args = vars(parser.parse_args())
     
    ## Get args
    m_path = args["model_path"]
	train_ds,val_ds,test_ds = prep.getDatasets(model_name,appearance_path,motion_path,labels_path,txt_files_paths,tfrecord_path, batch_size=batch_size, timesteps=timesteps, subset=subset, subset_read = subset_train, val_split = val_split , test_split =test_split,write_txt_files=wtxt, create_tfrecord=wtfr,rot=1)
    model = load_model(m_path)

    if model_name == "FaceTrack_rPPG":
        from modules.tfrecordhandler_FTR import TFRWriter
        from modules.tfrecordhandler_FTR import TFRReader
    elif model_name == "DeepPhys" :
        from modules.tfrecordhandler_DP import TFRWriter
        from modules.tfrecordhandler_DP import TFRReader

    #tfwrite = TFRWriter()

    #tfwrite.writeTFRecords(roi_path,nd_path, txt_files_path, tfrecord_path, file_list, batch_size,split,timesteps )
    tfrpath = os.path.join(tfrecord_path.as_posix(),split)
    # make a dataset iterator
    data = TFRReader(batch_size, timesteps)

    model.predict