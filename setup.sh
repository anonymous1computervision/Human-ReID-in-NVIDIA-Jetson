mkdir models
mkdir models/id
mkdir models/reid
CHECKPOINT_DIR=pretrained
mkdir ${CHECKPOINT_DIR}
wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
tar -xvf resnet_v1_50_2016_08_28.tar.gz
mv resnet_v1_50.ckpt ${CHECKPOINT_DIR}
rm resnet_v1_50_2016_08_28.tar.gz
