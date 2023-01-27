SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`



if [ ! -d ${SCRIPTPATH}/../datasets ]; then
    mkdir ${SCRIPTPATH}/../datasets
fi

if [ -z $1 ]; then
    echo "please specify datasets url"
else
    cd ${SCRIPTPATH}/../datasets
    wget https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset/download?datasetVersionNumber=1 --no-check-certificate
fi

# unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
# rm shapenetcore_partanno_segmentation_benchmark_v0.zip
cd -
