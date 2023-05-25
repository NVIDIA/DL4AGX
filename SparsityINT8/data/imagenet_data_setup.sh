export IMAGENET_HOME=/media/Data/imagenet_data
export TRAIN_SUBDIR=train
export VAL_SUBDIR=val
echo "Formatting raw data in ${IMAGENET_HOME}..."

# Setup folders
mkdir -p $IMAGENET_HOME/$VAL_SUBDIR
mkdir -p $IMAGENET_HOME/$TRAIN_SUBDIR

# Extract validation and training
echo "  [1/3] Extracting validation and training data"
tar xf $IMAGENET_HOME/ILSVRC2012_img_val.tar -C $IMAGENET_HOME/$VAL_SUBDIR
tar xf $IMAGENET_HOME/ILSVRC2012_img_train.tar -C $IMAGENET_HOME/$TRAIN_SUBDIR

# Extract and then delete individual training tar files This can be pasted
# directly into a bash command-line or create a file and execute.
echo "  [2/3] Extracting train subfolders"
cd $IMAGENET_HOME/$TRAIN_SUBDIR

for f in *.tar; do
  echo $f
  d=`basename $f .tar`
  mkdir $d
  tar -xf $f -C $d
  wait
done

cd $IMAGENET_HOME  # Move back to the base folder

# [Optional] Delete tar files if desired as they are not needed
rm $IMAGENET_HOME/$TRAIN_SUBDIR/*.tar

## ###### Modification: Re-format val data to split images into their respective folders #############
echo "  [3/3] Re-formatting validation data"
cd $IMAGENET_HOME/$VAL_SUBDIR
wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
chmod +x valprep.sh && ./valprep.sh
rm valprep.sh
## ###################################################################################################
