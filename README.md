# PyTorch-deeplabv2

PyTorch implementation of [*DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*](https://arxiv.org/abs/1606.00915),  [ENet-training](https://github.com/e-lab/ENet-training) created by huachen_yu.

This implementation has been tested on the CamVid and Cityscapes datasets. Currently, a pre-trained version of the model trained in CamVid and Cityscapes is available [here](https://github.com/davidtvs/PyTorch-ENet/tree/master/save).


|                               Dataset                                | Classes <sup>1</sup> | Input resolution | Batch size | iteractions |   Mean IoU (%)    | GPU memory (GiB) | Training time (hours)<sup>2</sup> |
| :------------------------------------------------------------------: | :------------------: | :--------------: | :--------: |  :--------: | :---------------: | :--------------: | :-------------------------------: |
|          [Cityscapes](https://www.cityscapes-dataset.com/)           |          19          |     512x256      |     1      |    100000   |      47.24        |       11         |               4.5                 |

<sup>1</sup> When referring to the number of classes, the void/unlabeled class is always excluded.<br/>
<sup>2</sup> Just for reference since changes in implementation, datasets, and hardware can lead to very different results. Reference hardware: Nvidia GTX 1080Ti. You can also train for 100000 iteractions to get the best mIoU for the validation set.<br/>
<sup>3</sup> Validation set from the dataset used to validate the model performance.<br/>
<sup>4</sup> Test set can be used to gain some color label maps.


## Installation

1. Anaconda will be recommended.
2. Set up a virtual environment for pytorch with the python=3.7.
3. Install the pytorch followed the https://pytorch.org/get-started/locally/.
4. Install matplotlib (pip install matplotlib)


## Usage

Run [``main.py``](https://github.com/davidtvs/PyTorch-ENet/blob/master/main.py), the main script file used for training and/or testing the model. The following options are supported:

```
python main.py [--learning-rate LEARNING_RATE] [--lr-decay LR_DECAY]
               [--lr-decay-epochs LR_DECAY_EPOCHS]
               [--weight-decay WEIGHT_DECAY] [--dataset cityscapes]
               [--dataset-dir DATASET_DIR] [--height HEIGHT] [--width WIDTH]
               [--weighing {deeplabv2,none}] [--with-unlabeled]
               [--workers WORKERS] [--print-step] [--imshow-batch]
               [--device DEVICE] [--name NAME] [--save-dir SAVE_DIR]
```

For help on the optional arguments run: ``python main.py -h``


### Examples: Training
### directly train
```
python main.py 
```
### train with settings
```
python main.py --save-dir save/folder/ --name model_name --device cuda(cpu)
```

### Examples: Testing

### directly test
```
python pred.py 
```
### test with settings
```
python pred.py --test_img test-img/ --test_out outputs --device cuda(cpu)
```

Some codes of this project are borrowed from the [PyTorch-ENet](https://github.com/davidtvs/PyTorch-ENet) 

