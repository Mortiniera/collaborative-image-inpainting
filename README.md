# Collaborative Sampling for Image Inpainting

## Authors 

- Thevie Mortiniera
- Yuejiang Liu (Supervisor)
- Alexandre Alahi (Professor)


## Inpainting

Results on FASHION_MNIST

<p float="left">
  <img src="../master/assets/inpaint1.png" width="32%"> 
  <img src="../master/assets/inpaint2.png" width="32%"> 
  <img src="../master/assets/inpaint4.png" width="32%">
</p>

Sampling results for image inpainting by targeting the corrupted region. (Top) Input data with masked region(second row) Output of the generator surrounded by itscontext image (third row) Heatmap highlighting visualdifferences between the inpainted output of thegenerator in the 2nd row and the refined results in thefourth row. The closer to the red, the higher thedifferences (fourth row) refined samples after applyingcollaborative sampling and discriminator shapingsurrounded by its context image (bottom) Originalimages.

## Documentation

### Download dataset

The following command allow to download the FASHION-MNIST data set or CelebA and create the corresponding folders as in the directory hierarchy below.

``` python download.py [fashion_mnist | celebA] ```

### Directory hierarchy

```
.
│   collaborative-image-inpainting
│   ├── src
│   │   ├── collaborator.py
│   │   ├── dataset.py
│   │   ├── dcgan.py
│   │   ├── download.py
│   │   ├── inpaint_main.py
│   │   ├── inpaint_model.py
│   │   ├── inpaint_solver.py
│   │   ├── main.py
│   │   ├── mask_generator.py
│   │   ├── ops.py
│   │   ├── policy.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   │   └── utils_2.py
│   Data
│   ├── celebA
│   │   ├── train
│   │   └── val
│   ├── fashion_mnist
│   │   ├── train
│   │   └── val

```

### Run the app

* First of all, one need to train a DCGAN model on the choosen dataset.
* Then, use the pretrained DCGAN model to compute offline the closest latent vectors encodings of the images
in the training set to be used during the collaborative sampling scheme. 
* Finally, use the pretrained DCGAN model along with the saved latent vectors to experiment and compare the collaborative image inpainting scheme against the previous semantic image inpainting method.


#### Training

As an example, use the following command to train the DCGAN model. Other arguments are available in the ```main.py``` file to use different parameters.

``` python main.py --is_train=true --iters=25000 --dataset=fashion_mnist```

### Offline computing of closest latent vectors encoding

``` python inpaint_main.py --offline=true --dataset=fashion_mnist```

### Experiment between the collaborative scheme and oginal inpainting method. 

Two modes are available between [inpaint | standard] to choose between collaborative image inpainting and standard collaborative sampling scheme.

``` python inpaint_main.py --mode=inpaint --dataset=fashion_mnist```


### Attribution / Thanks

* This project borrowed some code from [ChengBinJin](https://github.com/ChengBinJin/semantic-image-inpainting), mostly regarding the inpainting process and from [vita-epfl](https://github.com/vita-epfl/collaborative-gan-sampling) regarding the collaborative sampling scheme.
