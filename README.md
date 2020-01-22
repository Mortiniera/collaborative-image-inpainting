# Collaborative Sampling for Image Inpainting

## Author

- Thevie Mortiniera

## Inpainting on FASHION-MNIST

### Visual Results

<p float="left">
  <img src="../master/metrics/images_0-3.png" width="32%"> 
  <img src="../master/metrics/images_4-7.png" width="32%"> 
  <img src="../master/metrics/images_16_19.png" width="32%">
</p>

Sampling results for image inpainting by targeting the corrupted region. (Top) Input data with masked region(second row) Output of the generator surrounded by itscontext image (third row) Heatmap highlighting visualdifferences between the inpainted output of thegenerator in the 2nd row and the refined results in thefourth row. The closer to the red, the higher thedifferences (fourth row) refined samples after applyingcollaborative sampling and discriminator shapingsurrounded by its context image (bottom) Originalimages.

### Quantitative Results (PSNR : Peak-signal-to-noise-ratio) 

From left to right :

```

| Method | Im1 | Im2 | Im3 | Im4 | Im4 | Im5 | Im6 | Im7 | Im8 | Im9 | Im10 | Im11 | Im12 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Semantic Image Inpainting | 13.31 | 21.07 | 25.54 | 29.93 | 28.39 | 28.19 | 28.94 | 25.25 | 27.07 | 34.80 | 20.07 | 34.63 |
| Collaborative Image Inpainting | 14.65 | 23.84 | 28.63 | 23.43 | 24.53 | 26.77 | 29.22 | 26.57 | 28.18 | 38.27 | 20.10 | 35.97 |
```




## Documentation

### Download dataset

The following command allow to download the FASHION-MNIST data set and create the corresponding folders as in the directory hierarchy below.

``` python download.py fashion_mnist ```

### Directory hierarchy

If using an already pretrained DCGAN model, its root folder should be placed at the same hierarchy level as the collaborative-image-inpainting and Data folders, e.g below, with a pretrained model from fashion_mnist.

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
│   ├── fashion_mnist
│   │   ├── train
│   │   └── val
│   fashion_mnist
│   ├── images
│   ├── inpaint
│   ├── logs
│   ├── model
│   ├── sample
│   ├── vectors
```

### Run the app

* First of all, one need to train a DCGAN model on the choosen dataset.
* Then, use the pretrained DCGAN model to compute, offline, the closest latent vectors encodings of the images
in the training set to be used during the collaborative sampling scheme. 
* Finally, use the pretrained DCGAN model along with the saved latent vectors to experiment and compare the collaborative image inpainting scheme against the previous semantic image inpainting method.


#### Training

As an example, use the following command to train the DCGAN model. Other arguments are available in the ```main.py``` file to use different parameters.

``` python main.py --is_train=true --iters=25000 --dataset=fashion_mnist```

#### Offline computing of closest latent vectors encoding

``` python inpaint_main.py --offline=true --dataset=fashion_mnist```

#### Experiment between the collaborative scheme and original inpainting method. 

Two modes are available between [inpaint | standard] to choose between collaborative image inpainting and standard collaborative sampling scheme.  Other arguments are available in the inpaint_main.py file to use different parameters.

``` python inpaint_main.py --mode=inpaint --dataset=fashion_mnist```


### Attribution / Thanks

* This project borrowed some readme formatting and code from [ChengBinJin](https://github.com/ChengBinJin/semantic-image-inpainting), mostly regarding the inpainting process.
* Most of the collaborative sampling scheme was borrowed from [vita-epfl](https://github.com/vita-epfl/collaborative-gan-sampling)
