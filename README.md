## Convolutional Neural Network
This is a convolutional Neural Network used to identify the digital signal made by hand.
### Build
If you use pipenv, just run `pipenv install`. Or install `numpy`, `h5py`, `scipy`, `pillow` by yourself
### Config
1. Put your image with name "data_xx"(xx is the number that show the order of the images) in the directory
2. Assignment the path of the directory to "img_path" in *config.py*
3. change the setting of the neural network in *config.py*
> if you keep "img_path" value None, the default path is `./image`
### Start
Just run `python main.py` and then it will output the array of result (the order as same as the images)
### structure
![](model.png)
### More
- It sometimes report an error the **RuntimeWarning: overflow encountered in exp**. Just ignore it
- Sadly, though the test accuracy is not bad, it will not perform well on our own image
- For further information, you can read [my blog](https://masterwangzx.com/2019/01/05/cnn/)