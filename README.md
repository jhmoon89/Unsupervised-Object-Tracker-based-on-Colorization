# Unsupervised-Object-Tracker-based-on-Colorization
Tensorflow implementation of Unsupervised Object Tracker based on Colorization

Reference: Vondrick, Carl, et al. "Tracking emerges by colorizing videos." arXiv preprint arXiv:1806.09594 (2018).

# Network Architectures
![intro](https://user-images.githubusercontent.com/25393387/49177528-748f1200-f31b-11e8-9033-28d69098cd87.png)
We put several reference frames(semantic label) and one target frame(gray image) as input, which passes through 2D and 3D Convolutional Neural Networks(CNNs).

Output is predicted semantic image of target frame.

# Results
![horse-jump](https://user-images.githubusercontent.com/25393387/49187553-dd838380-f335-11e8-82b4-27f28aa8cf56.gif)
![lady-running](https://user-images.githubusercontent.com/25393387/49187916-01939480-f337-11e8-9873-6aa4b4af4cbd.gif)
