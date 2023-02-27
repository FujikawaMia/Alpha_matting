# Shared Matting Python Implementation

## Original paper

>[Shared Sampling for Real-Time Alpha Matting](http://inf.ufrgs.br/~eslgastal/SharedMatting/)
  Eduardo S. L. Gastal and Manuel M. Oliveira
  Computer Graphics Forum. Volume 29 (2010), Number 2.
  Proceedings of Eurographics 2010, pp. 575-584.
  
## Usage for test images

Ensure that you possess a CUDA-capable GPU.

- Prepare an image along with its corresponding trimap, both of which should be in PNG format;
  - Ensure that both the image and its trimap are located in the same directory;
  - Ensure that the names of the image and its trimap adhere to the format, which is "image.png" and "image_trimap.png", respectively;
- Launch the command "python3 http://demo.py" and input the path to your image when prompted;

The results will be saved into the same directory as your own image.

