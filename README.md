# Edge detection

## Methodology.
1. Moving Average arcross 5 pictures
2. Convolve with a Gaussian 5x5 kernel
3. Use Sobel or Canny 

## Note
1. Retina 2 contains original image
2. retina2withmovingaverage contains all images after moving average
3. It doesn't run for all pictures, to save time/space
4. Only use roberts.py or sobel.py, they are essentailly the same thing but with different gaussian operators

