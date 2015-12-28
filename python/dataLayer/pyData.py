import caffe

class PyDataLayer(caffe.Layer):
   def setup(self, bottom, top):
      #image_path = "."
      top[0].reshape(1, 3, 100, 100) # 3 channels, 100x100 are dummy values

   def reshape(self, bottom, top):
      pass

   def forward(self, bottom, top):
      #img = caffe.io.load_image(image_path)
      #new_dims = (1, img.shape[2]) + img.shape[:2]
      #top[0].reshape(*new_dims)
      #top[0].data[...] = img.copy().transpose((2,0,1))
      pass

   def backward(self, top, propagate_down, bottom):
      # This layer does not propagate gradients.
      pass
