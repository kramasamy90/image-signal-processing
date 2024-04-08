import numpy as np

class BWImage:
    '''
    This class stores an image in  an np array and provide methods to 
    access the image and other information about the image.
    '''

    # Class variables.
    blank_value = 255


    def __init__(self, im = None):
        '''
            Class initializer.

            Usage Examples:
                image = BWImage(im)
                image = BWImage(cv2.imread(path))

            Args:
                im -> (numpy.ndarray) of dimension (x, y, 3)

            Returns:
                Does not return anything.
        '''

        # im holds the np array of the pixel values.
        self.im = im

        # Some standard colors.
        self.blank_pixel = [self.blank_value for i in range(3)]
    
    def make_blank(self, shape=None):
        '''
            Converts im to a blank image.
            Usage Examples:
                image.make_blank()
                image.make_blank((256, 512))
                image.make_blank((256, 512, 3))
                image.make_blank(source.shape)

            Args:
                shape -> A tuple with 2D image dimensions.
                            If shape = None, then the shape is set 
                                    as shape of im.

            Returns:
                Returns nothing.
        '''
        if (shape == None):
            shape = self.im.shape
        self.im = np.full((shape[0], shape[1], 3), self.blank_pixel)
    
    def get_image(self):
        '''
            Return im.
            Usage Examples:
                image_array = image.get_image()
                plt.imshow(image.get_image()

            Args:
                No argument.

            Returns:
                Numpy array of dimension (x, y, 3) with pixel values.
        '''
        return self.im
    
    def shape(self):
        '''
            Returns:
                (tuple of length 2) im.shape
        '''
        return (self.im.shape[0], self.im.shape[1])

    def __getitem__(self, index):
        '''
            Overload the index operator [] to retrieve values.

            Usage: 
                image[i, j] -> In general, returns the pixel value at i, j. 
                            But, returns value of blank pixel if i or j 
                            is out of bound.

            Args:
                index: (int, int)
            
            Returns:
                (int) Pixel value.
        '''

        # Unpack the index.
        i, j = index

        # Bilinear interpolation is implemented within the operator overloading.
        # If i and j are integer, then the input is interpreted as a pixel.
        # If they are not integer, it is interpreted as a postion (not pixel).
        # For now self.interpolation is set as bilinear interpolation.
        # In future this could be changed to any other interpolation in the __init__.
        if(not isinstance(i, int)):
            if(not isinstance(i, float) and not isinstance(j, float)):
                return self.im[i, j, 0]
            else:
                return self.bilinear_interpolation([i, j])

        # Use blank pixel without any value.
        # like pixels out of the target image bound.
        if(i < 0 or j < 0 or i >= self.im.shape[0] or j >= self.im.shape[1]):
            return self.blank_pixel[0]
        
        return self.im[i, j][0]
    
    def __setitem__(self, index, value):
        '''
            Overload the index operator to assign values.

            Usage Example:
                image[i, j] = 255
            
            Args:
                index -> (int , int) i, j
                value -> (int) Pixel value.
            
            Returns:
                Returns nothing.
        '''

        i, j = index
        self.im[i, j] = [value, value, value]

    
    def bilinear_interpolation(self, source_pos):
        '''
        Gives the pixel value at the target_coordinate by bilinear interpolation
            source_pos: Position in the image where the target pixel maps to. Could be a fraction.
        
        Usage -> Used internally, like a private method.

        Args:
            source_pos -> (List) [i, j]
        
        Return:
            (int) pixel value at i, j.
        '''

        x_size, y_size = self.im.shape[0], self.im.shape[1]
        xs_, ys_ = source_pos

        # There are four pixels near any position.
        # xs, ys, is the top left pixel.
        xs, ys = int(np.floor(xs_)), int(np.floor(ys_))

        # a, b -> x and y distance of the positin from (xs, ys).
        a = xs_ - xs
        b = ys_ - ys

        # Return the value of the image at a position as 
        # bilinear interpolation of the 4 nearest pixels.
        return \
            (1-a) * (1-b) * self[xs, ys] + \
            (1- a) * b * self[xs, ys + 1] + \
            a * (1-b) * self[xs + 1, ys] + \
            a * b * self[xs + 1, ys + 1]
