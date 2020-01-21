

# %% Generate tasks
# objects: Define object as a square area with maybe a couple of random touching pixels. 
# Define location, object size, then create object randomly. Hollow shapes, vs solids. 
# Create tasks to move objects around, move objects until hit wall, untill hit other objects, 
# create a slowly morphing shape: -size increase -weird lines coming out of its sides. 
# Shoot lasers.
# Flip patterns
# Flip objects.
# flip the biggest object. 
def create_object( size, color, style):
    ''' Creates a small canvas with size specified containing an object of the selected style.
        INputs:
            Size: size of the object in pixels, a tuple (height,width)
            Color: int, 1-9, the predominent color to use, 1 through 9
            Style: String, One of the following styles:
                'SolidRect': a rectangular shape filled solidly with pixels. 
                'HollowRect': a rectangular shape outlined with pixels at the border
                'Random': a random rectuangular patch
                'shabby'  a rectangular shape filled solidly with pixels, with some protruding pixels
    '''
    canvas = np.zeros(size)

    if style is 'SolidRect':
        canvas = np.zeros(size)+ color
    elif 'HollowRect':
        canvas[[0, -1], :] = color # ?Does this work?

        
    return (canvas of size size)

def flip_patch(canvas, direction):
    if direction is 'horizontal':
        flipped_canvas = canvas[:, ::-1]
    elif direction is 'vertical':
        flipped_canvas = canvas[::-1, :]

    return flipped_canvas

def create_canvas(canvas_size):
    ''' Creates an empty canvas. It would be padded to 30x30 with the padding codeword 10
        Inputs:
            Canvas_size: A tuple of (x,y) specifying the height and width, in pixels
        Output:
            An empty canvas of zeros, padded by 10s.
    '''

    padded_canvas = np.ones( (30,30)) * 10. #create canvas filled with masked value 10
    zeroed_canvas = np.zeros(canvas_size) #this is the actual canvas area, empty with zeros.
    canvas = padded_canvas[:canvas_size[0], :canvas_size[0]] = zeroed_canvas #put the zeros on the upper left

    return canvas

