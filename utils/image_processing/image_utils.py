from google_images_download import google_images_download
import os
import PIL


def download_images(all_image_keywords):

    response = google_images_download.googleimagesdownload()
    
    for keyword_index in range(len(all_image_keywords)):
        keyword = all_image_keywords[keyword_index]
        print("downloading images of {}".format(keyword))

        response.download({'keywords': '{}'.format(keyword),
                           'exact_size': '500,500',
                           'output_directory': './training_data',
                           'no_directory': True,
                           'format': 'jpg'
                           })


def remove_download_error_images(path):
    count = 0
    for filename in os.listdir(path):
        if filename[-4:] != ".jpg":
            os.remove(os.path.join(path, filename))
            count += 1
            continue
        try:
            im = PIL.Image.open(os.path.join(path, filename))
        except IOError:
            os.remove(os.path.join(path, filename))
            count += 1
    print(count, 'erroneous images in training data deleted.')

def load_images(path):
    images = []
    for filename in os.listdir(path):
        full_path = os.path.join(path, filename)
        images.append(PIL.Image.open(full_path))

    return images


def concat_images(images, direction='horizontal',
                  bg_color=(255,255,255), aligment='center'):
    """
    Appends images in horizontal/vertical direction.

    Args:
        images: List of PIL images
        direction: direction of concatenation, 'horizontal' or 'vertical'
        bg_color: Background color (default: white)
        aligment: alignment mode if images need padding;
           'left', 'right', 'top', 'bottom', or 'center'

    Returns:
        Concatenated image as a new PIL image object.
    """
    widths, heights = zip(*(i.size for i in images))

    if direction=='horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = PIL.Image.new('RGB', (new_width, new_height), color=bg_color)


    offset = 0
    for im in images:
        if direction=='horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1])/2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0])/2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im