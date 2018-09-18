import tensorflow as tf

MEAN = tf.constant([123.68, 116.78, 103.94], dtype=tf.float32) # IMAGENET

def preprocess_image(image, output_height, output_width, is_training=False):
    # Get the input Dimensions
    img_resize_crop = tf.image.resize_image_with_crop_or_pad(image, output_height, output_width)

    # Subtract the imagenet mean (mean over all imagenet images)
    imgnet_mean = tf.reshape(MEAN, [1, 1, 3])
    img_float = tf.to_float(img_resize_crop)
    img_standardized = tf.subtract(img_float, imgnet_mean)
    return img_standardized
