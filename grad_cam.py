import argparse
import os

import numpy as np
from PIL import Image
from matplotlib import cm
import tensorflow as tf

def _parse_args():
	parser = argparse.ArgumentParser(description='Produce Class Activation Maps (CAM) by Grad-CAM.')
	parser.add_argument('--model_path', default=None, help='a frozen model path, either pb or saved_model is supported. NOTE that you MUST assign a directory path which contains model files for save_model format')
	parser.add_argument('--print_tensors', action='store_true', help='load model then print all tensors\' name, it will do nothing else')
	parser.add_argument('--input_path', default=None, help='an image file path or a directory path that contains images (jpg/png/bmp are supported)')
	parser.add_argument('--output_path', default=None, help='a directory path for CAMs output, the default is the same as input_path')
	parser.add_argument('--input_width', type=int, default=None, help='image width of model input')
	parser.add_argument('--input_height', type=int, default=None, help='image height of model input')
	parser.add_argument('--preprocess', default=None, help='vgg or inception, the default is do nothing for preprocess')
	parser.add_argument('--class_index', type=int, default=-1, help='a class index you interested, the default is the most probable class')
	parser.add_argument('--input_tensor_name', default=None, help='input tensor\'s name (it doesn\'t matter whether suffix \':0\' is added)')
	parser.add_argument('--featuremap_tensor_name', default=None, help='feature map tensor\'s name (it doesn\'t matter whether suffix \':0\' is added)')
	parser.add_argument('--logits_tensor_name', default=None, help='logits tensor\'s name (it doesn\'t matter whether suffix \':0\' is added)')
	parser.add_argument('--prediction_tensor_name', default=None, help='(optional) prediction tensor\'s name (it doesn\'t matter whether suffix \':0\' is added)')
	parser.add_argument('--per_process_gpu_memory_fraction', type=float, default=None, help='(optional) fraction of the available GPU memory to allocate for each process')
	args = parser.parse_args()
	if args.print_tensors:
		assert args.model_path, 'Argument \'model_path\' is required.'
	else:
		assert args.input_path, 'Argument \'input_path\' is required.'
		assert args.input_width, 'Argument \'input_width\' is required.'
		assert args.input_height, 'Argument \'input_height\' is required.'
		assert args.model_path, 'Argument \'model_path\' is required.'
		assert args.input_tensor_name, 'Argument \'input_tensor_name\' is required.'
		assert args.featuremap_tensor_name, 'Argument \'featuremap_tensor_name\' is required.'
		assert args.logits_tensor_name, 'Argument \'logits_tensor_name\' is required.'
	return args

def _get_and_resize_numpy_image_batch(input_path, target_size):
	def _read_and_resize_image(file_path):
		if os.path.splitext(file_path)[1].lower() not in ['.jpg', '.png', '.bmp']:
			return None
		im = Image.open(file_path)
		if im.mode != 'RGB':
			im = im.convert('RGB')
		im = im.resize(target_size, Image.BICUBIC)
		return np.array(im)

	image_np_list = []
	if not os.path.exists(input_path):
		raise FileNotFoundError('Path \'%s\' does not exist.' % input_path)
	if os.path.isdir(input_path):
		for filename in next(os.walk(input_path))[2]:
			image_np = _read_and_resize_image(os.path.join(input_path, filename))
			if image_np is not None:
				image_np_list.append(image_np)
	elif os.path.isfile(input_path):
		image_np = _read_and_resize_image(input_path)
		if image_np is not None:
			image_np_list.append(image_np)
	else:
		raise ValueError('Path \'%s\' is not a directory or file.' % input_path)
	return np.stack(image_np_list)

def _load_model(sess, model_path):
	if os.path.isdir(model_path):
		tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
	elif os.path.splitext(model_path)[1].lower() == '.pb':
		with tf.gfile.GFile(model_path, 'rb') as fid:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(fid.read())
			tf.import_graph_def(graph_def, name='')
	else:
		raise ValueError('%s does not exist or it\'s an unsupported format.' % model_path)

def _ensure_tensor_suffix_name(name):
	if name[-2:] != ':0':
		return name + ':0'
	else:
		return name

def vgg_preprocess(image):
	IMAGENET_MEANS = [123.68, 116.779, 103.939]
	if image.shape[-1] != 3:
		raise ValueError('The last dimension of argument \'image\' must be 3.')
	if len(image.shape) == 3:
		means = np.reshape(IMAGENET_MEANS, (1, 1, 3))
	elif len(image.shape) == 4:
		means = np.reshape(IMAGENET_MEANS, (1, 1, 1, 3))
	else:
		raise ValueError('The number of dimensions of argument \'image\' (%d) is not supported.' % len(image.shape))
	return image - means

def inception_preprocess(image):
	return image * (2.0 / 255.0) - 1.0

def produce_cams(tf_session, input_tensor, featuremap_tensor, logits_tensor, input_batch, class_index, prediction_tensor=None):
	num_classes = logits_tensor.shape[-1]
	if class_index == -1:
		logits = tf_session.run(logits_tensor, feed_dict={input_tensor: input_batch})
		class_indices = np.argmax(logits, axis=1)
	else:
		class_indices = [class_index]
	y = tf.reduce_sum(logits_tensor * tf.one_hot(class_indices, num_classes), axis=1)
	gradients = tf.gradients(y, featuremap_tensor)[0]
	weights = tf.reduce_mean(gradients, axis=[1, 2])
	# Make sure the shape of weights and feature map are compatible.
	# i.e. original shape [batch, #channels] => new shape [batch, 1, 1, #channels]
	weights_new_shape = [1] * len(featuremap_tensor.shape)
	weights_new_shape[0] = -1
	weights_new_shape[-1] = featuremap_tensor.shape[-1]
	weights = tf.reshape(weights, weights_new_shape)
	cams_tensor = tf.nn.relu(tf.reduce_sum(weights * featuremap_tensor, axis=3))

	if prediction_tensor is None:
		logits, cams = tf_session.run([logits_tensor, cams_tensor], feed_dict={input_tensor: input_batch})
		pred_cls_indices = np.argmax(logits, axis=1)
		print('The most probable class indices for this batch:', pred_cls_indices)
	else:
		prediction, cams = tf_session.run([prediction_tensor, cams_tensor], feed_dict={input_tensor: input_batch})
		pred_cls_indices = np.argmax(prediction, axis=1)
		print('The most probable class indices for this batch:')
		print('(class_index: probability)')
		for i in range(len(pred_cls_indices)):
			cls_idx = pred_cls_indices[i]
			print('%d: %.4f' % (cls_idx, prediction[i,cls_idx]))
	return cams

def visualize_cam(cam, image, save_image_path):
	image_im = Image.fromarray(image)
	cam_im = Image.fromarray(np.uint8(cam / np.amax(cam) * 255))
	cam_im = cam_im.resize(image_im.size, Image.BICUBIC)
	# matplotlib colormap accepts a NumPy array in the range of [0, 1].
	cam = np.array(cam_im, dtype=np.float64) / 255
	cam_im = Image.fromarray(np.uint8(cm.jet(cam) * 255)).convert('RGB')	# convert from RGBA to RGB
	result_im = Image.blend(cam_im, image_im, 0.5)
	result_im.save(save_image_path)

def main():
	args = _parse_args()

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.per_process_gpu_memory_fraction, allow_growth=False)
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
		_load_model(sess, args.model_path)
		if args.print_tensors:
			print('-- Tensors\' name start --')
			for n in tf.get_default_graph().as_graph_def().node:
				print(n.name)
			print('-- Tensors\' name end --')
			return
		input_tensor = tf.get_default_graph().get_tensor_by_name(_ensure_tensor_suffix_name(args.input_tensor_name))
		featuremap_tensor = tf.get_default_graph().get_tensor_by_name(_ensure_tensor_suffix_name(args.featuremap_tensor_name))
		logits_tensor = tf.get_default_graph().get_tensor_by_name(_ensure_tensor_suffix_name(args.logits_tensor_name))
		prediction_tensor = None
		if args.prediction_tensor_name is not None:
			prediction_tensor = tf.get_default_graph().get_tensor_by_name(_ensure_tensor_suffix_name(args.prediction_tensor_name))
		print('This model was trained for %d classes.' % logits_tensor.shape[-1])

		images_np = _get_and_resize_numpy_image_batch(
						args.input_path,
						(args.input_width, args.input_height))

		for num in range(images_np.shape[0]):
			print('On %d / %d' % (num + 1, images_np.shape[0]))
			image_np = images_np[num,:,:,:]
			input = image_np
			if args.preprocess == 'vgg':
				input = vgg_preprocess(input)
			elif args.preprocess == 'inception':
				input = inception_preprocess(input)
			elif args.preprocess is not None:
				raise ValueError('Invalid preprocessing method \'%s\'.' % str(args.preprocess))
			input = np.expand_dims(input, axis=0)
			assert len(input.shape) == 4

			cams = produce_cams(
					tf_session=sess,
					input_tensor=input_tensor,
					featuremap_tensor=featuremap_tensor,
					logits_tensor=logits_tensor,
					input_batch=input,
					class_index=args.class_index,
					prediction_tensor=prediction_tensor)

			assert cams.shape[0] == 1, 'Batch size=1 is only supported for the visualization.'
			visualize_cam(cam=cams[0,:,:], image=image_np, save_image_path='result%d.png' % (num + 1))

if __name__ == '__main__':
	main()
