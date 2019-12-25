def getPSNR(img1, img2):
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 0.0
    else:
        return 10.0*np.log10((1)/mse)

def readYUVFile(filename, path, frames_num=32, with_HM=False):
    name, ext = filename.rsplit('.', 1)
    width, height = [int(i) for i in name.split('_')[1].split('x')]
    if '10bit' in  filename:
        bitDepth='uint16'
    else:
        bitDepth='uint8'
    wuv = width // 2
    huv = height // 2

    Y = np.zeros((frames_num, height, width), dtype=bitDepth)
    U = np.zeros((frames_num, huv, wuv), dtype=bitDepth)
    V = np.zeros_like(U)
    if with_HM:
        mask = np.zeros_like(Y)

    with open(os.path.join(path, filename), 'r') as rfile:
        file_frames_num = os.fstat(rfile.fileno()).st_size /(width*height*3/2)
        try:
            assert frames_num <= file_frames_num
        except AssertionError:
            print(filename, frames_num, "vs", file_frames_num)

        for frame_idx in range(frames_num):
            Y[frame_idx, :, :] = np.fromfile(rfile, bitDepth, width * height).reshape([height, width])
            U[frame_idx, :, :] = np.fromfile(rfile, bitDepth, wuv * huv).reshape([huv, wuv])
            V[frame_idx, :, :] = np.fromfile(rfile, bitDepth, wuv * huv).reshape([huv, wuv])

            if with_HM:
                fp = open(os.path.join(path, name, 'PU_split_%d.txt' %(frame_idx)))
                lines = fp.readlines()
                weight_array = np.zeros((height, width),dtype=bitDepth)
                for line in lines:
                    int_list = [ int(i) for i in line.split(' ')]
                    ly,lx,ry,rx,part_size,pre_mode= int_list

                    value = np.mean(Y[frame_idx, :, :][lx:rx+1, ly:ry+1])
                    weight_array[lx:rx+1, ly:ry+1] = value
                mask[frame_idx, :, :] = weight_array
    if with_HM:
        return Y, U, V, mask
    else:
        return Y, U, V

def read_all_yuvs(img_list, path='', n_threads=32):
    """ Returns all images in array by given path and name of each image file. """
    imgs = []
    for idx in range(0, len(img_list), n_threads):
        b_imgs_list = img_list[idx : idx + n_threads]
        b_imgs = tl.prepro.threading_data(b_imgs_list, fn=readYUVFile, path=path)
        imgs.extend(b_imgs)
        print('read %d from %s' % (len(imgs), path))
    return imgs

def partition_aware(t_image, is_train=False, reuse=False, scope='partition_aware'):
    w_init = tf.contrib.layers.variance_scaling_initializer()
    b_init = tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(scope, reuse=reuse) as vs:
        i_img, heatmap = tf.split(t_image, 2, 3)
        tl.layers.set_name_reuse(reuse)
        n = InputLayer(i_img, name='in')
        hm = InputLayer(heatmap, name='in_2')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        n_hm = Conv2d(hm, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='hm_n64s1/c')

        temp = n_hm
        for i in range(8):
            nn = Conv2d(n_hm, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='hm64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='hm64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='hm64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='hm64s1/b2/%s' % i)
            nn = ElementwiseLayer([n_hm, nn], tf.add, 'hm_b_residual_add/%s' % i)
            n_hm = nn

        n_hm = Conv2d(n_hm, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='hm64s1/c/m')
        n_hm = BatchNormLayer(n_hm, is_train=is_train, gamma_init=g_init, name='hm64s1/b/m')
        n_hm = ElementwiseLayer([n_hm, temp], tf.add, 'hm_add3')

        temp = n

        for i in range(32):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c1/%s' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c2/%s' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/b2/%s' % i)
            nn = ElementwiseLayer([n, nn], tf.add, 'b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, 'add3')
        # B residual blacks end

        n = ElementwiseLayer([n_hm, n], tf.add, 'fusion_add')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/1')

        n = Conv2d(n, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n256s1/2')

        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='out')

        return n

def VRCNN_partition(t_image, is_train=False, reuse=False, scope='vrcnn_partition'):
    w_init = tf.contrib.layers.variance_scaling_initializer() #tf.random_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(value=0.0)

    with tf.variable_scope(scope, reuse=reuse) as vs:
        tl.layers.set_name_reuse(reuse)
        i_img, heatmap = tf.split(t_image, 2, 3)

        n = InputLayer(i_img, name='in')
        temp = n
        hm = InputLayer(heatmap, name='in_2')

        n = Conv2d(n, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64ks5s1')
        n_hm = Conv2d(hm, 64, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64ks5s1_hm')
        n_hm = Conv2d(n_hm, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='n64ks3s1_hm')

        n_1 = Conv2d(n, 16, (5, 5), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,name='n16ks5s1')
        n_2 = Conv2d(n, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,name='n32ks3s1')
        n = ConcatLayer(layer = [n_1, n_2], concat_dim=3, name='concat_1')
        n_1 = Conv2d(n, 16, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,name='n16ks3s1')
        n_2 = Conv2d(n, 32, (1, 1), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,name='n32ks1s1')
        n = ConcatLayer(layer = [n_1, n_2], concat_dim=3, name='concat_2')

        n = Conv2d(n, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,name='img_path')
        n = ElementwiseLayer([n_hm, n], tf.add, 'add_fusion')

        n = Conv2d(n, 32, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,name='conv_fusion')

        n = Conv2d(n, 1, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init,name='out')
        
        n = ElementwiseLayer([temp, n], tf.add, 'add')

        return n
if __name__ == '__main__':

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--QP', '-q', type=int, default=37, help='test QP value')
        parser.add_argument('--checkpoint', '-c', type=str, default=' ', help='checkpoint to be evaluted')
        parser.add_argument('--test_num', '-n', type=int, default=32, help='test frames number, default is 32')
        parser.add_argument('--info', type=str, default='test_result', help='output json filename')
        parser.add_argument('--recYuv_path', type=str, default='', help='rec yuv dir')
        parser.add_argument('--origYuv_path', type=str, default='', help='original yuv dir')
        parser.add_argument('--patch_size', type=int, default=64, help='patch_size, default is 64')
        parser.add_argument('--Yonly', action="store_true", help='only test Y channel if specified')
        args = parser.parse_args()

        from collections import OrderedDict
        import tensorflow as tf
        import numpy as np
        import sys, os, time
        import scipy
        import json, glob
        import tensorlayer as tl
        from tensorlayer.layers import *
        from scipy.misc import imread
        from scipy.misc import imsave

        tf_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        tf_config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf_config)

        ### LOAD MODEL ###
        t_image = tf.placeholder('float32', [None, None, None, 2], name='input_image')
        # net = VRCNN_partition(t_image, is_train=False, reuse=False)
        net = partition_aware(t_image, is_train=False, reuse=False)


        tl.layers.initialize_global_variables(sess)
        tl.files.load_and_assign_npz(sess=sess, name=args.checkpoint, network=net)

        test_seq_list = glob.glob(os.path.join(args.recYuv_path, '*QP%d*yuv' % args.QP))
        results = OrderedDict()
        print("Loaded %d sequences " % len(test_seq_list))

        for seq_path in test_seq_list:
            seq_filename = os.path.basename(seq_path)
            seq_name = os.path.splitext(seq_filename)[0]
            seq_name = seq_name.rsplit('_', 1)[0] #del QP part
            video_name = seq_name

            # results[video_name][comp_index][0] input psnr list
            # results[video_name][comp_index][1] output psnr list
            if not results.has_key(video_name):
                results[video_name] = [[[],[]],[[],[]],[[],[]]]

            orig_seq_filename =  seq_name + '.yuv'
            recYuvs_withHM = readYUVFile(seq_filename, args.recYuv_path,
                                         args.test_num, True)
            origYuvs = readYUVFile(orig_seq_filename, args.origYuv_path,
                                   args.test_num, False)
            sub_img_w = args.patch_size
            sub_img_h = args.patch_size
            stride_w = args.patch_size
            stride_h = args.patch_size

            if '10bit' in seq_name:
                max_pixel_val = 1023
            else:
                max_pixel_val = 255

            for index in range(args.test_num):
                frame_result = []
                for comp_index in range(3):
                    img_name = '%s_%d' %(seq_name, index)
                    if args.Yonly == 1 and comp_index > 0:
                        results[video_name][comp_index][0].append(0.0)
                        results[video_name][comp_index][1].append(0.0)
                        frame_result.append(0.0)
                        continue

                    img = recYuvs_withHM[comp_index][index].astype(np.float32) / max_pixel_val
                    label = origYuvs[comp_index][index].astype(np.float32) / max_pixel_val
                    mask = recYuvs_withHM[-1][index].astype(np.float32) / max_pixel_val
                    if comp_index > 0:
                        mask = mask[::2, ::2]

                    h, w = img.shape[0], img.shape[1]
                    img = np.stack((img, mask), -1)
                    output = np.ones((h,w), dtype=np.float32)

                    infer_time = list()
                    for l_w in range(0, w , stride_w):
                        for l_h in range(0, h , stride_h):
                            if l_h + sub_img_h >h:
                                l_h = h - sub_img_h
                            if l_w + sub_img_w > w:
                                l_w = w - sub_img_w
                            sub_img = img[l_h:l_h + sub_img_h, l_w:l_w + sub_img_w]
                            start = time.time()
                            sub_out = sess.run(net.outputs, {t_image: [sub_img]})
                            end = time.time()
                            infer_time.append(end-start)
                            output[l_h:l_h + sub_img_h, l_w:l_w + sub_img_w] = sub_out.reshape(sub_img_h, sub_img_w)

                    # cal PSNR img, output, label
                    in_psnr = getPSNR(img[:,:,0], label)
                    out_psnr = getPSNR(output, label)

                    results[video_name][comp_index][0].append(in_psnr)
                    results[video_name][comp_index][1].append(out_psnr)
                    frame_result.append(out_psnr - in_psnr)

        final_results = OrderedDict()
        for key, value in results.iteritems():
            assert len(value[0]) == len(value[1])
            gain = (np.asarray(value)[:,1] - np.asarray(value)[:,0]).mean(axis=-1)
            final_results[key] = list(gain)

        #output the average psnr gain
        print(final_results)

        average_gain = list(np.asarray(final_results.values()).mean(axis=0))
        print(average_gain)

        final_results['average gain'] = average_gain
        if not os.path.exists('./psnr_gain'):
            os.mkdir('./psnr_gain')

        with open(os.path.join('./psnr_gain', args.info + '.json'),'w') as fp:
            json.dump(final_results, fp, indent = 4)




