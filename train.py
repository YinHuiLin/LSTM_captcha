#-*- coding:utf-8 -*


import tensorflow as tf
from util import *
from computational_graph_lstm import *
import predict



def train():

    # defining placeholders
    x = tf.placeholder("float",[None,time_steps,n_input], name = "x") #input image placeholder
    y = tf.placeholder("float",[None,captcha_num,n_classes], name = "y")  #input label placeholder

    # computational graph
    opt, loss, accuracy, pre_arg, y_arg = computational_graph_lstm(x, y)

    saver = tf.train.Saver()  # 创建训练模型保存类
    init = tf.global_variables_initializer()    #初始化变量值

    with tf.Session() as sess:  # 创建tensorflow session
        sess.run(init)


        # merge所有的summary node
        merged_summary_op= tf.summary.merge_all()
        # 可视化存储目录为当前文件夹下的 log
        merged_writer = tf.summary.FileWriter(log_dir, sess.graph)


        iter = 1
        while iter < iteration:
            batch_x, batch_y = get_batch()
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})   #只运行优化迭代计算图


            merged_summary ,_=sess.run([merged_summary_op,opt], feed_dict={x: batch_x, y: batch_y})


            if iter %100==0:
                los, acc, parg, yarg = sess.run([loss, accuracy, pre_arg, y_arg],feed_dict={x:batch_x,y:batch_y})
                print("For iter ",iter)
                print("Accuracy ",acc)
                print("Loss ",los)
                if iter % 1000 ==0:
                    print("predict arg:",parg[0:10])
                    print("yarg :",yarg[0:10])
                print("__________________")
            if iter % 100== 0:   #保存模型
                saver.save(sess, model_path, global_step=iter)
                merged_writer.add_summary(merged_summary, iter)
            iter += 1
        # 计算验证集准确率
        valid_x, valid_y = get_batch(data_path=validation_path, is_training=False)
        print("Validation Accuracy:", sess.run(accuracy, feed_dict={x: valid_x, y: valid_y}))



if __name__ == '__main__':
    train()

