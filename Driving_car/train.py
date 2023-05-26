import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model

LOGDIR = './save'
sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

loss = tf.reduce_mean(tf.square(tf.subtract(model.y, model.y_))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) *L2NormConst
train_step = tf.train.AdamOptimier(1e-4).minimize(loss)
sess.run(tf.global_variales_initializer())

tf.summary.scalar('loss', loss)

merged_summary_op = tf.summary.merge_all()

saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V2)

logs_path = "./logs"
summary_writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

epochs = 30
batch_size = 100

for epoch in range(epochs):
    for i in range(int(driving_data.num_images/batch_size)):
        xs, ys = driving_data.LoadTrainBatch(batch_size)
        train_step.run(feed_dict = {model.x : xs, model.y_ : ys, model.keep_prob : 0.8})
        if i %  10 == 0:
            xs,ys = driving_data.LoadTrainBatch(batch_size)
            loss_value = loss.eval(feed_dict = {model.x : xs, model.y_: ys, model.keep_prob : 1.0})
            print("Epoch : %d, Step: %d, Loss: %g" %(epoch, epoch*batch_size + i, loss_value))


        summary = merged_summary_op.eval(feed_dict = {model.x:xs, model.y_ : ys, model.keep_prob: 1.0})
        summary_writer.add_summary(summary, epoch*driving_data.num_images/batch_size +i)

        if i % batch_size == 0:
            if not os.path.exists(LOGDIR):
                os.makedirs(LOGDIR)
            checkpoint_path = os.path.join(LOGDIR, 'model.ckpt')
            filename = saver.save(sess, checkpoint_path)
    print("Model saved in file: %s" %filename)
