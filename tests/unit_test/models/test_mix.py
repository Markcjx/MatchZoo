import sys
sys.path.append('D:\\mine\\MatchZoo')
import matchzoo as mz
import tensorflow as tf
model = mz.models.Mix()
# model.params['idf_table'] = idf_df['idf']
ranking_task = mz.tasks.Ranking(loss=mz.losses.RankHingeLoss())
ranking_task.metrics = [
    mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
    mz.metrics.NormalizedDiscountedCumulativeGain(k=5),
    mz.metrics.MeanAveragePrecision()
]
print("`ranking_task` initialized with metrics", ranking_task.metrics)
model.params['input_shapes'] = [(10,), (40,)]
model.params['task'] = ranking_task
model.params['embedding_output_dim'] = 100
model.params['embedding_trainable'] = True
model.params['num_blocks'] = 2
model.params['kernel_count'] = [32, 32]
model.params['kernel_size'] = [[3, 3], [3, 3]]
model.params['dpool_size'] = [3, 10]
model.params['optimizer'] = 'adam'
model.params['dropout_rate'] = 0.1
model.build()
tf.reduce_max()