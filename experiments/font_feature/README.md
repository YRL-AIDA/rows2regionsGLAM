# Исследование вопроса влияния признаков шрифта на качество детекции

node_featch:	18
edge_featch:	4
epochs:	10
batch_size:	64
learning_rate:	0.001
Tag:	[{'in': -1, 'size': 512, 'out': 512, 'k': 3}, {'in': 512, 'size': 256, 'out': 256, 'k': 3}]
NodeLinear:	[-1, 64, 32]
NodeLinearClassifier:	[-1, 16, 8]
EdgeLinear:	[-1, 16, 4]
batchNormNode:	True
batchNormEdge:	True
seg_k:	0.5
loss_params:	{'edge_coef': 0.8, 'node_coef': 0.2, 'publaynet_imbalance': [0.07937020808458328, 0.007516949903219938, 0.40734851360321045, 0.2979268431663513, 0.01713326945900917, 0.19070428609848022], 'edge_imbalance': 0.3884359300136566}
sigmoidEdge:	False
NodeClasses:	6
EPOCH #0	 0.23527778 (VAL: 0.13090509)
EPOCH #1	 0.11247621 (VAL: 0.09034955)
EPOCH #2	 0.08078049 (VAL: 0.06746254)
EPOCH #3	 0.06578160 (VAL: 0.06022294)
EPOCH #4	 0.05889988 (VAL: 0.05572631)
EPOCH #5	 0.05344955 (VAL: 0.05346141)
EPOCH #6	 0.04936723 (VAL: 0.05038043)
EPOCH #7	 0.04577651 (VAL: 0.05001221)
EPOCH #8	 0.04232542 (VAL: 0.04883345)
EPOCH #9	 0.03893763 (VAL: 0.04945286)


Test Result
mAP@IoU[0.50:0.95]   :0.50563002
==================================================
threshold_05--------------------
precision_row       :0.7891
recall_row          :0.7644
f1_row              :0.7766
precision_word      :0.7683
recall_word         :0.7445
f1_word             :0.7562
threshold_95--------------------
precision_row       :0.6974
recall_row          :0.6734
f1_row              :0.6852
precision_word      :0.6750
recall_word         :0.6515
f1_word             :0.6631

node_featch:	18
edge_featch:	4
epochs:	32
batch_size:	64
learning_rate:	0.001
Tag:	[{'in': -1, 'size': 512, 'out': 512, 'k': 3}, {'in': 512, 'size': 256, 'out': 256, 'k': 3}]
NodeLinear:	[-1, 64, 32]
NodeLinearClassifier:	[-1, 16, 8]
EdgeLinear:	[-1, 16, 4]
batchNormNode:	True
batchNormEdge:	True
seg_k:	0.5
loss_params:	{'edge_coef': 0.8, 'node_coef': 0.2, 'publaynet_imbalance': [0.07937020808458328, 0.007516949903219938, 0.40734851360321045, 0.2979268431663513, 0.01713326945900917, 0.19070428609848022], 'edge_imbalance': 0.3884359300136566}
sigmoidEdge:	False
NodeClasses:	6
EPOCH #0	 0.23938078 (VAL: 0.16392652)
EPOCH #1	 0.11424209 (VAL: 0.08761628)
EPOCH #2	 0.07230400 (VAL: 0.06954269)
EPOCH #3	 0.06033815 (VAL: 0.06451580)
EPOCH #4	 0.05462853 (VAL: 0.06150483)
EPOCH #5	 0.04964720 (VAL: 0.05986263)
EPOCH #6	 0.04580726 (VAL: 0.05958392)
EPOCH #7	 0.04179272 (VAL: 0.05848273)
EPOCH #8	 0.03886794 (VAL: 0.05814540)
EPOCH #9	 0.03601260 (VAL: 0.05840935)
EPOCH #10	 0.03430916 (VAL: 0.05856342)
EPOCH #11	 0.03284716 (VAL: 0.05926307)
EPOCH #12	 0.03105497 (VAL: 0.06006919)
EPOCH #13	 0.02823145 (VAL: 0.05971777)
EPOCH #14	 0.02637505 (VAL: 0.05923811)
EPOCH #15	 0.02546853 (VAL: 0.06139713)
EPOCH #16	 0.02415880 (VAL: 0.06551022)
EPOCH #17	 0.02352164 (VAL: 0.06394664)
EPOCH #18	 0.02282242 (VAL: 0.06641656)
EPOCH #19	 0.02176196 (VAL: 0.06640180)
EPOCH #20	 0.02041866 (VAL: 0.06928625)
EPOCH #21	 0.02154601 (VAL: 0.06876830)
EPOCH #22	 0.02013591 (VAL: 0.06895516)
EPOCH #23	 0.01917107 (VAL: 0.07186610)
EPOCH #24	 0.01838091 (VAL: 0.07074430)
EPOCH #25	 0.01756230 (VAL: 0.07011918)
EPOCH #26	 0.01693138 (VAL: 0.07551060)
EPOCH #27	 0.01583352 (VAL: 0.07754556)
EPOCH #28	 0.01518139 (VAL: 0.07790663)
EPOCH #29	 0.01481756 (VAL: 0.08278203)
EPOCH #30	 0.01503398 (VAL: 0.08164445)
EPOCH #31	 0.01544752 (VAL: 0.08085968)

mAP@IoU[0.50:0.95]   :0.54036778
==================================================
threshold_05--------------------
precision_row       :0.8538
recall_row          :0.7784
f1_row              :0.8143
precision_word      :0.8119
recall_word         :0.7408
f1_word             :0.7748
threshold_95--------------------
precision_row       :0.7706
recall_row          :0.6987
f1_row              :0.7329
precision_word      :0.7268
recall_word         :0.6597
f1_word             :0.6916

node_featch:	18
edge_featch:	4
learning_rate:	0.001
model_type:	2
concat_gcn:	True
mlp_pred:	[{'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 256}, {'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 128}]
gcn_node:	[{'linear_in': -1, 'linear_out': 256, 'batch_norm': True, 'activation': 'gelu', 'aggregation': 'tag', 'K': 3, 'size': 128}, {'linear_in': -1, 'linear_out': 256, 'batch_norm': False, 'activation': 'gelu', 'aggregation': 'tag', 'K': 3, 'size': 64}]
mlp_node_class:	[{'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 256}, {'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 128}, {'in': -1, 'batch_norm': False, 'activation': 'softmax', 'out': 6}]
mlp_node_pred:	[{'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 256}, {'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 256}, {'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 128}]
gcn_node_post:	[{'linear_in': -1, 'linear_out': 256, 'batch_norm': True, 'activation': 'gelu', 'aggregation': 'tag', 'K': 3, 'size': 128}, {'linear_in': -1, 'linear_out': 256, 'batch_norm': False, 'activation': 'gelu', 'aggregation': 'tag', 'K': 3, 'size': 128}]
mlp_node_edge:	[{'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 256}, {'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 128}, {'in': -1, 'batch_norm': False, 'activation': None, 'out': 128}]
mlp_edge_class:	[{'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 256}, {'in': -1, 'batch_norm': False, 'activation': 'gelu', 'out': 64}, {'in': -1, 'batch_norm': False, 'activation': None, 'out': 1}]
seg_k:	0.5
loss_params:	{'edge_coef': 0.8, 'node_coef': 0.2, 'publaynet_imbalance': [0.07937020808458328, 0.007516949903219938, 0.40734851360321045, 0.2979268431663513, 0.01713326945900917, 0.19070428609848022], 'edge_imbalance': 0.3884359300136566}
sigmoidEdge:	False
NodeClasses:	6
epochs:	10
batch_size:	64
EPOCH #0	 0.22079478 (VAL: 0.09975751)
EPOCH #1	 0.08537285 (VAL: 0.07775161)
EPOCH #2	 0.07038620 (VAL: 0.06602354)
EPOCH #3	 0.06240240 (VAL: 0.06431158)
EPOCH #4	 0.05787403 (VAL: 0.05878517)
EPOCH #5	 0.05299900 (VAL: 0.05578286)
EPOCH #6	 0.05108592 (VAL: 0.05440329)
EPOCH #7	 0.04774002 (VAL: 0.05288750)
EPOCH #8	 0.04619152 (VAL: 0.05436699)
EPOCH #9	 0.04449124 (VAL: 0.05481186)

mAP@IoU[0.50:0.95]   :0.58775783
==================================================
threshold_05--------------------
precision_row       :0.8578
recall_row          :0.8215
f1_row              :0.8393
precision_word      :0.8213
recall_word         :0.7864
f1_word             :0.8035
threshold_95--------------------
precision_row       :0.7846
recall_row          :0.7543
f1_row              :0.7691
precision_word      :0.7450
recall_word         :0.7158
f1_word             :0.7301