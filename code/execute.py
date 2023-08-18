import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from models import UCDMI, LogReg
from utils.clustering import assement_result,assement_directly
from utils.process import dataframe,parse_skipgram,process_tu,micro_f1,adj_to_bias,parse_index_file,sample_mask,load_data_our,load_data,sparse_to_tuple,standardize_data,preprocess_features,normalize_adj,preprocess_adj,sparse_mx_to_torch_sparse_tensor,convert_label,convert_label2,kmeans,get_vttdata,get_adj_lilmatrix,get_labelmatrix
#from utils import process, clustering
from sklearn.cluster import KMeans

#dataset = 'cora'
#dataset = 'citeseer'
# dataset = 'pubmed'
dataset = 'Protein'

# training params
batch_size = 1
nb_epochs = 200
patience = 200
#lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 128 #32
jaccard = 0.3 #0.3


b_xent = nn.BCEWithLogitsLoss()
xent = nn.CrossEntropyLoss()
cnt_wait = 0
best = 1e9
best_t = 0
best_nmi = 0
best_epoch = 0


hyperparameter_ranges = {
    'hid_units': [128, 64, 32],
    'lr': [0.001, 0.01, 0.1],
    #'batch_size': [16, 32, 64]
    'jaccard':[0.2,0.3,0.5]
}

best_hyperparameters = {}

best_metric = 0.5#float('-inf')  # Initialize with a negative value for maximization
for hid_units in hyperparameter_ranges['hid_units']:
    for lr in hyperparameter_ranges['lr']:
        for jaccard in hyperparameter_ranges['jaccard']:
            print(str(hid_units)+' '+str(lr)+' '+str(jaccard))


            save_Finally_values_list = []
            print("the value of jaccard is:{}".format(jaccard))
            sparse = True
            nonlinearity = 'prelu' # special name to separate parameters

            #adj, features, labels, kmeans_labels, idx_train, idx_val, idx_test, graph = load_data_our(dataset, jaccard)   ##labels: arraya类型[2708,7]
            data = dataframe(dataset, jaccard)

            print("dataframe done")

            """
adj = data.iloc[0,0]
features = data.iloc[0,1]
labels = data.iloc[0,2]
kmeans_labels = data.iloc[0,3]
idx_train = data.iloc[0,4]
idx_val = data.iloc[0,5]
idx_test = data.iloc[0,6]
graph = data.iloc[0,7]
    
labels_ori = labels
#adj_ori = process.sparse_mx_to_torch_sparse_tensor(adj)

vadj_ori = sparse_mx_to_torch_sparse_tensor(adj)

#cc_label = process.convert_label2(kmeans_labels)
#features, _ = process.preprocess_features(features)
cc_label = convert_label2(kmeans_labels)
features, _ = preprocess_features(features)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = labels.shape[1]

#adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
adj = normalize_adj(adj + sp.eye(adj.shape[0])) #COORDINATE matrix

if sparse:
    #sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
    sp_adj = sparse_mx_to_torch_sparse_tensor(adj)
else:
    adj = (adj + sp.eye(adj.shape[0])).todense()

features = torch.FloatTensor(features[np.newaxis]) # ADD new dimension
if not sparse:
    adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

#model = UCDMI(ft_size, hid_units, nonlinearity)
#optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    if sparse:
        sp_adj = sp_adj.cuda()
    else:
        adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda() """



            ft_size = 1
            model = UCDMI(ft_size, hid_units, nonlinearity)
            optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

            for epoch in range(nb_epochs):

                model.train()
                total_loss = 0.0
                n = 0
                #optimiser.zero_grad()

####################################################################
                for index, sample in data.iterrows():
                    optimiser.zero_grad()
        
                    adj = sample['adj']
                    features = sample['features']
                    labels = sample['labels']
                    kmeans_labels = sample['kmeans_features_labels'] 
                    idx_train = sample['idx_train']
                    idx_val = sample['idx_val']
                    idx_test = sample['idx_test']
                    graph = sample['G']
                    weight = sample['weight_matrix']
        
      
                    labels_ori = labels
    
                    vadj_ori = sparse_mx_to_torch_sparse_tensor(adj)
    
                    cc_label = convert_label2(kmeans_labels)
                    features, _ = preprocess_features(features)

                    nb_nodes = features.shape[0]
                    ft_size = features.shape[1]
                    nb_classes = labels.shape[1]

                    adj = normalize_adj(adj + sp.eye(adj.shape[0])) #COORDINATE matrix

                    if sparse:
                        #sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
                        sp_adj = sparse_mx_to_torch_sparse_tensor(adj)
                    else:
                        adj = (adj + sp.eye(adj.shape[0])).todense()

                    features = torch.FloatTensor(features[np.newaxis]) # ADD new dimension
                    if not sparse:
                        adj = torch.FloatTensor(adj[np.newaxis])
                    labels = torch.FloatTensor(labels[np.newaxis])
                    idx_train = torch.LongTensor(idx_train)
                    idx_val = torch.LongTensor(idx_val)
                    idx_test = torch.LongTensor(idx_test)

                    if torch.cuda.is_available():
                        print('Using CUDA')
                        model.cuda()
                        features = features.cuda()
                        if sparse:
                            sp_adj = sp_adj.cuda()
                        else:
                            adj = adj.cuda()
                        labels = labels.cuda()
                        idx_train = idx_train.cuda()
                        idx_val = idx_val.cuda()
                        idx_test = idx_test.cuda()
        

           ####################################################################

                    idx = np.random.permutation(nb_nodes)
                    shuf_fts = features[:, idx, :] #shuffle feature matrix

                    lbl_1 = torch.ones(batch_size, nb_nodes)
                    lbl_2 = torch.zeros(batch_size, nb_nodes)
                    lbl = torch.cat((lbl_1, lbl_2), 1)


                    if torch.cuda.is_available():
                        shuf_fts = shuf_fts.cuda()
                        lbl = lbl.cuda()

                    logits= model(cc_label, features, shuf_fts, sp_adj if sparse else adj, sparse, None, None, None)
                                                  #suffled features    #adj = S
                    loss = b_xent(logits, lbl)
        
                    loss.backward()
                    optimiser.step()
        
                    total_loss += loss#.item()
                    n= n+1
                    embeds_1, embeds_2 = model.embed(features, sp_adj if sparse else adj, sparse, None)
                    #average_NMI, average_F1score, average_ARI, average_Acc = assement_result(labels_ori, embeds_1, nb_classes)

                # print('Loss:', loss)
                #print("{0}th epoch | loss:{1} | nmi:{2} | acc:{3} | f-score:{4} | ari:{5}".format(epoch, loss, average_NMI, average_Acc, average_F1score, average_ARI))
                average_loss = total_loss / n
        
                print("{0}th epoch | loss:{1} ".format(epoch, average_loss))
    
                """
                if average_NMI > best_nmi:
                    best_nmi = average_NMI
                    best_epoch = epoch
                    torch.save(model.state_dict(), 'best_ucdmi_nmi.pkl') """

                if loss < best:
                    best = loss
                    best_t = epoch
                    cnt_wait = 0
                    torch.save(model.state_dict(), 'best_ucdmi.pkl')
                else:
                    cnt_wait += 1

                if cnt_wait == patience:
                    print('Early stopping!')
                    break
            
            if total_loss < best_metric:
                best_metric = total_loss
                best_hyperparameters = {
                    'hidden_units': hid_units,
                    'learning_rate': lr,
                    'jaccard': jaccard
                }

print("Best Hyperparameters:", best_hyperparameters)

res_1, res_2 = model.embed(features, sp_adj if sparse else adj, sparse, None)
print('the {}th epoch is the best epoch'.format(best_epoch))

print(embeds_1) #H
print("shape:",embeds_1.shape)
print("***************************")
print(embeds_2)
print("shape:",embeds_2.shape)

kmeans = KMeans(n_clusters=7, random_state=0)
community_labels = kmeans.fit_predict(res_1[0])
print(community_labels)

cluster_assignments = community_labels
"""
# Initialize K-Means
kmeans = KMeans(n_clusters=7, random_state=0)
output = _
output = np.asarray(output)
output = np.mat(output)
output = output.T
output = sp.lil_matrix(output)
cluster_assignments = kmeans.fit_predict(output)
print(cluster_assignments)
"""
# Get unique cluster IDs
unique_clusters = np.unique(cluster_assignments)

# Initialize arrays to store community points and community sizes
community_points = {cluster_id: [] for cluster_id in unique_clusters}
community_sizes = {cluster_id: 0 for cluster_id in unique_clusters}

# Iterate through cluster assignments and accumulate points for each community
for node_id, cluster_id in enumerate(cluster_assignments):
    community_points[cluster_id].append(node_id)
    community_sizes[cluster_id] += 1

# Convert community points and sizes to NumPy arrays
for cluster_id in unique_clusters:
    community_points[cluster_id] = np.array(community_points[cluster_id])
    community_sizes[cluster_id] = community_sizes[cluster_id]

# Print the points in each community and the community sizes
for community_id, points in community_points.items():
    print(f"Community {community_id}: Points = {points}, Size = {community_sizes[community_id]}")
#torch.save(model, 'finalmodel')

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Example cluster assignments and distance matrix (replace with your data)
#cluster_assignments = np.array([0, 1, 0, 1, 2, 2, 1, 0, 0, 2])
"""distance_matrix = np.loadtxt('./data_part/Protein/weight_matrix.txt')
distance_matrix = np.asarray(distance_matrix)
distance_matrix = np.mat(distance_matrix)
distance_matrix = distance_matrix.T
distance_matrix = np.array(distance_matrix)
import math
shape = math.sqrt(distance_matrix.size)
distance_matrix = distance_matrix.reshape(int(shape),int(shape))
distance_matrix = np.round(distance_matrix, decimals=2)
print(distance_matrix)"""

# Perform Multi-Dimensional Scaling (MDS) to get positions
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=0)
pos = mds.fit_transform(weight)

# Create a graph
G = nx.Graph()

# Add nodes with positions
for node_id, position in enumerate(pos):
    G.add_node(node_id, pos=position)

# Plot points for each community
for cluster_id in np.unique(cluster_assignments):
    community_nodes = np.where(cluster_assignments == cluster_id)[0]
    nx.draw_networkx_nodes(G, pos, nodelist=community_nodes, node_color=f'C{cluster_id}', label=f'Community {cluster_id}')

# Draw edges
nx.draw_networkx_edges(G, pos)

# Add labels
labels = {i: f'Node {i}' for i in range(len(cluster_assignments))}
nx.draw_networkx_labels(G, pos, labels)

# Show the plot
plt.title("Points in Communities")
plt.legend()
plt.show()
