import networkx as nx
import pandas as pd

from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

__all__ = ["YelpGNNDataset", ]


class YelpGNNDataset:

    def __init__(
        self, user_df: pd.DataFrame, destinations_df: pd.DataFrame, reviews_df: pd.DataFrame
    ): 
      self.user_df = user_df
      self.destinations_df = destinations_df
      self.reviews_df = reviews_df

      self.G: nx.Graph = None  

      self.train_data: Data = None
      self.test_data: Data = None 
      self.val_data: Data = None


    def create_interaction_graph(    
        self, 
        user_df: pd.DataFrame, 
        destinations_df: pd.DataFrame, 
        reviews_df: pd.DataFrame,
        user_exclude_columns: list[str], 
        business_exclude_columns: list[str]
    ):
        user_feature_columns = [col for col in user_df.columns if col not in user_exclude_columns]
        business_feature_columns = [col for col in destinations_df.columns if col not in business_exclude_columns]

        user_features = user_df[user_feature_columns].copy()

        business_features = destinations_df[business_feature_columns].copy()

        user_id_mapping = {user_id: idx for idx, user_id in enumerate(user_df['user_id'].unique())}
        business_id_mapping = {business_id: idx + len(user_id_mapping) for idx, business_id in enumerate(destinations_df['business_id'].unique())}

        G = nx.Graph()

        for _, row in user_df.iterrows():
            user_id = row['user_id']
            user_idx = user_id_mapping[user_id]  # Use the numeric index for user
            features = torch.tensor(user_features.loc[row.name].values, dtype=torch.float)
            G.add_node(user_idx, type='user', features=features)

        for _, row in destinations_df.iterrows():
            business_id = row['business_id']
            business_idx = business_id_mapping[business_id]  # Use the numeric index for business
            features = torch.tensor(business_features.loc[row.name].values, dtype=torch.float)
            G.add_node(business_idx, type='business', features=features)

        for _, interaction in reviews_df.iterrows():
            user_id = interaction['user_id']
            business_id = interaction['business_id']
            user_idx = user_id_mapping[user_id]
            business_idx = business_id_mapping[business_id]

            rating = interaction['stars']

            G.add_edge(user_idx, business_idx, type='interaction', rating=rating)

        return G
    

    def split_edges(self, edge_index, split_ratio=[0.7, 0.15, 0.15]):
        
        num_edges = edge_index.shape[1]
        indices = list(range(num_edges))
        random.shuffle(indices)

        train_size = int(split_ratio[0] * num_edges)
        val_size = int(split_ratio[1] * num_edges)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_edge_index = edge_index[:, train_indices]
        val_edge_index = edge_index[:, val_indices]
        test_edge_index = edge_index[:, test_indices]

        return train_edge_index, val_edge_index, test_edge_index
    

    def create_graph_datasets(self, G: nx.Graph):
        
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

        train_edge_index, val_edge_index, test_edge_index = self.split_edges(edge_index)

        train_edge_attr = torch.tensor([G[int(u)][int(v)]['rating'] for u, v in zip(train_edge_index[0].tolist(), train_edge_index[1].tolist())], dtype=torch.float)
        val_edge_attr = torch.tensor([G[int(u)][int(v)]['rating'] for u, v in zip(val_edge_index[0].tolist(), val_edge_index[1].tolist())], dtype=torch.float)
        test_edge_attr = torch.tensor([G[int(u)][int(v)]['rating'] for u, v in zip(test_edge_index[0].tolist(), test_edge_index[1].tolist())], dtype=torch.float)

        user_features = torch.stack([node_data['features'].float() for node_id, node_data in G.nodes(data=True) if node_data['type'] == 'user'])
        item_features = torch.stack([node_data['features'].float() for node_id, node_data in G.nodes(data=True) if node_data['type'] == 'business'])
        user_feature_sizes = [node_data['features'].shape for node_id, node_data in G.nodes(data=True) if node_data['type'] == 'user']
        item_feature_sizes = [node_data['features'].shape for node_id, node_data in G.nodes(data=True) if node_data['type'] == 'business']

        max_user_feature_size = max(user_feature_sizes, key=lambda x: x[0])[0]
        max_item_feature_size = max(item_feature_sizes, key=lambda x: x[0])[0]
        max_feature_size = max(max_user_feature_size, max_item_feature_size)

        user_features = torch.stack([
            F.pad(node_data['features'].float(), (0, max_feature_size - node_data['features'].shape[0]))
            for node_id, node_data in G.nodes(data=True) if node_data['type'] == 'user'
        ])

        item_features = torch.stack([
            F.pad(node_data['features'].float(), (0, max_feature_size - node_data['features'].shape[0]))
            for node_id, node_data in G.nodes(data=True) if node_data['type'] == 'business'
        ])

        node_features = torch.cat([user_features, item_features], dim=0)

        train_data = Data(edge_index=train_edge_index, edge_attr=train_edge_attr, x=node_features)
        val_data = Data(edge_index=val_edge_index, edge_attr=val_edge_attr, x=node_features)
        test_data = Data(edge_index=test_edge_index, edge_attr=test_edge_attr, x=node_features)

        return train_data, val_data, test_data

    def get_gnn_datasets(self):
        return self.train_data, self.val_data, self.test_data

    def process(self):
       
       self.G = self.create_interaction_graph(
           user_df=self.user_df, destinations_df=self.destinations_df, reviews_df=self.reviews_df,
           user_exclude_columns=['user_id'], business_exclude_columns=['business_id']
       )

       self.train_data, self.val_data, self.test_data = self.create_graph_datasets(G= self.G)

    def save(self, save_dir: str):
        save_dir = Path(save_dir)

        torch.save(
            self.get_gnn_datasets(),
            f= save_dir / "gnn_datasets.pt"
        )