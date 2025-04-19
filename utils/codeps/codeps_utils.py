import torch
from torch import Tensor
from torchvision import transforms
import numpy as np
import random
import faiss
import traceback


class EdgeAwareSmoothnessLoss:
    """Edge-aware smoothness loss
    """

    def __init__(self):
        pass

    @staticmethod
    def _compute_loss(disp: Tensor, img: Tensor) -> Tensor:
        """Compute the edge-aware smoothness loss for a normalized disparity image.
        Parameters
        ----------
        disp : torch.Tensor
            The normalized disparity image
        img : torch.Tensor
            The corresponding RGB image used to consider edge-aware smoothness
        Returns
        -------
        loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        loss_x = grad_disp_x * torch.exp(-grad_img_x)
        loss_y = grad_disp_y * torch.exp(-grad_img_y)
        loss = loss_x.mean() + loss_y.mean()

        return loss

    def __call__(self, target_image: Tensor, disparity_map: Tensor) -> Tensor:
        """Compute the edge-aware smoothness loss for a disparity image.
        Parameters
        ----------
        disp : torch.Tensor
            The disparity image, i.e., the inverse depth
        img : torch.Tensor
            The corresponding RGB image used to consider edge-aware smoothness
        Returns
        -------
        loss : torch.Tensor
            A scalar tensor with the computed loss multiplied by the smoothness factor
        """
        mean_disparity = disparity_map.mean(2, True).mean(3, True)
        norm_disparity = disparity_map / (mean_disparity + 1e-7)
        loss = self._compute_loss(norm_disparity, target_image).sum()
        return loss


class DiversityBuffer:
    def __init__(self, feature_dim, max_size, similarity_threshold=0.8):
        self.feature_dim = feature_dim  # Dimension of feature vectors
        self.max_size = max_size  # Maximum buffer size
        self.similarity_threshold = similarity_threshold  # Cosine similarity threshold
        
        # Initialize FAISS index for cosine similarity (inner product on normalized vectors)
        self.index = faiss.IndexIDMap(faiss.index_factory(feature_dim, "Flat", faiss.METRIC_INNER_PRODUCT))
        self.buffer = []  # List to store features
        self.images = []
        self.features = []
        self.depths = []
        self.ids = []  # List to store unique IDs for features in the buffer

    def add_feature(self, feature, feature_img, feature_depth, feature_id):
        feature = feature.detach()
        # Normalize new feature for cosine similarity
        # new_feature = new_feature / np.linalg.norm(new_feature.mean(-1).mean(-1).cpu().detach().numpy())
        normalized_feature = feature.mean(-1).mean(-1).cpu().numpy()
        normalized_feature /= np.linalg.norm(normalized_feature)

        # Check if buffer is empty, if so add directly
        if len(self.buffer) == 0:
            self._add_to_buffer(normalized_feature, feature_img, feature, feature_depth, feature_id)
            # print(f"Added feature {feature_id} to the buffer (buffer was empty).")
            return
        
        # Query FAISS index for cosine similarity with existing features
        similarity, _ = self.index.search(normalized_feature.reshape(1, -1), 1)
        # If similarity is below threshold, add the new feature
        if similarity[0][0] < self.similarity_threshold:
            self._add_to_buffer(normalized_feature, feature_img, feature, feature_depth, feature_id)
            # print(f"Added feature {feature_id} with similarity {similarity[0][0]:.2f}.")
        else:
            # print(f"Skipped feature {feature_id} with similarity {similarity[0][0]:.2f} (too similar).")
            pass

    def _add_to_buffer(self, normalized_feature, image, feature, depth, feature_id):
        # Add feature and ID to buffer
        self.buffer.append(normalized_feature)
        self.images.append(image)
        self.features.append(feature)
        self.depths.append(depth)
        self.ids.append(feature_id)
        
        # Add feature to FAISS index with its ID
        self.index.add_with_ids(np.array(normalized_feature, dtype=np.float32).reshape(1, -1), np.array([int(feature_id)], dtype=np.int64))

        # Remove the least diverse sample if buffer exceeds max size
        if len(self.buffer) > self.max_size:
            self._remove_least_diverse()

    def _remove_least_diverse(self):
        # Calculate pairwise distances and identify the most similar pair
        all_features = np.vstack(self.buffer)
        dist_matrix, _ = self.index.search(all_features, len(self.buffer))
        
        # Sum of distances for each feature
        diversity_scores = dist_matrix.sum(axis=1) - dist_matrix.diagonal()
        least_diverse_idx = np.argmin(diversity_scores)
        
        # Remove least diverse feature from buffer and FAISS index
        self.index.remove_ids(np.array([self.ids[least_diverse_idx]]))
        removed_id = self.ids.pop(least_diverse_idx)
        self.buffer.pop(least_diverse_idx)
        self.images.pop(least_diverse_idx)
        self.features.pop(least_diverse_idx)
        self.depths.pop(least_diverse_idx)
        
        print(f"Removed least diverse feature with ID {removed_id}.")
    
    def sample_features(self, batch_size):
        # Randomly sample a batch of features from the buffer
        sampled_indices = random.sample(range(len(self.buffer)), batch_size)
        sampled_features = [self.features[i] for i in sampled_indices]
        sampled_images = [self.images[i] for i in sampled_indices]
        sampled_depths = [self.depths[i] for i in sampled_indices]
        
        return (torch.stack(sampled_features), torch.stack(sampled_images), torch.stack(sampled_depths))  # Return as a tensor for the decoder
    
    def __len__(self):
        return len(self.buffer)