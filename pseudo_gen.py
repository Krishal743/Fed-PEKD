# FILE: pseudo_gen.py

import torch
from sklearn.mixture import GaussianMixture
import numpy as np

def generate_gmm_embeddings(client_knowledge, num_samples_per_class=100, num_classes=10):
   # ... (This function remains the same) ...
    all_features = []
    all_logits = []

    for k in client_knowledge:
        all_features.append(k['features'])
        all_logits.append(k['logits'])

    all_features = torch.cat(all_features)
    all_logits = torch.cat(all_logits) # Keep this as a tensor

    all_labels = torch.argmax(all_logits, dim=1).numpy()
    all_features_np = all_features.detach().numpy()

    generated_features = []
    generated_logits = []
    valid_classes = set() # Track classes we actually generated data for

    for c in range(num_classes):
        class_mask = (all_labels == c)
        class_features = all_features_np[class_mask]

        if len(class_features) < 10:
            # print(f"Warning: Skipping GMM for class {c}, not enough samples ({len(class_features)}).")
            continue

        n_components = 1
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type='diag',
                              reg_covar=1e-4)

        try:
            gmm.fit(class_features)
        except ValueError:
            # print(f"Warning: GMM fit failed for class {c} even with safeguards. Skipping.")
            continue

        new_features, _ = gmm.sample(num_samples_per_class)
        class_logits = all_logits[class_mask]
        mean_logits = class_logits.mean(dim=0)
        new_logits = mean_logits.unsqueeze(0).repeat(num_samples_per_class, 1)

        generated_features.append(torch.tensor(new_features, dtype=torch.float32))
        generated_logits.append(new_logits)
        valid_classes.add(c) # Mark class as valid


    if not generated_features:
        print("Warning: GMM generator (from samples) failed to produce any samples.")
        return None, None, set() # Return empty set

    # Return valid_classes as the third element
    return torch.cat(generated_features), torch.cat(generated_logits), valid_classes


def generate_gmm_from_prototypes(client_knowledge, num_samples_per_class=1000, num_classes=10):
    """
    Generates pseudo-embeddings by fitting GMMs to the CLASS PROTOTYPES
    provided by clients. Returns the generated features, logits, and the set of valid classes.
    """
    
    all_prototypes = []
    all_prototype_logits = []
    
    for k in client_knowledge:
        all_prototypes.append(k['features'])
        all_prototype_logits.append(k['logits'])
        
    all_prototypes = torch.cat(all_prototypes)
    all_prototype_logits = torch.cat(all_prototype_logits)
    
    if all_prototypes.shape[0] == 0:
        print("Error: No prototypes received from any client. Skipping round.")
        return None, None, set() # Return empty set
    
    all_labels = torch.argmax(all_prototype_logits, dim=1).numpy()
    all_prototypes_np = all_prototypes.detach().numpy()
    
    generated_features = []
    generated_logits = []
    valid_classes = set() # Track classes we actually generated data for
    
    global_mean_prototype = all_prototypes_np.mean(axis=0)
    
    for c in range(num_classes):
        class_mask = (all_labels == c)
        class_prototypes = all_prototypes_np[class_mask]
        
        # Determine if we have enough data to proceed for this class
        can_generate = False
        if len(class_prototypes) == 0:
            print(f"Info: No prototypes for class {c}. Skipping generation for this class.")
            # We don't add to valid_classes
            
        elif len(class_prototypes) == 1:
            # print(f"Info: Using single prototype for class {c}.")
            mean = class_prototypes[0]
            covariance = np.ones_like(mean) * 0.1 
            new_features = np.random.normal(mean, np.sqrt(covariance), 
                                            size=(num_samples_per_class, len(mean)))
            can_generate = True
        
        else: # 2+ Prototypes
            n_components = 1 
            gmm = GaussianMixture(n_components=n_components, 
                                  covariance_type='diag', 
                                  reg_covar=1e-4)
            try:
                gmm.fit(class_prototypes)
                new_features, _ = gmm.sample(num_samples_per_class)
                can_generate = True
            except ValueError:
                print(f"Warning: GMM (prototype) fit failed for class {c}. Skipping generation for this class.")
                # We don't add to valid_classes
        
        # If we successfully generated features for this class
        if can_generate:
            class_logits = all_prototype_logits[class_mask]
            # Need to handle case where class_mask might be empty if len(prototypes)==0
            if class_logits.shape[0] > 0:
                 mean_logits = class_logits.mean(dim=0)
            else:
                 # Fallback: uncertain logits if no prototypes were found
                 mean_logits = torch.full((num_classes,), 1.0 / num_classes)
                 
            new_logits = mean_logits.unsqueeze(0).repeat(num_samples_per_class, 1)
            
            generated_features.append(torch.tensor(new_features, dtype=torch.float32))
            generated_logits.append(new_logits)
            valid_classes.add(c) # Mark class as valid only if generated
            
    if not generated_features:
        print("Warning: Proto-GMM generator failed to produce any samples.")
        return None, None, set() # Return empty set

    # Return valid_classes as the third element
    return torch.cat(generated_features), torch.cat(generated_logits), valid_classes