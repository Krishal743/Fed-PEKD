# FILE: evaluate_model.py

import torch
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import os

# Make sure these imports match your project structure
from models import SimpleCNN, Classifier
from data import get_data # Use the same data loading

def load_model(method, state_dict_path_classifier, state_dict_path_extractor=None):
    """ Loads the model based on the method """
    if method == 'fedavg':
        model = SimpleCNN()
        if os.path.exists(state_dict_path_classifier): # FedAvg saves the whole model
             model.load_state_dict(torch.load(state_dict_path_classifier))
             print(f"Loaded full FedAvg model from {state_dict_path_classifier}")
        else:
             raise FileNotFoundError(f"FedAvg model state dict not found at {state_dict_path_classifier}")
        return model
    elif method == 'fed-pekd':
        model = SimpleCNN()
        # Load the server-trained classifier
        if os.path.exists(state_dict_path_classifier):
            model.classifier.load_state_dict(torch.load(state_dict_path_classifier))
            print(f"Loaded classifier from {state_dict_path_classifier}")
        else:
            raise FileNotFoundError(f"Classifier state dict not found at {state_dict_path_classifier}")

        # Load the feature extractor state dict from the saved client model
        if state_dict_path_extractor and os.path.exists(state_dict_path_extractor):
             client_state = torch.load(state_dict_path_extractor)
             # Filter the state dict for only feature_extractor parts
             # The keys in the saved client state dict will start with 'feature_extractor.'
             extractor_state_dict = {k.replace('feature_extractor.', ''): v
                                     for k, v in client_state.items()
                                     if k.startswith('feature_extractor.')}
             if extractor_state_dict:
                 model.feature_extractor.load_state_dict(extractor_state_dict)
                 print(f"Loaded feature extractor from {state_dict_path_extractor}")
             else:
                 print("Warning: No feature extractor keys found in client state dict. Using initial extractor.")
        else:
             print("Warning: Feature extractor state dict not found or specified. Using initial state.")

        return model
    else:
        raise ValueError("Unknown method")


def evaluate_and_plot_cm(model, test_loader, num_classes=10, save_path='results/confusion_matrix.png'):
    """ Evaluates the model and plots the confusion matrix """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy()) # Ensure tensors are moved to CPU
            all_labels.extend(labels.cpu().numpy())

    # Calculate overall accuracy
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    print(f"Overall Test Accuracy: {accuracy:.2f}%")

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))

    # FashionMNIST class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f}%)')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Trained Model and Generate Confusion Matrix")
    parser.add_argument('--method', type=str, required=True, choices=['fedavg', 'fed-pekd'], help='Method used for training')
    parser.add_argument('--classifier_path', type=str, required=True, help='Path to the saved state_dict for the classifier (or full model for FedAvg)')
    parser.add_argument('--extractor_path', type=str, default=None, help='(Fed-PEKD only) Path to saved full client model state_dict')
    parser.add_argument('--alpha', type=float, default=5.0, help='Alpha used for data splitting during training (for test loader consistency)')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name used for saving files (used to generate default save paths)')


    args = parser.parse_args()

    # --- **REMOVED THE PLACEHOLDER ERROR BLOCK** ---
    # --- Now we actually run the evaluation ---

    # Construct the default save path using the experiment name if provided
    save_filename = f'cm_{args.method}'
    if args.exp_name:
        save_filename += f'_{args.exp_name}'
    save_filename += '.png'
    save_path = os.path.join('results', save_filename)


    # Load the appropriate test dataset
    # Note: num_clients and max_samples don't affect the test set loading
    _, test_loader = get_data(num_clients=5, alpha=args.alpha, max_samples_per_client=1000)

    # Load the model state
    model = load_model(args.method, args.classifier_path, args.extractor_path)

    # Evaluate and plot
    evaluate_and_plot_cm(model, test_loader, save_path=save_path)