import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_models():
    """Load the pre-trained CNN and ViT models."""
    
    # Load CNN model (ResNet50)
    cnn_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_features = cnn_model.fc.in_features
    cnn_model.fc = nn.Linear(num_features, 2)  # Binary classification: cat vs non-cat
    cnn_model.load_state_dict(torch.load(r'E:\UChicago\MachineLearning2\cnn-vit-cat-detection\best_models\cnn_model.pth', map_location=device))
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    
    # Load ViT model
    vit_model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    
    # Modify the head for binary classification - CORRECTED VERSION
    # The ViT has a nested structure: heads.head, not just heads
    vit_model.heads.head = nn.Linear(vit_model.heads.head.in_features, 2)  # Binary classification: cat vs non-cat
    vit_model.load_state_dict(torch.load(r'E:\UChicago\MachineLearning2\cnn-vit-cat-detection\best_models\vit_model.pth', map_location=device))
    vit_model = vit_model.to(device)
    vit_model.eval()
    
    return cnn_model, vit_model

def preprocess_image(image_path):
    """Preprocess the image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, image_tensor

def predict_image(models_dict, image_tensor):
    """Make predictions using both models."""
    results = {}
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            image_tensor = image_tensor.to(device)
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
            
            # CORRECTED Class mapping - based on actual model behavior
            # The models appear to have been trained with: 0 = Cat, 1 = Non-Cat
            class_names = ['Cat', 'Non-Cat']  # Swapped to match actual model behavior
            predicted_label = class_names[predicted_class]
            
            results[model_name] = {
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': {
                    'Cat': probabilities[0][0].item(),      # Index 0 = Cat (corrected)
                    'Non-Cat': probabilities[0][1].item()   # Index 1 = Non-Cat (corrected)
                }
            }
            
            # Debug info for troubleshooting
            print(f"{model_name} - Raw output: {outputs.cpu().numpy()}")
            print(f"{model_name} - Probabilities: Cat={probabilities[0][0].item():.3f}, Non-Cat={probabilities[0][1].item():.3f}")
    
    return results

def visualize_results(image, results, image_path):
    """Visualize the image and prediction results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Display original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # CNN Results
    cnn_result = results['CNN']
    axes[1].bar(['Cat', 'Non-Cat'], 
                [cnn_result['probabilities']['Cat'], cnn_result['probabilities']['Non-Cat']],
                color=['green', 'red'])
    axes[1].set_title(f'CNN Prediction: {cnn_result["predicted_label"]}\n'
                     f'Confidence: {cnn_result["confidence"]:.3f}')
    axes[1].set_ylabel('Probability')
    axes[1].set_ylim(0, 1)
    
    # ViT Results
    vit_result = results['ViT']
    axes[2].bar(['Cat', 'Non-Cat'], 
                [vit_result['probabilities']['Cat'], vit_result['probabilities']['Non-Cat']],
                color=['green', 'red'])
    axes[2].set_title(f'ViT Prediction: {vit_result["predicted_label"]}\n'
                     f'Confidence: {vit_result["confidence"]:.3f}')
    axes[2].set_ylabel('Probability')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Path to your test image (replace with your image path)
    image_path = input("Enter the path to your test image: ").strip()
    
    try:
        print("Loading models...")
        cnn_model, vit_model = load_models()
        models_dict = {'CNN': cnn_model, 'ViT': vit_model}
        
        print("Preprocessing image...")
        image, image_tensor = preprocess_image(image_path)
        
        print("Making predictions...")
        results = predict_image(models_dict, image_tensor)
        
        # Print results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        
        for model_name, result in results.items():
            print(f"\n{model_name} Model:")
            print(f"  Prediction: {result['predicted_label']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Probabilities:")
            print(f"    Cat: {result['probabilities']['Cat']:.3f}")
            print(f"    Non-Cat: {result['probabilities']['Non-Cat']:.3f}")
        
        # Compare models
        print(f"\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        cnn_pred = results['CNN']['predicted_label']
        vit_pred = results['ViT']['predicted_label']
        
        if cnn_pred == vit_pred:
            print(f"✅ Both models agree: {cnn_pred}")
            print(f"CNN confidence: {results['CNN']['confidence']:.3f}")
            print(f"ViT confidence: {results['ViT']['confidence']:.3f}")
        else:
            print(f"❌ Models disagree!")
            print(f"CNN predicts: {cnn_pred} (confidence: {results['CNN']['confidence']:.3f})")
            print(f"ViT predicts: {vit_pred} (confidence: {results['ViT']['confidence']:.3f})")
        
        # Visualize results
        print("\nGenerating visualization...")
        visualize_results(image, results, image_path)
        print("Results saved as 'prediction_results.png'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        print("Make sure the model files (cnn_model.pth, vit_model.pth) are in the current directory.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()