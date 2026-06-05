
import streamlit as st
import torch
import torchvision
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Constants ---
MODEL_PATH = 'model.pth'
CLASSES_PATH = 'class_names.json'
CONFIDENCE_THRESHOLD = 0.7 # Minimum confidence score to display a detection

# --- Helper Functions ---

@st.cache_resource
def load_class_names():
    """Loads the class ID to label mapping from a JSON file."""
    try:
        with open(CLASSES_PATH, 'r') as f:
            id_to_label = json.load(f)
        # Convert keys from string to int as they are stored as strings in JSON
        id_to_label = {int(k): v for k, v in id_to_label.items()}
        return id_to_label
    except FileNotFoundError:
        st.error(f"Error: {CLASSES_PATH} not found. Please ensure it's in the same directory as app.py.")
        return None

@st.cache_resource
def get_mask_rcnn_model(num_classes):
    """Initializes a Mask R-CNN model with a custom head for specified number of classes."""
    # Load an instance segmentation model, without pre-trained weights
    # as we will load our custom-trained weights.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    # num_classes includes the background class
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model

@st.cache_resource
def load_model(num_classes):
    """Loads the trained Mask R-CNN model weights."""
    model = get_mask_rcnn_model(num_classes)
    try:
        # Load the state dictionary, mapping to CPU for broader compatibility
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval() # Set model to evaluation mode
        return model
    except FileNotFoundError:
        st.error(f"Error: Model weights file '{MODEL_PATH}' not found. Please ensure it's in the same directory as app.py.")
        return None
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

def visualize_predictions(image_pil, predictions, id_to_label_map, threshold):
    """Visualizes bounding boxes, labels, confidence scores, and masks on an image."""
    image_np = np.array(image_pil) # Convert PIL Image to NumPy array for matplotlib
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image_np)
    ax.set_axis_off() # Hide axes for cleaner display

    # Define a color map for different classes
    colors = plt.cm.get_cmap('hsv', len(id_to_label_map) + 1) 

    for box, label_id, score, mask_logits in zip(predictions['boxes'], predictions['labels'], predictions['scores'], predictions['masks']):
        if score > threshold:
            xmin, ymin, xmax, ymax = box.cpu().numpy().astype(int)
            label_name = id_to_label_map.get(label_id.item(), "Unknown") # Get label name from map

            color = colors(label_id.item() % colors.N) # Assign a color based on label ID

            # Draw bounding box rectangle
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            # Display label and score text
            ax.text(xmin, ymin - 10, f"{label_name}: {score:.2f}",
                    bbox=dict(facecolor=color, alpha=0.7),
                    fontsize=10, color='white')
            
            # Draw mask (alpha blended)
            # Select the mask for the predicted class (shape [H, W])
            mask_for_label = mask_logits[0]
            # Apply sigmoid to convert logits to probabilities, then threshold
            mask_probs = mask_for_label.sigmoid().cpu().numpy()
            binary_mask = (mask_probs > 0.5).astype(np.uint8) # Binary mask (0 or 1)

            # Create a colored overlay from the binary mask
            colored_mask = np.zeros_like(image_np, dtype=np.uint8)
            color_rgb = (np.array(color[:3]) * 255).astype(np.uint8) # Convert matplotlib color to 0-255 RGB
            colored_mask[binary_mask > 0] = color_rgb

            # Apply transparency (alpha blending) only where the mask is present
            alpha_blend = 0.5
            ax.imshow(colored_mask, alpha=alpha_blend * binary_mask) # Multiply alpha by binary_mask

     # Convert the matplotlib figure to a PIL Image for display in Streamlit
       fig.canvas.draw()

    buf = np.asarray(fig.canvas.buffer_rgba())
    img_viz = Image.fromarray(buf[:, :, :3])

    plt.close(fig)
    return img_viz


# --- Streamlit Application Layout ---
st.title("Face Mask Detection with Mask R-CNN")
st.markdown("Upload an image to detect faces with and without masks using a trained Mask R-CNN model.")

# Load class names and model once using caching
id_to_label_map = load_class_names()
if id_to_label_map is None:
    st.stop() # Stop if class names couldn't be loaded

# Calculate num_classes for the model (actual classes + 1 for background)
num_classes = len(id_to_label_map) + 1
model = load_model(num_classes)
if model is None:
    st.stop() # Stop if model couldn't be loaded

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image using PIL
    image_pil = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image_pil, caption="Original Uploaded Image", use_column_width=True)

    # Preprocess the image for model input
    transform = torchvision.transforms.ToTensor() # Converts PIL Image to tensor and normalizes to [0,1]
    image_tensor = transform(image_pil)

    # Perform inference
    # Determine the device (CPU or GPU) for inference
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device) # Move model to the selected device
    image_tensor = image_tensor.to(device) # Move image tensor to the selected device

    with torch.no_grad(): # Disable gradient calculations for inference
        predictions = model([image_tensor])
        predictions = predictions[0] # Get predictions for the single image

    st.subheader("Detected Objects")
    # Filter predictions based on the confidence threshold
    high_confidence_indices = predictions['scores'] > CONFIDENCE_THRESHOLD
    filtered_predictions = {
        'boxes': predictions['boxes'][high_confidence_indices],
        'labels': predictions['labels'][high_confidence_indices],
        'scores': predictions['scores'][high_confidence_indices],
        'masks': predictions['masks'][high_confidence_indices]
    }

    if len(filtered_predictions['boxes']) == 0:
        st.write("No objects detected with a confidence score above the threshold.")
    else:
        # Visualize the filtered predictions on the image
        processed_image_pil = visualize_predictions(image_pil, filtered_predictions, id_to_label_map, CONFIDENCE_THRESHOLD)
        st.image(processed_image_pil, caption="Image with Detections", use_column_width=True)

        st.subheader("Raw Prediction Data (Confidence > 0.7)")
        # Display prediction details in a DataFrame
        pred_df = []
        for i, (box, label_id, score) in enumerate(zip(filtered_predictions['boxes'], filtered_predictions['labels'], filtered_predictions['scores'])):
            xmin, ymin, xmax, ymax = box.cpu().numpy().astype(int)
            label_name = id_to_label_map.get(label_id.item(), "Unknown")
            pred_df.append({
                "Label": label_name,
                "Confidence": f"{score:.2f}",
                "Bounding Box": f"[{xmin}, {ymin}, {xmax}, {ymax}]"
            })
        
        if pred_df:
            import pandas as pd
            st.dataframe(pd.DataFrame(pred_df))
        else:
            st.write("No detections to display in raw data.")
