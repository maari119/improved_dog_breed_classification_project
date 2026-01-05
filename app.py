import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import io
import timm
import sys
import json

# --- CONFIGURATION ---
MODEL_PATH = "resnet50_old.pth" 

# ==============================================================================
# 🚨 PASTE THE OUTPUT FROM YOUR NOTEBOOK HERE 🚨
# It must match the "final_list" exactly.
# ==============================================================================
CLASS_NAMES = [
    # Replace this list with the one you copied from the notebook
    'n02085620-Chihuahua', 'n02085782-Japanese_spaniel', 'n02085936-Maltese_dog', 
    'n02086079-Pekinese', 'n02086240-Shih-Tzu', 'n02086646-Blenheim_spaniel', 
    'n02086910-papillon', 'n02087046-toy_terrier', 'n02087394-Rhodesian_ridgeback', 
    'n02088094-Afghan_hound', 'n02088238-basset', 'n02088364-beagle', 
    'n02088466-bloodhound', 'n02088632-bluetick', 'n02089078-black-and-tan_coonhound', 
    'n02089867-Walker_hound', 'n02089973-English_foxhound', 'n02090379-redbone', 
    'n02090622-borzoi', 'n02090721-Irish_wolfhound', 'n02091032-Italian_greyhound', 
    'n02091134-whippet', 'n02091244-Ibizan_hound', 'n02091467-Norwegian_elkhound', 
    'n02091635-otterhound', 'n02091831-Saluki', 'n02092002-Scottish_deerhound', 
    'n02092339-Weimaraner', 'n02093256-Staffordshire_bullterrier', 
    'n02093428-American_Staffordshire_terrier', 'n02093647-Bedlington_terrier', 
    'n02093754-Border_terrier', 'n02093859-Kerry_blue_terrier', 
    'n02093991-Irish_terrier', 'n02094114-Norfolk_terrier', 'n02094258-Norwich_terrier', 
    'n02094433-Yorkshire_terrier', 'n02095314-wire-haired_fox_terrier', 
    'n02095570-Lakeland_terrier', 'n02095889-Sealyham_terrier', 'n02096051-Airedale', 
    'n02096177-cairn', 'n02096294-Australian_terrier', 'n02096437-Dandie_Dinmont', 
    'n02096585-Boston_bull', 'n02097047-miniature_schnauzer', 
    'n02097130-giant_schnauzer', 'n02097209-standard_schnauzer', 
    'n02097298-Scotch_terrier', 'n02097474-Tibetan_terrier', 
    'n02097658-silky_terrier', 'n02098105-soft-coated_wheaten_terrier', 
    'n02098286-West_Highland_white_terrier', 'n02098413-Lhasa', 
    'n02099267-flat-coated_retriever', 'n02099429-curly-coated_retriever', 
    'n02099601-golden_retriever', 'n02099712-Labrador_retriever', 
    'n02099849-Chesapeake_Bay_retriever', 'n02100236-German_short-haired_pointer', 
    'n02100583-vizsla', 'n02100735-English_setter', 'n02100877-Irish_setter', 
    'n02101006-Gordon_setter', 'n02101388-Brittany_spaniel', 'n02101556-clumber', 
    'n02102040-English_springer', 'n02102177-Welsh_springer_spaniel', 
    'n02102318-cocker_spaniel', 'n02102480-Sussex_spaniel', 
    'n02102973-Irish_water_spaniel', 'n02104029-kuvasz', 'n02104365-schipperke', 
    'n02105056-groenendael', 'n02105162-malinois', 'n02105251-briard', 
    'n02105412-kelpie', 'n02105505-komondor', 'n02105641-Old_English_sheepdog', 
    'n02105855-Shetland_sheepdog', 'n02106030-collie', 'n02106166-Border_collie', 
    'n02106382-Bouvier_des_Flandres', 'n02106550-Rottweiler', 
    'n02106662-German_shepherd', 'n02107142-Doberman', 'n02107312-miniature_pinscher', 
    'n02107574-Greater_Swiss_Mountain_dog', 'n02107683-Bernese_mountain_dog', 
    'n02107908-Appenzeller', 'n02108000-EntleBucher', 'n02108089-boxer', 
    'n02108422-bull_mastiff', 'n02108551-Tibetan_mastiff', 'n02108915-French_bulldog', 
    'n02109047-Great_Dane', 'n02109525-Saint_Bernard', 'n02109961-Eskimo_dog', 
    'n02110063-malamute', 'n02110185-Siberian_husky', 'n02110627-affenpinscher', 
    'n02110806-basenji', 'n02110958-pug', 'n02111129-Leonberg', 
    'n02111277-Newfoundland', 'n02111500-Great_Pyrenees', 'n02111889-Samoyed', 
    'n02112018-Pomeranian', 'n02112137-chow', 'n02112350-keeshond', 
    'n02112706-Brabancon_griffon', 'n02113023-Pembroke', 'n02113186-Cardigan', 
    'n02113624-toy_poodle', 'n02113712-miniature_poodle', 'n02113799-standard_poodle', 
    'n02113978-Mexican_hairless', 'n02115641-dingo', 'n02115913-dhole', 
    'n02116738-African_hunting_dog'
]

# Fallback
if not CLASS_NAMES:
    print("⚠️ WARNING: CLASS_NAMES is empty. Predictions will show ID numbers only.")
    CLASS_NAMES = [f"Class_{i}" for i in range(1000)]

# Initialize App
app = FastAPI(title="Dog Breed Classifier")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
LOADING_ERROR = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ROBUST MODEL LOADING ---
print(f"Attempting to load model from: {MODEL_PATH}")
try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    state_dict = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    elif isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    else:
        raise ValueError(f"Unknown model file format: {type(checkpoint)}")

    if state_dict:
        if 'fc.weight' in state_dict:
            num_classes_in_file = state_dict['fc.weight'].shape[0]
        elif 'classifier.weight' in state_dict:
             num_classes_in_file = state_dict['classifier.weight'].shape[0]
        else:
            num_classes_in_file = 120 # Default fallback

        print(f"Detected {num_classes_in_file} classes in model weights.")
        
        model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes_in_file)
        
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")

except Exception as e:
    LOADING_ERROR = str(e)
    print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
async def read_index():
    return FileResponse('index.html')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            k = min(5, len(probabilities))
            top_prob, top_catid = torch.topk(probabilities, k)

        results = []
        # DEBUG PRINT - Check your terminal to see what index the model is picking
        print(f"Top Index Predicted: {top_catid[0].item()}") 
        
        for i in range(k):
            idx = top_catid[i].item()
            score = top_prob[i].item()
            
            if idx < len(CLASS_NAMES):
                # Format: Remove the ID prefix (e.g., "n02099712-") for clean display
                raw_name = CLASS_NAMES[idx]
                if '-' in raw_name and raw_name.split('-')[0].startswith('n0'):
                     name = raw_name.split('-', 1)[1]
                else:
                    name = raw_name
            else:
                name = f"Class Index {idx}"
                
            results.append({
                "breed": str(name).replace("_", " ").title(),
                "confidence": f"{score * 100:.2f}%"
            })

        return {
            "prediction": results[0]["breed"],
            "confidence": results[0]["confidence"],
            "top_5": results
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))