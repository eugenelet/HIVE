import torch
import torchvision
import os
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from functools import partial
from tqdm import tqdm
from llava_next.model.builder import load_pretrained_model
from llava_next.mm_utils import get_model_name_from_path, expand2square
from transformers import CLIPModel, CLIPImageProcessor

model_path = "./checkpoints_out/finetune/Finetune_llava1.5_qwen2-1.5B_CLIP336_promptaware_clspatch/checkpoint-11540"
model_base = None
model_name = get_model_name_from_path(model_path)
data_path = "../Multimodal_Hyperminer/data/"

# tokenizer, model, image_processor, max_length = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", attn_implementation="sdpa")

# clip_image_tower = model.get_model().get_vision_tower()
# del model
# del tokenizer

clip_image_tower = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", device_map="cuda")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

@torch.no_grad()
def get_image_embedding(images):
    images = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].to('cuda')
    image_feature = clip_image_tower.get_image_features(images) # [:, 0]  # cls token
    return image_feature

def pil_collate_fn(batch):
    images, labels = zip(*batch)  # Separate images and labels
    return list(images), torch.tensor(labels)  # Return images as a list and labels as a tensor

pad = partial(expand2square, background_color=tuple(int(x * 255) for x in image_processor.image_mean))

trainset = torchvision.datasets.CIFAR10(
    root=data_path, train=True, download=True, transform=pad)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=10, shuffle=False, num_workers=10, collate_fn=pil_collate_fn)

testset = torchvision.datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=pad)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=10, shuffle=False, num_workers=10, collate_fn=pil_collate_fn)

train_feats = []
train_labels = []
test_feats = []
test_labels = []

for batch_data, batch_labels in tqdm(trainloader):
    image_embedding = get_image_embedding(batch_data)
    train_feats.append(image_embedding.detach().cpu().numpy())
    train_labels.append(batch_labels.detach().cpu().numpy())
    
for batch_data, batch_labels in tqdm(testloader):
    image_embedding = get_image_embedding(batch_data)
    test_feats.append(image_embedding.detach().cpu().numpy())
    test_labels.append(batch_labels.detach().cpu().numpy())
    
train_feats = np.concatenate(train_feats, axis=0)
train_labels = np.concatenate(train_labels, axis=0)
test_feats = np.concatenate(test_feats, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

clf = SVC(random_state=0)
clf.fit(train_feats, train_labels)
acc = accuracy_score(clf.predict(test_feats), test_labels)

print(f"Accuracy: {acc}")
    
