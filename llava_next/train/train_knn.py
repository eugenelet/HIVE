import os
import argparse
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers.models.siglip.modeling_siglip import SiglipVisionTransformer
from transformers.models.clip.modeling_clip import CLIPVisionTransformer
import datasets
from joblib import parallel_backend

from llava_next.mm_utils import get_model_name_from_path
from llava_next.model.builder import load_pretrained_model

# Use tf32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Define Lazy Preprocessing
class LazyPreprocessor:
    def __init__(self, image_processor, transform=None, dataset_name=None):
        self.image_processor = image_processor
        if transform is None:
            self.transform = transforms.Compose([])
        else:
            self.transform = transform
        self.dataset_name = dataset_name

    def __call__(self, examples):
        try:
            pixel_values = [
                self.image_processor(self.transform(
                    image.convert("RGB")
                ))["pixel_values"][0] for image in examples["img"]
            ]
        except:
            pixel_values = [
                self.image_processor(self.transform(
                    image.convert("RGB")
                ))["pixel_values"][0] for image in examples["image"]
            ]

        if "Caltech-256" in self.dataset_name:
            labels = [label - 1 for label in examples["label"]]
        else:
            labels = examples["label"]

        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# Load model and datasets using existing APIs
def load_existing_model_and_datasets(parser):
    image_processor = AutoImageProcessor.from_pretrained(parser.vision_tower)
    dataset = datasets.load_dataset(parser.dataset_name, cache_dir=os.path.join(parser.data_path, parser.dataset_name))
    lazy_preprocessor = LazyPreprocessor(image_processor, None, parser.dataset_name)  # No transformation for evaluation
    train_dataset = dataset['train'].with_transform(lazy_preprocessor)
    test_dataset = dataset.get('validation', dataset.get('valid', dataset.get('test', None))).with_transform(lazy_preprocessor)
    if "label" not in dataset["train"].features:  # for cifar100
        train_dataset = train_dataset.rename_column("fine_label", "label")
        train_dataset = train_dataset.remove_columns("coarse_label")
        test_dataset = test_dataset.rename_column("fine_label", "label")
        test_dataset = test_dataset.remove_columns("coarse_label")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=parser.batch_size, num_workers=parser.num_workers, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=parser.batch_size, num_workers=parser.num_workers, shuffle=False)

    model = AutoModelForImageClassification.from_pretrained(
        parser.vision_tower,
        num_labels=1,
        device_map="cuda"
    )
    model = model.vision_model
    del model.encoder.layers[-1:] 
    model.post_layernorm = nn.Identity()

    if parser.model_name_or_path is not None:
        model_name = get_model_name_from_path(parser.model_name_or_path)
        _, LLM_model, image_processor, _ = load_pretrained_model(parser.model_name_or_path, parser.model_base, model_name, device_map="cpu", attn_implementation="sdpa", torch_dtype=torch.float16)
        
        model.load_state_dict(LLM_model.get_model().get_vision_tower().vision_tower.vision_model.state_dict(), strict=False)

        del LLM_model

    model = torch.compile(model)

    return model, train_dataloader, test_dataloader

# Extract patch embeddings and average them to a single vector
def extract_patch_embeddings(model, dataloader):
    model.eval()
    embeddings, labels = [], []

    for sample in tqdm(dataloader, desc="Extracting patch embeddings"):
        pixel_values = torch.as_tensor(sample['pixel_values'], device="cuda")
        with torch.no_grad():
            outputs = model(pixel_values)
            patch_embeddings = outputs[0]
            if isinstance(model, CLIPVisionTransformer):
                patch_embeddings = patch_embeddings[:, 1:, :]  # Exclude CLS token
            sequence_output = torch.mean(patch_embeddings, dim=1)  # Average pooling of patch embeddings
        embeddings.append(sequence_output.cpu().numpy())
        labels.append(sample['labels'])

    # Concatenate all collected batches into single arrays
    embeddings = np.vstack(embeddings)
    labels = np.concatenate(labels)

    return embeddings, labels

# Evaluate with KNN and Linear Probe using parallelism for faster computation
def evaluate_knn(train_emb, train_labels, test_emb, test_labels):
    with parallel_backend('threading', n_jobs=-1):
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        knn.fit(train_emb, train_labels)
        preds = knn.predict(test_emb)
    acc = accuracy_score(test_labels, preds)
    print("KNN Accuracy:", acc)
    return acc

def evaluate_linear_probe(train_emb, train_labels, test_emb, test_labels):
    with parallel_backend('threading', n_jobs=-1):
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(train_emb, train_labels)
        preds = clf.predict(test_emb)
    acc = accuracy_score(test_labels, preds)
    print("Linear Probe Accuracy:", acc)
    return acc

# Main evaluation function
def run_evaluation(model, train_dataloader, test_dataloader):
    train_emb, train_labels = extract_patch_embeddings(model, train_dataloader)
    test_emb, test_labels = extract_patch_embeddings(model, test_dataloader)

    print("Evaluate KNN")
    knn_acc = evaluate_knn(train_emb, train_labels, test_emb, test_labels)
    print("Evaluate Linear Probe")
    linear_acc = evaluate_linear_probe(train_emb, train_labels, test_emb, test_labels)

    print("KNN", knn_acc)
    print("Linear Probe", linear_acc)

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_tower", type=str, default="google/siglip-so400m-patch14-384")
    parser.add_argument("--dataset_name", type=str, default="ILSVRC/imagenet-1k")
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    
    parser = parser.parse_args()

    model, train_dataloader, test_dataloader = load_existing_model_and_datasets(parser)
    run_evaluation(model, train_dataloader, test_dataloader)
