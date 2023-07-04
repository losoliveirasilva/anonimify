import os
import sys
import argparse
import pickle
from src.KnownFaceEmbedder import KnownFaceEmbedder
from src.KnownEmbeddingTrainer import KnownEmbeddingTrainer
from src.VideoAnonymizer import VideoAnonymizer

def create_file_structure():
    os.makedirs("outputs", exist_ok=True)

def get_args():
    parser = argparse.ArgumentParser(prog="Anonimify", description="Anonimize people in video")
    parser.add_argument('filename')
    parser.add_argument('-pi', '--preserved-identities', nargs='+', default=[], metavar="Preserved Identities")
    parser.add_argument('-l', '--layers', nargs='+', default=[256, 256, 512], type=int, metavar="Layers")
    parser.add_argument('-e', '--epochs', default=1, type=int, metavar="Epochs")
    parser.add_argument('-le', '--labeled-embeddings', metavar="Labeled Embeddings file")

    return parser

def parse_args(parser):
    args = parser.parse_args()
    return args

def create_labeled_embeddings(labeled_embeddings_filename):
    embedder = KnownFaceEmbedder()

    if labeled_embeddings_filename == None:
        embedder.process_images("datasets/")
        embedder.save_file(path=f"outputs/labeled_embeddings.pickle")
        return embedder

    embedder.load_embeddings_from_file(labeled_embeddings_filename)

    return embedder

def train_known_labeled_embeddings(embedder, layers, epochs):
    trainer = KnownEmbeddingTrainer(embedder.labeled_embeddings(), layers=len(layers), units_per_layer=layers, epochs=epochs)
    trainer.train(model_name="trained_known_labeled_embeddings")

def anonymize_video(video_filename, preserved_identities, embedder):
    anonymizer = VideoAnonymizer(embedder.labeled_embeddings(), "outputs/encoded_labels.pickle")
    regions = anonymizer.save_regions(video_filename, "outputs/trained_known_labeled_embeddings.h5", "outputs/face_regions.pickle")
    anonymizer.anonymize_regions(video_filename, f"anonymized.mp4", regions, preserved_identities)

def main():
    create_file_structure()
    parser = get_args()
    args = parse_args(parser)

    embedder = create_labeled_embeddings(args.labeled_embeddings)
    train_known_labeled_embeddings(embedder, args.layers, args.epochs)
    anonymize_video(args.filename, args.preserved_identities, embedder)

if __name__ == "__main__":
    main()
