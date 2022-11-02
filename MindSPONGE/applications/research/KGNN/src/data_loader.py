"""data_loader"""
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from src.utils import format_filename, pickle_dump
from src.model_utils.config import config

SEPARATOR = {'drug': '\t', 'kegg': '\t'}

DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'
TRAIN_DATA_TEMPLATE = '{dataset}_train.npy'
DEV_DATA_TEMPLATE = '{dataset}_dev.npy'
TEST_DATA_TEMPLATE = '{dataset}_test.npy'

DRUG_EXAMPLE = '{dataset}_{type}_examples.npy'


def load_data(dataset: str, data_type: str):
    if data_type == 'train':
        return np.load(format_filename(config.PROCESSED_DATA_DIR, TRAIN_DATA_TEMPLATE, dataset=dataset))
    if data_type == 'dev':
        return np.load(format_filename(config.PROCESSED_DATA_DIR, DEV_DATA_TEMPLATE, dataset=dataset))
    if data_type == 'test':
        return np.load(format_filename(config.PROCESSED_DATA_DIR, TEST_DATA_TEMPLATE, dataset=dataset))
    raise ValueError('`data_type` not understood: {data_type}')


def read_entity2id_file(file_path: str, drug_vocab: dict, entity_vocab: dict):
    """read_entity2id_file"""
    print(f'Logging Info - Reading entity2id file: {file_path}')
    assert not drug_vocab and not entity_vocab
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            _, entity = line.strip().split('\t')
            drug_vocab[entity] = len(drug_vocab)
            entity_vocab[entity] = len(entity_vocab)


def read_example_file(file_path: str, separator: str, drug_vocab: dict):
    """read_example_file"""
    print(f'Logging Info - Reading example file: {file_path}')
    assert drug_vocab
    examples = []
    with open(file_path, encoding='utf8') as reader:
        for _, line in enumerate(reader):
            d1, d2, flag = line.strip().split(separator)[:3]
            if d1 not in drug_vocab or d2 not in drug_vocab:
                continue
            if d1 in drug_vocab and d2 in drug_vocab:
                examples.append([drug_vocab[d1], drug_vocab[d2], int(flag)])

    examples_matrix = np.array(examples)
    print(f'size of example: {examples_matrix.shape}')
    return examples_matrix


def read_kg(file_path: str, entity_vocab: dict, relation_vocab: dict, neighbor_sample_size: int):
    """read_kg"""
    print(f'Logging Info - Reading kg file: {file_path}')

    kg = defaultdict(list)
    with open(file_path, encoding='utf8') as reader:
        count = 0
        for line in reader:
            if count == 0:
                count += 1
                continue
            head, tail, relation = line.strip().split(' ')

            if head not in entity_vocab:
                entity_vocab[head] = len(entity_vocab)
            if tail not in entity_vocab:
                entity_vocab[tail] = len(entity_vocab)
            if relation not in relation_vocab:
                relation_vocab[relation] = len(relation_vocab)

            # undirected graph
            kg[entity_vocab[head]].append((entity_vocab[tail], relation_vocab[relation]))
            kg[entity_vocab[tail]].append((entity_vocab[head], relation_vocab[relation]))
    print(f'Logging Info - num of entities: {len(entity_vocab)}, '
          f'num of relations: {len(relation_vocab)}')

    print('Logging Info - Constructing adjacency matrix...')
    n_entity = len(entity_vocab)
    adj_entity = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)
    adj_relation = np.zeros(shape=(n_entity, neighbor_sample_size), dtype=np.int64)

    np.random.seed(0)
    for entity_id in range(n_entity):
        all_neighbors = kg[entity_id]
        n_neighbor = len(all_neighbors)
        sample_indices = np.random.choice(
            n_neighbor,
            neighbor_sample_size,
            replace=n_neighbor < neighbor_sample_size
        )

        adj_entity[entity_id] = np.array([all_neighbors[i][0] for i in sample_indices])
        adj_relation[entity_id] = np.array([all_neighbors[i][1] for i in sample_indices])

    return adj_entity, adj_relation


def process_data(dataset: str, neighbor_sample_size: int):
    """
    Convert text data to numpy data.
    Perform neighbor sampling.
    Compute the vocabulary of drug, entity and relation.
    Split dataset into train and test subsets.

    Args:
        dataset: Dataset name
        neighbor_sample_size: Number of sampling neighbors
    """
    drug_vocab = {}
    entity_vocab = {}
    relation_vocab = {}

    read_entity2id_file(config.ENTITY2ID_FILE, drug_vocab, entity_vocab)

    pickle_dump(format_filename(config.PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset), drug_vocab)
    pickle_dump(format_filename(config.PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset), entity_vocab)

    train_examples_file = format_filename(config.PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=dataset, type="train")
    test_examples_file = format_filename(config.PROCESSED_DATA_DIR, DRUG_EXAMPLE, dataset=dataset, type="test")
    examples = read_example_file(config.EXAMPLE_FILE, SEPARATOR.get(dataset), drug_vocab)
    x = examples[:, :2]
    y = examples[:, 2:3]
    train_data_x, test_data_x, train_y, test_y = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
    train_data = np.c_[train_data_x, train_y]
    test_data = np.c_[test_data_x, test_y]
    np.save(train_examples_file, train_data)
    np.save(test_examples_file, test_data)

    adj_entity, adj_relation = read_kg(config.KG_FILE, entity_vocab, relation_vocab,
                                       neighbor_sample_size)

    pickle_dump(format_filename(config.PROCESSED_DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=dataset),
                drug_vocab)
    pickle_dump(format_filename(config.PROCESSED_DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=dataset),
                entity_vocab)
    pickle_dump(format_filename(config.PROCESSED_DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=dataset),
                relation_vocab)
    adj_entity_file = format_filename(config.PROCESSED_DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=dataset)
    np.save(adj_entity_file, adj_entity)
    print('Logging Info - Saved:', adj_entity_file)

    adj_relation_file = format_filename(config.PROCESSED_DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=dataset)
    np.save(adj_relation_file, adj_relation)
    print('Logging Info - Saved:', adj_relation_file)
