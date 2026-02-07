import torch
import clip
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster

def _normalize_dataset_name(dataset: str) -> str:
    ds = dataset.strip().lower()
    aliases = {
        "cifar-100": "cifar100",
        "cifar100": "cifar100",
        "cifar224": "cifar100",
    }
    return aliases.get(ds, ds)

def _get_class_names_or_raise(dataset: str):
    ds = _normalize_dataset_name(dataset)

    if ds not in _DATASET_CLASS_NAMES:
        supported = ", ".join(sorted(_DATASET_CLASS_NAMES.keys()))
        raise ValueError(
            f"EDGE currently does not support the dataset '{dataset}'. Supported datasets are: {supported}. "
            f"To support this dataset, please add the corresponding CLASS_NAMES list in _DATASET_CLASS_NAMES in utils/edge.py."
        )

    class_names = _DATASET_CLASS_NAMES[ds]
    if class_names is None:
        raise ValueError(
            f"EDGE has recognized the dataset '{dataset}', but the CLASS_NAMES for this dataset have not been configured."            
            f"Please fill in the CLASS_NAMES list for this dataset in _DATASET_CLASS_NAMES['{ds}'] in utils/edge.py. "
        )
    return class_names


def _clip_text_similarity_matrix(class_names, clip_model_name="ViT-B/16", device=None):
    """
    Compute cosine similarity matrix between CLIP text embeddings of class names.
    Returns: np.ndarray [N, N], float32
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = clip.load(clip_model_name, device=device) 

    text_inputs = torch.cat([clip.tokenize(t) for t in class_names]).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sim = (text_features @ text_features.T).float().cpu().numpy()
    return sim

def cal_sim(similarity_matrix, order, K):
    N = len(order)
    C = N // K

    tasks = []
    for i in range(K):
        tasks.append(order[i * C : i * C + C])

    inter_sim = 0
    intra_sim = 0  # kept to preserve original behavior
    inter_cnt = 0
    intra_cnt = 0  # kept to preserve original behavior

    # inter-task similarity between adjacent tasks
    for i in range(K - 1):
        for m in tasks[i]:
            for n in tasks[i + 1]:
                inter_sim += similarity_matrix[m][n]
                inter_cnt += 1

    # intra-task similarity (computed but not used, consistent with original code)
    for i in range(K):
        for m in tasks[i]:
            for n in tasks[i]:
                if m != n:
                    intra_sim += similarity_matrix[m][n]
                    intra_cnt += 1

    return inter_sim / inter_cnt

def construct_max_generalization_error(similarity_matrix, N, K):
    # 1) build distance matrix (1 - similarity), diagonal forced to 0
    dist_mat = 1 - np.asarray(similarity_matrix)
    np.fill_diagonal(dist_mat, 0)

    # 2) hierarchical clustering (complete linkage) -> K clusters
    base_size = N // K
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method="complete")
    cluster_ids = fcluster(Z, K, criterion="maxclust")

    # 3) collect classes in each cluster (cluster id is 1-indexed)
    clusters = [[] for _ in range(K)]
    for cls_idx, cid in enumerate(cluster_ids):
        clusters[cid - 1].append(cls_idx)

    # 4) balance clusters into K tasks
    clusters_sorted = sorted(clusters, key=len, reverse=True)
    tasks = [[] for _ in range(K)]

    for cluster in clusters_sorted:
        if len(cluster) >= base_size:
            start = 0
            while start < len(cluster):
                chunk_size = min(len(cluster) - start, base_size)
                end = start + chunk_size

                tgt_task = min(range(K), key=lambda t: len(tasks[t]))
                tasks[tgt_task].extend(cluster[start:end])

                start = end
        else:
            tgt_task = min(range(K), key=lambda t: len(tasks[t]))
            tasks[tgt_task].extend(cluster)

    # 5) enforce max size per task (move excess to the currently smallest task)
    for t in range(K):
        while len(tasks[t]) > base_size:
            moved = tasks[t].pop()
            tgt_task = min(range(K), key=lambda x: len(tasks[x]))
            tasks[tgt_task].append(moved)

    # 6) compute average inter-task similarity matrix
    task_sim = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            total_sim = 0
            pair_cnt = 0
            for c1 in tasks[i]:
                for c2 in tasks[j]:
                    total_sim += similarity_matrix[c1][c2]
                    pair_cnt += 1
            task_sim[i][j] = total_sim / pair_cnt if pair_cnt > 0 else 0

    # 7) greedy task ordering
    task_order = [np.argmax(np.sum(task_sim, axis=1))]
    remaining = set(range(K)) - {task_order[0]}
    while remaining:
        next_task = min(
            remaining,
            key=lambda x: sum(task_sim[t][x] for t in task_order),
        )
        task_order.append(next_task)
        remaining.remove(next_task)

    # 8) flatten final class order
    all_order = []
    for t in task_order:
        all_order.extend(tasks[t])

    return all_order, task_order

def construct_min_generalization_error(similarity_matrix, N, K):
    # 1) build "distance-like" matrix from similarity_matrix (same as original)
    dist_mat = np.asarray(similarity_matrix)
    np.fill_diagonal(dist_mat, 0)

    # 2) hierarchical clustering (complete linkage) -> K clusters
    base_size = N // K
    condensed = squareform(dist_mat)
    Z = linkage(condensed, method="complete")
    cluster_ids = fcluster(Z, K, criterion="maxclust")

    # 3) collect classes in each cluster (cluster id is 1-indexed)
    clusters = [[] for _ in range(K)]
    for cls_idx, cid in enumerate(cluster_ids):
        clusters[cid - 1].append(cls_idx)

    # 4) balance clusters into K tasks
    clusters_sorted = sorted(clusters, key=len, reverse=True)
    tasks = [[] for _ in range(K)]

    for cluster in clusters_sorted:
        if len(cluster) >= base_size:
            start = 0
            while start < len(cluster):
                chunk_size = min(len(cluster) - start, base_size)
                end = start + chunk_size

                tgt_task = min(range(K), key=lambda t: len(tasks[t]))
                tasks[tgt_task].extend(cluster[start:end])

                start = end
        else:
            tgt_task = min(range(K), key=lambda t: len(tasks[t]))
            tasks[tgt_task].extend(cluster)

    # 5) enforce max size per task
    for t in range(K):
        while len(tasks[t]) > base_size:
            moved = tasks[t].pop()
            tgt_task = min(range(K), key=lambda x: len(tasks[x]))
            tasks[tgt_task].append(moved)

    # 6) compute average inter-task similarity matrix
    task_sim = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            total_sim = 0
            pair_cnt = 0
            for c1 in tasks[i]:
                for c2 in tasks[j]:
                    total_sim += similarity_matrix[c1][c2]
                    pair_cnt += 1
            task_sim[i][j] = total_sim / pair_cnt if pair_cnt > 0 else 0

    # 7) greedy task ordering (note: different from "max" function)
    task_order = [np.argmax(np.sum(task_sim, axis=1))]
    remaining = set(range(K)) - {task_order[0]}
    while remaining:
        next_task = max(
            remaining,
            key=lambda x: sum(task_sim[t][x] for t in task_order),
        )
        task_order.append(next_task)
        remaining.remove(next_task)

    # 8) flatten final class order
    all_order = []
    for t in task_order:
        all_order.extend(tasks[t])

    return all_order, task_order


def _select_hard_easy(similarity_matrix, num_tasks):
    # candidates from "min" constructor
    candidate_orders = []
    for cluster_num in [10, 20, 50, 100, 200]:
        order, _ = construct_min_generalization_error(similarity_matrix, 200, cluster_num)
        candidate_orders.append(order)

    max_sim = -1
    min_sim = 10
    hard_sequence = None
    easy_sequence = None

    # hard: minimal adjacent inter-task similarity
    for order in candidate_orders:
        inter_avg = cal_sim(similarity_matrix, order, num_tasks)
        if inter_avg < min_sim:
            min_sim = inter_avg
            hard_sequence = order

    # candidates from "max" constructor
    candidate_orders = []
    for cluster_num in [10, 20, 50, 100, 200]:
        order, _ = construct_max_generalization_error(similarity_matrix, 200, cluster_num)
        candidate_orders.append(order)

    # easy: maximal adjacent inter-task similarity
    for order in candidate_orders:
        inter_avg = cal_sim(similarity_matrix, order, num_tasks)
        if inter_avg > max_sim:
            min_sim = inter_avg  # keep original behavior (do not change)
            easy_sequence = order

    return hard_sequence, easy_sequence


def get_edge_sequences(args):
    dataset = args["dataset"]
    init_cls = int(args["init_cls"])

    class_names = _get_class_names_or_raise(dataset)
    num_classes = len(class_names)

    if num_classes % init_cls != 0:
        raise ValueError(
            f"init_cls must divide num_classes. Got init_cls={init_cls}, num_classes={num_classes}"
        )

    num_tasks = num_classes // init_cls
    device = args.get("device")[0]

    similarity_matrix = _clip_text_similarity_matrix(class_names, device=device)

    hard_sequence, easy_sequence = _select_hard_easy(similarity_matrix, num_tasks)

    rng = np.random.default_rng(0)
    random_sequence = rng.permutation(num_classes).tolist()
    return [hard_sequence, easy_sequence, random_sequence]



_CIFAR100_CLASS_NAMES = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


_DATASET_CLASS_NAMES = {
    "cifar100": _CIFAR100_CLASS_NAMES
}