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
        "cub-200": "cub200",
        "cub200": "cub200",
        "cub": "cub200",
        "imagenetr": "imagenet-r",
        "imageneta": "imagenet-a",
        "imagenet_r": "imagenet-r",
        "imagenet-r": "imagenet-r",
        "imagenet_a": "imagenet-a",
        "imagenet-a": "imagenet-a",
        "omni-benchmark": "omnibenchmark",
        "omnibenchmark": "omnibenchmark",
        "vtab": "vtab",
        "objectnet": "objectnet",
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

_CUB_CLASS_NAMES = ['Black footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet', 'Rhinoceros Auklet', 'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting', 'Cardinal', 'Spotted Catbird', 'Gray Catbird', 'Yellow breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird', 'Shiny Cowbird', 'Brown Creeper', 'American Crow', 'Fish Crow', 'Black billed Cuckoo', 'Mangrove Cuckoo', 'Yellow billed Cuckoo', 'Gray crowned Rosy Finch', 'Purple Finch', 'Northern Flicker', 'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive sided Flycatcher', 'Scissor tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow bellied Flycatcher', 'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied billed Grebe', 'Western Grebe', 'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous winged Gull', 'Heermann Gull', 'Herring Gull', 'Ivory Gull', 'Ring billed Gull', 'Slaty backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear', 'Long tailed Jaeger', 'Pomarine Jaeger', 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 'White breasted Kingfisher', 'Red legged Kittiwake', 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser', 'Red breasted Merganser', 'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird', 'Brown Pelican', 'White Pelican', 'Western Wood Pewee', 'Sayornis', 'American Pipit', 'Whip poor Will', 'Horned Puffin', 'Common Raven', 'White necked Raven', 'American Redstart', 'Geococcyx', 'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 'Black throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay colored Sparrow', 'House Sparrow', 'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp tailed Sparrow', 'Savannah Sparrow', 'Seaside Sparrow', 'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White crowned Sparrow', 'White throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow', 'Cliff Swallow', 'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern', 'Least Tern', 'Green tailed Towhee', 'Brown Thrasher', 'Sage Thrasher', 'Black capped Vireo', 'Blue headed Vireo', 'Philadelphia Vireo', 'Red eyed Vireo', 'Warbling Vireo', 'White eyed Vireo', 'Yellow throated Vireo', 'Bay breasted Warbler', 'Black and white Warbler', 'Black throated Blue Warbler', 'Blue winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler', 'Chestnut sided Warbler', 'Golden winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American Three toed Woodpecker', 'Pileated Woodpecker', 'Red bellied Woodpecker', 'Red cockaded Woodpecker', 'Red headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren', 'Winter Wren', 'Common Yellowthroat']

_IMAGENETR_CLASS_NAMES = ['goldfish, Carassius auratus', 'great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias', 'hammerhead, hammerhead shark', 'stingray', 'hen', 'ostrich, Struthio camelus', 'goldfinch, Carduelis carduelis', 'junco, snowbird', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'common newt, Triturus vulgaris', 'axolotl, mud puppy, Ambystoma mexicanum', 'tree frog, tree-frog', 'common iguana, iguana, Iguana iguana', 'African chameleon, Chamaeleo chamaeleon', 'Indian cobra, Naja naja', 'scorpion', 'tarantula', 'centipede', 'peacock', 'lorikeet', 'hummingbird', 'toucan', 'drake', 'goose', 'black swan, Cygnus atratus', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'jellyfish', 'snail', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'hermit crab', 'flamingo', 'American egret, great white heron, Egretta albus', 'pelican', 'king penguin, Aptenodytes patagonica', 'grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus', 'killer whale, killer, orca, grampus, sea wolf, Orcinus orca', 'sea lion', 'Chihuahua', 'Shih-Tzu', 'Afghan hound, Afghan', 'basset, basset hound', 'beagle', 'bloodhound, sleuthhound', 'Italian greyhound', 'whippet', 'Weimaraner', 'Yorkshire terrier', 'Boston bull, Boston terrier', 'Scotch terrier, Scottish terrier, Scottie', 'West Highland white terrier', 'golden retriever', 'Labrador retriever', 'cocker spaniel, English cocker spaniel, cocker', 'collie', 'Border collie', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'boxer', 'French bulldog', 'Saint Bernard, St Bernard', 'Siberian husky', 'dalmatian, coach dog, carriage dog', 'pug, pug-dog', 'Pomeranian', 'chow, chow chow', 'Pembroke, Pembroke Welsh corgi', 'toy poodle', 'standard poodle', 'timber wolf, grey wolf, gray wolf, Canis lupus', 'hyena, hyaena', 'red fox, Vulpes vulpes', 'tabby, tabby cat', 'leopard, Panthera pardus', 'snow leopard, ounce, Panthera uncia', 'lion, king of beasts, Panthera leo', 'tiger, Panthera tigris', 'cheetah, chetah, Acinonyx jubatus', 'ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus', 'meerkat, mierkat', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'cockroach, roach', 'mantis, mantid', 'dragonfly, darning needle, devil’s darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'starfish, sea star', 'wood rabbit, cottontail, cottontail rabbit', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'beaver', 'guinea pig, Cavia cobaya', 'zebra', 'hog, pig, grunter, squealer, Sus scrofa', 'hippopotamus, hippo, river horse, Hippopotamus amphibius', 'bison', 'gazelle', 'llama', 'skunk, polecat, wood pussy', 'badger', 'orangutan, orang, orangutang, Pongo pygmaeus', 'gorilla, Gorilla gorilla', 'chimpanzee, chimp, Pan troglodytes', 'gibbon, Hylobates lar', 'baboon', 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 'eel', 'anemone fish', 'puffer, pufferfish, blowfish, globefish', 'accordion, piano accordion, squeeze box', 'ambulance', 'assault rifle, assault gun', 'backpack, back pack, knapsack, packsack, rucksack, haversack', 'barn', 'barrow, garden cart, lawn cart, wheelbarrow', 'basketball', 'bathtub, bathing tub, bath, tub', 'beacon, lighthouse, beacon light, pharos', 'beer glass', 'binoculars, field glasses, opera glasses', 'birdhouse', 'bow tie, bow-tie, bowtie', 'broom', 'bucket, pail', 'caldron, cauldron', 'candle, taper, wax light', 'cannon', 'canoe', 'carousel, carrousel, merry-go-round, roundabout, whirligig', 'castle', 'cellular telephone, cellular phone, cellphone, cell, mobile phone', 'cowboy hat, ten-gallon hat', 'electric guitar', 'fire engine, fire truck', 'flute, transverse flute', 'gasmask, respirator, gas helmet', 'grand piano, grand', 'guillotine', 'hammer', 'harmonica, mouth organ, harp, mouth harp', 'harp', 'hatchet', 'jeep, landrover', 'joystick', 'lab coat, laboratory coat', 'lawn mower, mower', 'lipstick, lip rouge', 'mailbox, letter box', 'missile', 'mitten', 'parachute, chute', 'pickup, pickup truck', 'pirate, pirate ship', 'revolver, six-gun, six-shooter', 'rugby ball', 'sandal', 'sax, saxophone', 'school bus', 'schooner', 'shield, buckler', 'soccer ball', 'space shuttle', 'spider web, spider’s web', 'steam locomotive', 'stole', 'submarine, pigboat, sub, U-boat', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'tennis ball', 'tractor', 'trombone', 'vase', 'violin, fiddle', 'warplane, military plane', 'wine bottle', 'ice cream, icecream', 'bagel, beigel', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'head cabbage', 'broccoli', 'cucumber, cuke', 'bell pepper', 'mushroom', 'Granny Smith', 'strawberry', 'lemon', 'pineapple, ananas', 'banana', 'pomegranate', 'pizza, pizza pie', 'burrito', 'espresso', 'volcano', 'ballplayer, baseball player', 'scuba diver', 'acorn']

_IMAGENETA_CLASS_NAMES = ['stingray', 'goldfinch, Carduelis carduelis', 'junco, snowbird', 'robin, American robin, Turdus migratorius', 'jay', 'bald eagle, American eagle, Haliaeetus leucocephalus', 'vulture', 'eft', 'bullfrog, Rana catesbeiana', 'box turtle, box tortoise', 'common iguana, iguana, Iguana iguana', 'agama', 'African chameleon, Chamaeleo chamaeleon', 'American alligator, Alligator mississipiensis', 'garter snake, grass snake', 'harvestman, daddy longlegs, Phalangium opilio', 'scorpion', 'tarantula', 'centipede', 'sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita', 'lorikeet', 'hummingbird', 'toucan', 'drake', 'goose', 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus', 'jellyfish', 'sea anemone, anemone', 'flatworm, platyhelminth', 'snail', 'crayfish, crawfish, crawdad, crawdaddy', 'hermit crab', 'flamingo', 'American egret, great white heron, Egretta albus', 'oystercatcher, oyster catcher', 'pelican', 'sea lion', 'Chihuahua', 'golden retriever', 'Rottweiler', 'German shepherd, German shepherd dog, German police dog, alsatian', 'pug, pug-dog', 'red fox, Vulpes vulpes', 'Persian cat', 'lynx, catamount', 'lion, king of beasts, Panthera leo', 'American black bear, black bear, Ursus americanus, Euarctos americanus', 'mongoose', 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'rhinoceros beetle', 'weevil', 'fly', 'bee', 'ant, emmet, pismire', 'grasshopper, hopper', 'walking stick, walkingstick, stick insect', 'cockroach, roach', 'mantis, mantid', 'leafhopper', 'dragonfly, darning needle, devil’s darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk', 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus', 'cabbage butterfly', 'lycaenid, lycaenid butterfly', 'starfish, sea star', 'wood rabbit, cottontail, cottontail rabbit', 'porcupine, hedgehog', 'fox squirrel, eastern fox squirrel, Sciurus niger', 'marmot', 'bison', 'skunk, polecat, wood pussy', 'armadillo', 'baboon', 'capuchin, ringtail, Cebus capucinus', 'African elephant, Loxodonta africana', 'puffer, pufferfish, blowfish, globefish', 'academic gown, academic robe, judge’s robe', 'accordion, piano accordion, squeeze box', 'acoustic guitar', 'airliner', 'ambulance', 'apron', 'balance beam, beam', 'balloon', 'banjo', 'barn', 'barrow, garden cart, lawn cart, wheelbarrow', 'basketball', 'beacon, lighthouse, beacon light, pharos', 'beaker', 'bikini, two-piece', 'bow', 'bow tie, bow-tie, bowtie', 'breastplate, aegis, egis', 'broom', 'candle, taper, wax light', 'canoe', 'castle', 'cello, violoncello', 'chain', 'chest', 'Christmas stocking', 'cowboy boot', 'cradle', 'dial telephone, dial phone', 'digital clock', 'doormat, welcome mat', 'drumstick', 'dumbbell', 'envelope', 'feather boa, boa', 'flagpole, flagstaff', 'forklift', 'fountain', 'garbage truck, dustcart', 'goblet', 'go-kart', 'golfcart, golf cart', 'grand piano, grand', 'hand blower, blow dryer, blow drier, hair dryer, hair drier', 'iron, smoothing iron', 'jack-o’-lantern', 'jeep, landrover', 'kimono', 'lighter, light, igniter, ignitor', 'limousine, limo', 'manhole cover', 'maraca', 'marimba, xylophone', 'mask', 'mitten', 'mosque', 'nail', 'obelisk', 'ocarina, sweet potato', 'organ, pipe organ', 'parachute, chute', 'parking meter', 'piggy bank, penny bank', 'pool table, billiard table, snooker table', 'puck, hockey puck', 'quill, quill pen', 'racket, racquet', 'reel', 'revolver, six-gun, six-shooter', 'rocking chair, rocker', 'rugby ball', 'saltshaker, salt shaker', 'sandal', 'sax, saxophone', 'school bus', 'schooner', 'sewing machine', 'shovel', 'sleeping bag', 'snowmobile', 'snowplow, snowplough', 'soap dispenser', 'spatula', 'spider web, spider’s web', 'steam locomotive', 'stethoscope', 'studio couch, day bed', 'submarine, pigboat, sub, U-boat', 'sundial', 'suspension bridge', 'syringe', 'tank, army tank, armored combat vehicle, armoured combat vehicle', 'teddy, teddy bear', 'toaster', 'torch', 'tricycle, trike, velocipede', 'umbrella', 'unicycle, monocycle', 'viaduct', 'volleyball', 'washer, automatic washer, washing machine', 'water tower', 'wine bottle', 'wreck', 'guacamole', 'pretzel', 'cheeseburger', 'hotdog, hot dog, red hot', 'broccoli', 'cucumber, cuke', 'bell pepper', 'mushroom', 'lemon', 'banana', 'custard apple', 'pomegranate', 'carbonara', 'bubble', 'cliff, drop, drop-off', 'volcano', 'ballplayer, baseball player', 'rapeseed', 'yellow lady’s slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum', 'corn', 'acorn']



_DATASET_CLASS_NAMES = {
    "cifar100": _CIFAR100_CLASS_NAMES,
    "cub200": _CUB_CLASS_NAMES,
    "imagenet-r": _IMAGENETR_CLASS_NAMES,
    "imagenet-a": _IMAGENETA_CLASS_NAMES,
    "omnibenchmark": None,
    "vtab": None,
    "objectnet": None,
}