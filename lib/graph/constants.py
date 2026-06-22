# >>> by source


GRAPHLAND_DATASETS = [
    "hm-categories",
    "pokec-regions",
    "web-topics",
    "tolokers-2",
    "city-reviews",
    "artnet-exp",
    "web-fraud",
    "hm-prices",
    "avazu-ctr",
    "city-roads-M",
    "city-roads-L",
    "twitch-views",
    "artnet-views",
    "web-traffic",
]

PYG_DATASETS = [
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "tolokers",
    "questions",
    "cora",
    "citeseer",
    "pubmed",
    "coauthor-cs",
    "coauthor-physics",
    "amazon-computers",
    "amazon-photo",
    "lastfm-asia",
    "facebook",
    "flickr",
    "wiki-cs",
]

OGB_DATASETS = [
    "ogbn-arxiv",
    "ogbn-products",
]

OTHER_DATASETS = [
    "amazon-ratings-full",
]

DATASETS = [
    *GRAPHLAND_DATASETS,
    *PYG_DATASETS,
    *OGB_DATASETS,
    *OTHER_DATASETS,
]


# >>> by task


MULTICLASS_DATASETS = [
    "roman-empire",
    "amazon-ratings",
    "cora",
    "citeseer",
    "pubmed",
    "coauthor-cs",
    "coauthor-physics",
    "amazon-computers",
    "amazon-photo",
    "lastfm-asia",
    "facebook",
    "flickr",
    "wiki-cs",
    "ogbn-arxiv",
    "ogbn-products",
    "hm-categories",
    "pokec-regions",
    "web-topics",
    "amazon-ratings-full",
]

BINCLASS_DATASETS = [
    "minesweeper",
    "tolokers",
    "questions",
    "tolokers-2",
    "city-reviews",
    "artnet-exp",
    "web-fraud",
]

REGRESSION_DATASETS = [
    "hm-prices",
    "avazu-ctr",
    "city-roads-M",
    "city-roads-L",
    "twitch-views",
    "artnet-views",
    "web-traffic",
]

assert (
    len(MULTICLASS_DATASETS) +
    len(BINCLASS_DATASETS) +
    len(REGRESSION_DATASETS)
) == len(DATASETS)  # fmt: skip


# >>> by features


HETEROPHILOUS_DATASETS = [
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "tolokers",
    "questions",
]

PREDEFINED_SPLIT_DATASETS = [
    *HETEROPHILOUS_DATASETS,
    *GRAPHLAND_DATASETS,
    *OTHER_DATASETS,
]

HOMOGENEOUS_DATASETS = [
    *PYG_DATASETS,
    *OGB_DATASETS,
    *OTHER_DATASETS,
]

HETEROGENEOUS_DATASETS = [
    *GRAPHLAND_DATASETS,
]


# >>> project-specific


GFM_GRAPHLAND_DATASETS = [
    "artnet-exp",
    "city-reviews",
    "tolokers-2",
    "artnet-views",
    "avazu-ctr",
    "city-roads-M",
    "hm-prices",
    "twitch-views",
]

GFM_CLASSIC_DATASETS = [
    "amazon-ratings",
    "facebook",
    "pubmed",
    "questions",
    "wiki-cs",
]

GFM_GRAPHLAND_DATASETS_V2 = [
    "artnet-exp",
    "city-reviews",
    "tolokers-2",
    "artnet-views",
    "avazu-ctr",
    "city-roads-M",
    "city-roads-L",
    "hm-prices",
    "hm-categories",
    "twitch-views",
]

GFM_CLASSIC_DATASETS_V2 = [
    "cora",
    "citeseer",
    "pubmed",
    "coauthor-cs",
    "coauthor-physics",
    "amazon-computers",
    "amazon-photo",
    "lastfm-asia",
    "facebook",
    "flickr",
    "roman-empire",
    "amazon-ratings",
    "questions",
    "wiki-cs",
]

GFM_ECOC_DATASETS = [
    "hm-categories",
    "lastfm-asia",
    "coauthor-cs",
    "roman-empire",
]
