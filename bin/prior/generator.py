import os
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
from graph_tool import all as gt
from joblib import Parallel, delayed


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--num_graphs', type=int, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=1)

    args = parser.parse_args()

    return args


def main():
    args = get_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    os.makedirs(args.save_dir)

    parallel = Parallel(n_jobs=args.num_workers)
    delayed_fn = delayed(generate_and_save_graph)
    parallel(delayed_fn(config=config, graph_id=i, save_dir=args.save_dir) for i in tqdm(range(1, args.num_graphs + 1)))


def random_float(low, high):
    return np.random.random() * (high - low) + low


def generate_and_save_graph(config, graph_id, save_dir):
    graph_gt, first_level_group_ids, second_level_group_ids = generate_graph(config)
    edgelist = graph_gt.get_edges()

    if second_level_group_ids is not None:
        categorical_features = np.stack([first_level_group_ids, second_level_group_ids], axis=1)
    else:
        categorical_features = first_level_group_ids[:, None]

    save_dir = f'{save_dir}/graph_{graph_id:08d}'
    os.makedirs(save_dir)
    np.save(f'{save_dir}/edgelist.npy', edgelist)
    np.save(f'{save_dir}/categorical_features.npy', categorical_features)


def generate_graph(config):
    if config['max_first_level_sbm_graphs'] == 1:
        graph, first_level_group_ids = generate_graph_from_sbm(min_nodes=config['min_sbm_nodes'],
                                                               max_nodes=config['max_sbm_nodes'],
                                                               min_groups=config['min_sbm_groups'],
                                                               max_groups=config['max_sbm_groups'],
                                                               min_group_size=config['min_sbm_group_size'],
                                                               max_group_size=config['max_sbm_group_size'],
                                                               divisor=config['sbm_divisor'])

        second_level_group_ids = None

    else:
        num_first_level_sbm_graphs = np.random.randint(low=max(2, config['min_first_level_sbm_graphs']),
                                                       high=config['max_first_level_sbm_graphs'] + 1)

        first_level_sbm_graphs, first_level_group_ids_list = zip(*[
            generate_graph_from_sbm(min_nodes=config['min_sbm_nodes'],
                                    max_nodes=config['max_sbm_nodes'],
                                    min_groups=config['min_sbm_groups'],
                                    max_groups=config['max_sbm_groups'],
                                    min_group_size=config['min_sbm_group_size'],
                                    max_group_size=config['max_sbm_group_size'],
                                    divisor=config['sbm_divisor'])
            for _ in range(num_first_level_sbm_graphs)
        ])

        num_nodes = sum(graph.num_vertices() for graph in first_level_sbm_graphs)
        second_level_sbm_graph, second_level_group_ids = generate_graph_from_sbm(
            min_nodes=num_nodes,
            max_nodes=num_nodes,
            min_groups=config['min_sbm_groups'] * num_first_level_sbm_graphs,
            max_groups=config['max_sbm_groups'] * num_first_level_sbm_graphs,
            min_group_size=config['min_sbm_group_size'],
            max_group_size=config['max_sbm_group_size'],
            divisor=config['sbm_divisor']
        )

        graph, first_level_group_ids, second_level_group_ids = merge_sbm_graphs(
            first_level_sbm_graphs=first_level_sbm_graphs,
            first_level_group_ids_list=first_level_group_ids_list,
            second_level_sbm_graph=second_level_sbm_graph,
            second_level_group_ids=second_level_group_ids
        )

    graph, first_level_group_ids, second_level_group_ids = ba_process(graph=graph,
                                                                      first_level_group_ids=first_level_group_ids,
                                                                      second_level_group_ids=second_level_group_ids,
                                                                      min_nodes=config['min_ba_nodes'],
                                                                      max_nodes=config['max_ba_nodes'],
                                                                      max_deg=config['max_ba_deg'])

    gt.remove_parallel_edges(graph)
    gt.remove_self_loops(graph)

    comp = gt.extract_largest_component(graph, prune=False)
    node_idx = comp.get_vertices()
    graph = gt.extract_largest_component(graph, prune=True)
    first_level_group_ids = first_level_group_ids[node_idx]
    if second_level_group_ids is not None:
        second_level_group_ids = second_level_group_ids[node_idx]

    return graph, first_level_group_ids, second_level_group_ids


def generate_graph_from_sbm(min_nodes, max_nodes, min_groups, max_groups, min_group_size, max_group_size, divisor):
    num_nodes = np.random.randint(low=min_nodes, high=max_nodes + 1)
    num_groups = np.random.randint(low=min_groups, high=max_groups + 1)
    group_sizes = get_sbm_group_sizes(num_nodes=num_nodes,
                                      num_groups=num_groups,
                                      min_group_size=min_group_size,
                                      max_group_size=max_group_size)

    group_memberships = np.concatenate(
        [np.full(shape=group_size, fill_value=i) for i, group_size in enumerate(group_sizes)], axis=0
    )

    edge_propensities, degs = get_edge_propensities_and_degs(group_sizes=group_sizes, divisor=divisor)

    graph = gt.generate_sbm(b=group_memberships,
                            probs=edge_propensities,
                            out_degs=degs / 2,
                            in_degs=degs / 2,
                            directed=False)

    return graph, group_memberships


def get_sbm_group_sizes(num_nodes, num_groups, min_group_size, max_group_size):
    num_nodes_left = num_nodes - num_groups * min_group_size
    group_sizes = []
    for _ in range(num_groups - 1):
        num_additional_nodes = np.random.randint(low=0, high=min(num_nodes_left, max_group_size - min_group_size))
        group_sizes.append(min_group_size + num_additional_nodes)
        num_nodes_left -= num_additional_nodes

    group_sizes.append(min_group_size + num_nodes_left)

    return group_sizes


def get_edge_propensities_and_degs(group_sizes, divisor):
    group_sizes = np.array(group_sizes)
    num_groups = len(group_sizes)
    edge_propensities = np.zeros((num_groups, num_groups), dtype=np.float64)

    for i, group_size in enumerate(group_sizes):
        mean_deg = random_float(low=2, high=min(20, group_size // 2))

        mean_intra_deg = random_float(low=max(1, 0.25 * mean_deg), high=0.9 * mean_deg)

        mean_degs = (edge_propensities[:i, i] / group_sizes[:i]).tolist()
        mean_degs.append(mean_intra_deg)

        deg_left = mean_deg - sum(mean_degs)
        for _ in range(i + 1, num_groups):
            deg = random_float(low=0, high=0.8 * deg_left)
            mean_degs.append(deg)
            deg_left -= deg

        mean_degs[i] += deg_left

        mean_degs = np.array(mean_degs)

        edge_propensities[i] = mean_degs * group_size
        edge_propensities[i, i] *= 2

    degs = []
    for i, group_size in enumerate(group_sizes):
        mean_degs = edge_propensities[i] / group_size
        mean_degs[i] /= 2
        mean_deg = mean_degs.sum()

        subgroup_size = group_size // 3
        deg_fracs_1 = np.random.random(size=subgroup_size) * mean_deg
        deg_fracs_2 = np.random.random(size=subgroup_size) * mean_deg

        cur_degs = np.concatenate([
            np.full(shape=subgroup_size, fill_value=mean_deg) + (deg_fracs_1 + deg_fracs_2) * mean_deg,
            np.full(shape=subgroup_size, fill_value=mean_deg) - deg_fracs_1 * mean_deg,
            np.full(shape=subgroup_size, fill_value=mean_deg) - deg_fracs_2 * mean_deg,
            np.full(shape=group_size - 3 * subgroup_size, fill_value=mean_deg)
        ], axis=0)

        for j in range(np.random.randint(low=0, high=int(0.02 * group_size) + 1)):
            additional_deg = random_float(low=0.05 * group_size, high=min(250, 0.4 * group_size))
            cur_degs[j] += additional_deg
            cur_degs += additional_deg / group_size
            edge_propensities[i, i] += additional_deg * 2

        degs.append(cur_degs)

    degs = np.concatenate(degs, axis=0)

    edge_propensities /= divisor
    degs /= divisor

    return edge_propensities, degs


def merge_sbm_graphs(first_level_sbm_graphs, first_level_group_ids_list, second_level_sbm_graph,
                     second_level_group_ids):
    first_level_sbm_graph = first_level_sbm_graphs[0]
    for graph in first_level_sbm_graphs[1:]:
        first_level_sbm_graph = gt.graph_merge(g1=first_level_sbm_graph, g2=graph, vmap=None, in_place=True)

    first_level_group_ids = first_level_group_ids_list[0]
    for group_ids in first_level_group_ids_list[1:]:
        group_ids += first_level_group_ids.max() + 1
        first_level_group_ids = np.concatenate([first_level_group_ids, group_ids], axis=0)

    num_first_level_groups = first_level_group_ids.max() + 1
    num_second_level_groups = second_level_group_ids.max() + 1

    group_to_nodes = {
        group_id: np.where(first_level_group_ids == group_id)[0] for group_id in range(num_first_level_groups)
    }
    for group_id in group_to_nodes.keys():
        np.random.shuffle(group_to_nodes[group_id])

    node_map = []
    for second_level_group_id in range(num_second_level_groups):
        second_level_group_size = (second_level_group_ids == second_level_group_id).sum()
        num_subgroups = np.random.randint(low=2, high=4)
        if num_subgroups == 2:
            bound = np.random.randint(low=0, high=second_level_group_size + 1)
            subgroup_sizes = [bound, second_level_group_size - bound]
        else:
            bound_1, bound_2 = np.random.randint(low=0, high=second_level_group_size + 1, size=2)
            if bound_1 > bound_2:
                bound_1, bound_2 = bound_2, bound_1

            subgroup_sizes = [bound_1, bound_2 - bound_1, second_level_group_size - bound_2]

        for i in range(len(subgroup_sizes)):
            subgroup_size = subgroup_sizes[i]
            random_group_ids = np.arange(num_first_level_groups)
            np.random.shuffle(random_group_ids)
            j = 0
            while subgroup_size > 0:
                group_id = random_group_ids[j]
                num_nodes_to_take = min(subgroup_size, len(group_to_nodes[group_id]))
                node_map.append(group_to_nodes[group_id][:num_nodes_to_take])
                group_to_nodes[group_id] = group_to_nodes[group_id][num_nodes_to_take:]
                subgroup_size -= num_nodes_to_take
                j += 1

    node_map = np.concatenate(node_map, axis=0)

    node_map = second_level_sbm_graph.new_vertex_property(value_type='long', vals=node_map)
    graph = gt.graph_merge(g1=first_level_sbm_graph, g2=second_level_sbm_graph, vmap=node_map, in_place=True)

    node_map_reversed = np.argsort(node_map)
    second_level_group_ids = second_level_group_ids[node_map_reversed]

    return graph, first_level_group_ids, second_level_group_ids


def ba_process(graph, first_level_group_ids, second_level_group_ids, min_nodes, max_nodes, max_deg):
    num_runs = np.random.randint(low=min_nodes // 25, high=max_nodes // 25)
    for _ in range(num_runs):
        deg = np.random.randint(low=1, high=max_deg + 1)
        graph = gt.price_network(seed_graph=graph, N=25, m=deg, c=1, directed=False)

    num_new_nodes = num_runs * 25
    first_level_group_ids = np.concatenate(
        [first_level_group_ids, np.full(shape=num_new_nodes, fill_value=-1)], axis=0
    )
    if second_level_group_ids is not None:
        second_level_group_ids = np.concatenate(
            [second_level_group_ids, np.full(shape=num_new_nodes, fill_value=-1)], axis=0
        )

    return graph, first_level_group_ids, second_level_group_ids


if __name__ == '__main__':
    main()
