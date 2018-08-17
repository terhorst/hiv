from intervaltree import IntervalTree, Interval
import numpy as np


def tree_seq_with_migrations(tree_seq):
    itv = IntervalTree(
        Interval(begin=m.left, end=m.right, data=m) for m in tree_seq.migrations()
    )
    for t in tree_seq.trees():
        t_start, t_end = t.interval
        s = set(
            [
                x
                for be in itv[t_start:t_end]
                for x in (be.begin, be.end)
                if t_start <= x < t_end
            ]
        ) | {t_start, t_end}
        bp = sorted(s)
        for start, end in zip(bp[:-1], bp[1:]):
            md = {}
            for m in itv[start:end]:
                md.setdefault(m.data.node, []).append(
                    (m.data.time, m.data.dest, m.data.source)
                )
            yield (start, end, t, md)


def weighted_segments(tree, migrations, rates):
    ret = []
    for node in tree.nodes():
        if node == tree.root:
            continue
        times = [(0, tree.population(node))]
        t0 = tree.time(node)
        for m in sorted(migrations.get(node, [])):
            # print(m)
            assert times[-1][1] == m[2]  # source pop = last pop
            times.append((m[0] - t0, m[1]))
        times.append((tree.branch_length(node), tree.population(tree.parent(node))))
        # print(node,times)
        for (t0, p0), (t1, _) in zip(times[:-1], times[1:]):
            ret.append((node, (t1 - t0) * rates[p0]))
    return np.array(ret)


def drop_mutations(tree_seq, rates):
    "Drop mutations on :tree_seq: using population-specific :rates:"

    for start, end, tree, migrations in tree_seq_with_migrations(tree_seq):
        ws = weighted_segments(tree, migrations, rates)
        k = np.random.poisson(ws[:, 1].sum() * (end - start))
        nodes = np.random.choice(
            ws[:, 0].astype(int), size=k, p=ws[:, 1] / ws[:, 1].sum(), replace=True
        )
        n = tree.get_num_samples()
        yield from sorted(
            [
                (np.random.uniform(*(tree.interval)), tree.get_num_samples(node))
                for node in nodes
            ]
        )
