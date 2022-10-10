from itertools import chain
from graph_generator.generator import generate_graph, Mode

def main(seed=888):

    # configs = [
    #     dict(name="small", size=10, lateral_connections=5, num_flags=5, add_defense=True, mode="random", mu=0),
    #     dict(name="small_no_defense", size=10, lateral_connections=5, num_flags=5, add_defense=False, mode="random", mu=0),
    #     dict(name="medium", size=100, lateral_connections=25, num_flags=10,),
    #     dict(name="big", size=500, lateral_connections=20, num_flags=20, add_defense=True, mode="random", mu=0),
    #     dict(name="big_no_defense", size=500, lateral_connections=20, num_flags=20, add_defense=False, mode="random", mu=0),
    # ]

    generate = lambda x: (
        dict(
            name=f"graph_{nodes}" if x else f"graph_{nodes}_no_defense",
            size=nodes,
            lateral_connections = nodes // 10,
            num_flags=nodes // 10,
            add_defense=x,
            mode=Mode.DEGREE,
            mu=2,
        )
        for nodes in [50, 100] #in np.arange(0, 500, 50) + 50
    )

    for config in chain(generate(True), generate(False)):
        generate_graph(**config, seed=seed)


if __name__ == "__main__":
    main()
